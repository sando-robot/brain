"""Brian2 (fixed SNN) -> MiniMind -> synchronized waveform + order label.

Synthetic proof-of-concept, not a speech or general audio generation model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from omni.model.model_minimind import MiniMindConfig, MiniMindModel

SR = 8000
FRAMES = 40
SAMPLES_PER_FRAME = 80
N_AUDIO = FRAMES * SAMPLES_PER_FRAME
N_NEURONS = 128
FEATURE_BANDS = 32


def generate_audio(n, seed=123):
    """A 220 Hz->440 Hz or B 440 Hz->220 Hz signal, exact aligned pairs."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_AUDIO, dtype=np.float32) / SR
    data = np.zeros((n, N_AUDIO), dtype=np.float32)
    labels = np.arange(n, dtype=np.int64) % 2
    rng.shuffle(labels)
    for k, label in enumerate(labels):
        f1, f2 = (220., 440.) if label == 0 else (440., 220.)
        # Fixed phase provides a learnable demonstration; frequency, gain and transition vary.
        boundary = int(rng.integers(1400, 1800))
        f = np.where(np.arange(N_AUDIO) < boundary, f1, f2).astype(np.float32)
        phase = 2 * np.pi * np.cumsum(f) / SR
        gain = rng.uniform(0.55, 0.9)
        env = np.minimum(1., np.arange(N_AUDIO) / 80.) * np.minimum(
            1., (N_AUDIO - 1 - np.arange(N_AUDIO)) / 80.
        )
        data[k] = gain * env * np.sin(phase)
    return data, labels


def spectral_features(wav):
    """40x32 frame FFT features, bounded and deterministic."""
    frames = wav.reshape(FRAMES, SAMPLES_PER_FRAME)
    magnitude = np.abs(np.fft.rfft(frames * np.hanning(SAMPLES_PER_FRAME), n=256, axis=-1))
    freqs = np.fft.rfftfreq(256, 1 / SR)
    centers = np.geomspace(120, 1400, FEATURE_BANDS)
    feat = np.empty((FRAMES, FEATURE_BANDS), dtype=np.float32)
    for j, center in enumerate(centers):
        bw = max(100., center * 0.27)
        weights = np.maximum(0, 1 - np.abs(freqs - center) / bw)
        weights /= max(weights.sum(), 1e-8)
        feat[:, j] = magnitude @ weights
    feat = np.log1p(feat)
    feat = feat / max(float(np.max(feat)), 1e-6)
    return feat


def get_snn_encoder():
    """Import Brian2 only for feature extraction; torch-only smoke test stays usable."""
    import brian2 as b
    b.prefs.codegen.target = 'numpy'  # avoids toolchain prerequisite on first run
    rng = np.random.default_rng(7)
    W = rng.normal(0.06, 0.24, size=(FEATURE_BANDS, N_NEURONS)).astype(np.float32)
    bias = rng.uniform(0.45, 0.85, size=N_NEURONS).astype(np.float32)

    def encode(wav):
        spec = spectral_features(wav)
        # Fixed projection; no gradients, STDP, or parameter updates.
        drive = np.clip(spec @ W + bias, 0, 2.8).astype(np.float64)
        b.start_scope()
        b.defaultclock.dt = 1 * b.ms
        input_current = b.TimedArray(drive, dt=10*b.ms)
        neurons = b.NeuronGroup(
            N_NEURONS,
            'dv/dt = (input_current(t, i) - v)/(12*ms) : 1',
            threshold='v > 1.0',
            reset='v = 0',
            refractory=2*b.ms,
            method='euler',
        )
        neurons.v = 0
        states = b.StateMonitor(neurons, 'v', record=True)
        spikes = b.SpikeMonitor(neurons)
        b.run(400*b.ms)
        state_values = np.asarray(states.v, dtype=np.float32)
        # StateMonitor uses defaultclock=1 ms: [128,400] -> [40,128] mean voltage
        if state_values.shape != (N_NEURONS, FRAMES * 10):
            raise RuntimeError(f'Unexpected Brian2 monitor shape: {state_values.shape}')
        voltage = state_values.reshape(N_NEURONS, FRAMES, 10).mean(axis=-1).T
        counts = np.zeros((FRAMES, N_NEURONS), dtype=np.float32)
        ti = np.floor(np.asarray(spikes.t / b.ms) / 10).astype(np.int64)
        ni = np.asarray(spikes.i, dtype=np.int64)
        mask = (ti >= 0) & (ti < FRAMES)
        np.add.at(counts, (ti[mask], ni[mask]), 1)
        # Normalize counts to [0,1] typical scale, voltages in dimensionless LIF model.
        return np.concatenate([counts / 10., voltage], axis=-1).astype(np.float32)
    return encode


def cache_dataset(out, n_train=128, n_val=32, seed=123):
    out.mkdir(parents=True, exist_ok=True)
    encoder = get_snn_encoder()
    for split, num, offset in [('train', n_train, 0), ('val', n_val, 10000)]:
        audios, labels = generate_audio(num, seed + offset)
        feats = np.stack([encoder(w) for w in audios])
        np.savez_compressed(out/f'{split}.npz', features=feats, audio=audios, labels=labels)
        print(f'cached {split}: X={feats.shape}, audio={audios.shape}')
        sf.write(out/f'{split}_input.wav', audios[0], SR)
    (out/'config.json').write_text(json.dumps({
        'sample_rate': SR, 'frames': FRAMES, 'samples_per_frame': SAMPLES_PER_FRAME,
        'neurons': N_NEURONS, 'output_features': N_NEURONS*2,
        'description': 'paired input waveform and target same waveform (autoencoding)'
    }, indent=2), encoding='utf-8')


class AudioTransformer(nn.Module):
    """SNN features -> Omni MiniMind backbone -> waveform and order heads.

    The name is kept for existing callers. Token embeddings are unused: SNN
    features supply continuous embeddings directly to the original MiniMind blocks.
    """
    def __init__(self, config=None, pretrained=None):
        super().__init__()
        self.config = config or MiniMindConfig()
        hidden = self.config.hidden_size
        if self.config.max_position_embeddings < FRAMES:
            raise ValueError(f'max_position_embeddings must be >= {FRAMES}')
        self.project = nn.Sequential(nn.Linear(N_NEURONS * 2, 128), nn.GELU(),
                                     nn.Linear(128, hidden), nn.LayerNorm(hidden))
        self.backbone = MiniMindModel(self.config)
        # No vocabulary lookup is needed for continuous SNN inputs.
        self.backbone.embed_tokens = nn.Identity()
        self.wave_head = nn.Linear(hidden, SAMPLES_PER_FRAME)
        self.class_head = nn.Linear(hidden, 2)
        self.aux_loss = None
        if pretrained is not None:
            self.load_pretrained(pretrained)

    def load_pretrained(self, path):
        """Load a raw MiniMind/Omni state dict or trainer's {'model': ...}.

        Require every backbone parameter, so an incompatible checkpoint cannot
        silently leave random layers. Non-backbone Omni heads are ignored.
        Architecture options must match the source checkpoint's config.
        """
        state = torch.load(path, map_location='cpu', weights_only=True)
        for key in ('state_dict', 'model'):
            if key in state and isinstance(state[key], dict):
                state = state[key]
        expected = self.backbone.state_dict()
        weights = {}
        for key, value in state.items():
            for prefix in ('module.', '_orig_mod.'):
                key = key.removeprefix(prefix)
            for prefix in ('model.', 'thinker.', 'backbone.'):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            if key in expected:
                weights[key] = value
        # strict=True also reports shape mismatches and missing backbone layers.
        self.backbone.load_state_dict(weights, strict=True)
        print(f'loaded MiniMind backbone: {path} ({len(weights)} tensors)')

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (FRAMES, N_NEURONS * 2):
            raise ValueError(f'Expected [batch, {FRAMES}, {N_NEURONS * 2}], got {tuple(x.shape)}')
        h = self.backbone.dropout(self.project(x))
        positions = (self.backbone.freqs_cos[:x.shape[1]],
                     self.backbone.freqs_sin[:x.shape[1]])
        for layer in self.backbone.layers:
            h, _ = layer(h, positions, use_cache=False)
        h = self.backbone.norm(h)
        self.aux_loss = sum((layer.mlp.aux_loss for layer in self.backbone.layers
                             if self.config.use_moe), h.new_zeros(()))
        audio = torch.tanh(self.wave_head(h)).reshape(-1, N_AUDIO)
        logits = self.class_head(h.mean(dim=1))
        return audio, logits


def load_split(folder, split):
    with np.load(folder/f'{split}.npz') as d:
        return (torch.from_numpy(d['features'].copy()),
                torch.from_numpy(d['audio'].copy()),
                torch.from_numpy(d['labels'].copy()).long())


def train(folder, epochs=12, batch_size=16, lr=1e-3, config=None, pretrained=None):
    torch.manual_seed(42)
    X, audio, labels = load_split(folder, 'train')
    VX, Vaudio, Vlabels = load_split(folder, 'val')
    loader = DataLoader(TensorDataset(X, audio, labels), batch_size=batch_size, shuffle=True)
    model = AudioTransformer(config=config, pretrained=pretrained)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = float('inf')
    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.
        for x, y, label in loader:
            pred, logits = model(x)
            # Paired, sample-aligned losses, both heads optimized in the same update.
            waveform_loss = nn.functional.mse_loss(pred, y)
            spectral_loss = (torch.stft(pred, n_fft=256, hop_length=80, return_complex=True).abs() -
                             torch.stft(y, n_fft=256, hop_length=80, return_complex=True).abs()).abs().mean()
            classification_loss = nn.functional.cross_entropy(logits, label)
            loss = waveform_loss + 0.04 * spectral_loss + 0.15 * classification_loss + model.aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += float(loss.detach()) * len(x)
        model.eval()
        with torch.no_grad():
            pred, logits = model(VX)
            vmse = float(nn.functional.mse_loss(pred, Vaudio))
            acc = float((logits.argmax(dim=1) == Vlabels).float().mean())
        print(f'epoch {ep:02d} train_loss={tr_loss/len(X):.4f} val_wave_mse={vmse:.4f} val_accuracy={acc:.2%}')
        if vmse < best_val:
            best_val = vmse
            torch.save({'model': model.state_dict(), 'config': model.config.to_dict()}, folder/'best.pt')
    print('saved', folder/'best.pt')


def infer(folder, index=0):
    X, Y, labels = load_split(folder, 'val')
    checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
    if 'config' not in checkpoint:
        raise ValueError('best.pt has no MiniMind config; retrain to create a compatible checkpoint')
    model = AudioTransformer(config=MiniMindConfig(**checkpoint['config']))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    with torch.no_grad():
        pred, logits = model(X[index:index+1])
    folder.mkdir(exist_ok=True, parents=True)
    sf.write(folder/'generated.wav', pred[0].cpu().numpy(), SR)
    sf.write(folder/'target.wav', Y[index].numpy(), SR)
    print('true order:', 'A->B' if labels[index] == 0 else 'B->A')
    print('predicted order:', 'A->B' if logits.argmax(1).item() == 0 else 'B->A')
    print('WAV output:', folder/'generated.wav')


def smoke():
    x = torch.randn(2, FRAMES, N_NEURONS * 2)
    model = AudioTransformer(MiniMindConfig(hidden_size=64, num_hidden_layers=2,
                                            max_position_embeddings=FRAMES))
    y, cls = model(x)
    assert y.shape == (2, N_AUDIO) and cls.shape == (2, 2)
    labels = torch.tensor([0, 1]); target = torch.randn_like(y)
    loss = nn.functional.mse_loss(y, target) + nn.functional.cross_entropy(cls, labels)
    loss.backward()
    assert model.project[0].weight.grad is not None
    assert model.backbone.layers[0].self_attn.q_proj.weight.grad is not None
    assert model.wave_head.weight.grad is not None and model.class_head.weight.grad is not None
    print('smoke OK: shapes, waveform head, classification head, backward')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'train', 'infer', 'all', 'smoke'])
    parser.add_argument('--out', type=Path, default=Path('outputs'))
    parser.add_argument('--train-samples', type=int, default=128)
    parser.add_argument('--val-samples', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--minimind-config', type=Path,
                        help='MiniMind/Omni config.json matching the pretrained architecture')
    parser.add_argument('--pretrained', type=Path,
                        help='MiniMind/Omni .pth/.pt backbone weights (optional)')
    args = parser.parse_args()
    if args.command == 'smoke': smoke()
    if args.command in ('prepare', 'all'):
        cache_dataset(args.out, args.train_samples, args.val_samples)
    if args.command in ('train', 'all'):
        config = (MiniMindConfig(**json.loads(args.minimind_config.read_text(encoding='utf-8')))
                  if args.minimind_config else MiniMindConfig())
        train(args.out, epochs=args.epochs, config=config, pretrained=args.pretrained)
    if args.command in ('infer', 'all'):
        infer(args.out, args.index)

if __name__ == '__main__':
    main()
