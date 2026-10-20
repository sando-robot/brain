import sys
import torch
import torchvision
import torchaudio
import brian2 as b2

print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('Brian2:', b2.__version__)
assert torch.cuda.is_available(), 'CUDA unavailable'
print('GPU:', torch.cuda.get_device_name(0))
x = torch.randn(32, 32, device='cuda', requires_grad=True)
(x @ x.T).square().mean().backward()
torch.cuda.synchronize()
print('CUDA forward/backward: OK')

b2.prefs.codegen.target = 'numpy'
b2.start_scope()
neurons = b2.NeuronGroup(2, 'dv/dt = (2-v)/(10*ms) : 1',
                        threshold='v>1', reset='v=0', method='exact')
spikes = b2.SpikeMonitor(neurons)
b2.run(30*b2.ms)
assert spikes.num_spikes > 0
print('Brian2 LIF simulation:', spikes.num_spikes, 'spikes')

sys.path.insert(0, '/home/sophda/sando-robot')
from brain.omni.model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from brain.omni.model.model_omni import OmniConfig, MiniMindOmni
config = MiniMindConfig(hidden_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2,
                      vocab_size=128, max_position_embeddings=32)
model = MiniMindForCausalLM(config).cuda()
tokens = torch.randint(0, 128, (1, 8), device='cuda')
out = model(tokens, labels=tokens)
out.loss.backward()
assert out.logits.shape == (1, 8, 128)
print('MiniMind CUDA forward/backward and Omni import: OK')