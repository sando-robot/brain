"""运行：python brain/forward.py。使用本地 tokenizer，不下载或加载模型权重。"""

from pathlib import Path

import torch
from transformers import AutoTokenizer

from omni.model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def trace_shape(name):
    """forward hook 只观察数据，不修改模型的计算结果。"""
    def hook(module, inputs, output):
        # Attention / Block 返回 (hidden_states, KV cache)。
        tensor = output[0] if isinstance(output, tuple) else output
        print(f"  {name:40s} {tuple(inputs[0].shape)} -> {tuple(tensor.shape)}")

    return hook


def main():
    torch.manual_seed(42)
    text = "hello,world"
    tokenizer_path = Path(__file__).resolve().parent / "omni" / "model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]

    # 缩小层数和维度，方便在 CPU 上观察；模块实现仍来自 model_minimind.py。
    config = MiniMindConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        dropout=0.0,
        flash_attn=False,
    )
    # 构造模型即随机初始化；没有调用 from_pretrained 或 load_state_dict。
    model = MiniMindForCausalLM(config).eval()
    backbone = model.model  # MiniMindModel：embedding + Transformer blocks + RMSNorm

    print(f"原文: {text!r}")
    print(f"tokens: {tokenizer.convert_ids_to_tokens(input_ids[0].tolist())}")
    print(f"input_ids: {input_ids.tolist()}, shape={tuple(input_ids.shape)}")
    print("维度约定：[batch, sequence, hidden]；logits 最后一维为 vocab_size。")

    with torch.inference_mode():
        print("\n1. 单独进入 embedding 层：token ID -> 向量")
        embeddings = backbone.embed_tokens(input_ids)
        print(f"embedding weight: {tuple(backbone.embed_tokens.weight.shape)}")
        print(f"embeddings: {tuple(embeddings.shape)}")
        print(f"第一个 token 的前 8 维: {embeddings[0, 0, :8].tolist()}")

        print("\n2. 进入模型主体 MiniMindModel，跟踪真实 forward 路径")
        print("主体接收 input_ids，内部会再次调用 embedding，不能直接传 embeddings。")
        print("每个 Block：RMSNorm -> Attention（Q/K 使用 RoPE）-> 残差相加")
        print("            -> RMSNorm -> FeedForward（SwiGLU）-> 残差相加")
        handles = []
        modules = [("embed_tokens", backbone.embed_tokens), ("dropout", backbone.dropout)]
        for i, layer in enumerate(backbone.layers):
            for name in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp"):
                modules.append((f"layers.{i}.{name}", getattr(layer, name)))
            modules.append((f"layers.{i} (含两次残差相加)", layer))
        modules.append(("final RMSNorm", backbone.norm))
        try:
            for name, module in modules:
                handles.append(module.register_forward_hook(trace_shape(name)))
            hidden_states, past_key_values, aux_loss = backbone(input_ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        print(f"主体输出 hidden_states: {tuple(hidden_states.shape)}")
        print(f"KV cache（未启用）: {past_key_values}, aux_loss: {aux_loss.item()}")

        print("\n3. 语言模型头：hidden_states -> 每个位置的词表分数")
        logits = model.lm_head(hidden_states)
        print(f"lm_head: {tuple(hidden_states.shape)} -> {tuple(logits.shape)}")
        print("权重随机初始化，因此这些分数没有训练后的语言意义。")
        print("完整调用 model(input_ids) 会依次执行上述主体和 lm_head。")


if __name__ == "__main__":
    main()
