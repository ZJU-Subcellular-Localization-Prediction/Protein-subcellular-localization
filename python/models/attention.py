"""
Bahdanau 加法注意力 (Additive Attention)

参考：原项目 utils/models.py 第 12-37 行
      Bahdanau et al. "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)

公式：
  score = V * tanh(W1 * features + W2 * hidden)
  alpha = softmax(score, dim=time)
  context = sum(alpha * features, dim=time)

其中 features 是 encoder 所有 hidden states (batch, seq_len, units)
      hidden   是 decoder 最后 hidden state (batch, units)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, units):
        super().__init__()
        self.W1 = nn.Linear(units, units, bias=False)   # 作用于 encoder states
        self.W2 = nn.Linear(units, units, bias=False)   # 作用于 decoder hidden
        self.V  = nn.Linear(units, 1, bias=False)       # 计算标量 score

    def forward(self, features, hidden):
        """
        Args:
            features: (batch, seq_len, units) — encoder 所有 hidden states
            hidden:   (batch, units)         — decoder 最后 hidden state (BiLSTM state_h)

        Returns:
            context_vector:   (batch, units)
            attention_weights: (batch, seq_len, 1)
        """
        hidden_exp = hidden.unsqueeze(1)  # (batch, 1, units)

        score = self.V(torch.tanh(self.W1(features) + self.W2(hidden_exp)))
        attention_weights = F.softmax(score, dim=1)       # (batch, seq_len, 1)
        context_vector = torch.sum(attention_weights * features, dim=1)  # (batch, units)

        return context_vector, attention_weights


if __name__ == "__main__":
    # 单元测试：维度验证
    batch, seq_len, units = 2, 1000, 128
    features = torch.randn(batch, seq_len, units)
    hidden = torch.randn(batch, units)

    attn = BahdanauAttention(units)
    context, weights = attn(features, hidden)

    assert context.shape == (batch, units), f"Expected ({batch}, {units}), got {context.shape}"
    assert weights.shape == (batch, seq_len, 1), f"Expected ({batch}, {seq_len}, 1), got {weights.shape}"
    assert torch.allclose(weights.sum(dim=1), torch.ones(batch, 1, 1)), "Weights must sum to 1"

    print("BahdanauAttention unit test passed!")
    print(f"  context: {context.shape}")
    print(f"  weights: {weights.shape}")
