"""
7 种对比架构的 PyTorch 实现

移植自 utils/models.py（Keras → PyTorch）

架构速查：
  1. FFN                          — 纯前馈网络
  2. CNN                          — 卷积神经网络
  3. BLSTM                        — 双向 LSTM
  4. CNN_BLSTM                    — CNN + 双向 LSTM
  5. BLSTM_Attention              — 双向 LSTM + Bahdanau Attention
  6. CNN_BLSTM_Attention          — CNN + 双向 LSTM + Attention
  7. CNN_BLSTM_Attention_Complete — 6 并行 Conv + BiLSTM + Attention（最终版，仅 location 输出）
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 支持直接运行和模块导入
try:
    from .attention import BahdanauAttention
except ImportError:
    from attention import BahdanauAttention


# ====== 权重初始化 ======

def init_orthogonal(module, gain=math.sqrt(2)):
    """对 nn.Linear / nn.Conv1d 应用正交初始化（等价于 Keras Orthogonal(gain=sqrt(2))）"""
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


# ====== 架构 1：纯前馈网络 ======

class FFN(nn.Module):
    """Flatten → Dense(n_hid, ReLU) → Dropout → 2×Dense(softmax)"""

    def __init__(self, seq_len, n_feat, n_hid, n_class, drop_prob, n_membrane_class=3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(seq_len * n_feat, n_hid)
        self.dropout = nn.Dropout(drop_prob)
        self.out_location = nn.Linear(n_hid, n_class)
        self.out_membrane = nn.Linear(n_hid, n_membrane_class)

    def forward(self, x):
        # x: (batch, seq_len, n_feat)
        x = self.flatten(x)
        x = F.relu(self.dense(x))
        x = self.dropout(x)
        return self.out_location(x), self.out_membrane(x)


# ====== 架构 2：卷积神经网络 ======

class CNN(nn.Module):
    """
    Input → Permute(channels_first) → Conv1d(3)+Conv1d(5) → Concat →
    Conv1d(3) → Permute → MaxPool1d(5) → Flatten → Dense → Dropout → 2×Dense
    """

    def __init__(self, seq_len, n_feat, n_hid, n_class, drop_prob, n_filt, n_membrane_class=3):
        super().__init__()
        self.conv_a = nn.Conv1d(n_feat, n_filt, 3, padding=1)
        self.conv_b = nn.Conv1d(n_feat, n_filt, 5, padding=2)
        self.conv_final = nn.Conv1d(n_filt * 2, n_filt * 2, 3, padding=1)
        self.pool = nn.MaxPool1d(5)
        self.dense = nn.Linear((seq_len // 5) * n_filt * 2, n_hid)
        self.dropout = nn.Dropout(drop_prob)
        self.out_location = nn.Linear(n_hid, n_class)
        self.out_membrane = nn.Linear(n_hid, n_membrane_class)

    def forward(self, x):
        # x: (batch, seq_len, n_feat)
        x = x.permute(0, 2, 1)                      # (batch, n_feat, seq_len)
        a = F.relu(self.conv_a(x))
        b = F.relu(self.conv_b(x))
        x = torch.cat([a, b], dim=1)                 # (batch, 2*n_filt, seq_len)
        x = F.relu(self.conv_final(x))
        # 直接在 channels_first 格式上池化序列维度，等价于 Keras Permute→MaxPool1D
        x = self.pool(x)                              # (batch, 2*n_filt, seq_len//5)
        x = x.permute(0, 2, 1)                        # (batch, seq_len//5, 2*n_filt)
        x = torch.flatten(x, 1)
        x = F.relu(self.dense(x))
        x = self.dropout(x)
        return self.out_location(x), self.out_membrane(x)


# ====== 架构 3：双向 LSTM ======

class BLSTM(nn.Module):
    """
    Input → LSTM(forward, return_sequences=False) + LSTM(backward, return_sequences=False) →
    Concat(2*n_hid) → Dense(n_hid*2, ReLU) → Dropout → 2×Dense(softmax)
    """

    def __init__(self, seq_len, n_feat, n_hid, n_class, drop_prob, n_membrane_class=3):
        super().__init__()
        # 等价于原代码两个独立的单向 LSTM
        self.lstm = nn.LSTM(n_feat, n_hid, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(n_hid * 2, n_hid * 2)
        self.dropout = nn.Dropout(drop_prob)
        self.out_location = nn.Linear(n_hid * 2, n_class)
        self.out_membrane = nn.Linear(n_hid * 2, n_membrane_class)

    def forward(self, x):
        # x: (batch, seq_len, n_feat)
        _, (h_n, _) = self.lstm(x)
        # h_n: (2, batch, n_hid) — [forward, backward]
        h = torch.cat([h_n[0], h_n[1]], dim=1)       # (batch, 2*n_hid)
        h = F.relu(self.dense(h))
        h = self.dropout(h)
        return self.out_location(h), self.out_membrane(h)


# ====== 架构 4：CNN + 双向 LSTM ======

class CNN_BLSTM(nn.Module):
    """
    CNN 特征提取 → BiLSTM(return_sequences=False) → Concat → Dense → Dropout → 2×Dense
    """

    def __init__(self, seq_len, n_feat, n_hid, n_class, drop_prob, n_filt, n_membrane_class=3):
        super().__init__()
        self.conv_a = nn.Conv1d(n_feat, n_filt, 3, padding=1)
        self.conv_b = nn.Conv1d(n_feat, n_filt, 5, padding=2)
        self.conv_final = nn.Conv1d(n_filt * 2, n_filt * 2, 3, padding=1)
        self.lstm = nn.LSTM(n_filt * 2, n_hid, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(n_hid * 2, n_hid * 2)
        self.dropout = nn.Dropout(drop_prob)
        self.out_location = nn.Linear(n_hid * 2, n_class)
        self.out_membrane = nn.Linear(n_hid * 2, n_membrane_class)

    def forward(self, x):
        x = x.permute(0, 2, 1)                      # (batch, n_feat, seq_len)
        a = F.relu(self.conv_a(x))
        b = F.relu(self.conv_b(x))
        x = torch.cat([a, b], dim=1)                 # (batch, 2*n_filt, seq_len)
        x = F.relu(self.conv_final(x))
        x = x.permute(0, 2, 1)                       # (batch, seq_len, 2*n_filt)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[0], h_n[1]], dim=1)       # (batch, 2*n_hid)
        h = F.relu(self.dense(h))
        h = self.dropout(h)
        return self.out_location(h), self.out_membrane(h)


# ====== 架构 5：BiLSTM + Attention ======

class BLSTM_Attention(nn.Module):
    """
    BiLSTM(return_sequences=True, return_state=True) → Attention → Dropout → 2×Dense(softmax)
    """

    def __init__(self, seq_len, n_feat, n_hid, n_class, drop_prob, n_membrane_class=3):
        super().__init__()
        self.drop_in = nn.Dropout(drop_prob)  # Keras LSTM(dropout=drop_prob) 是输入 dropout
        self.lstm = nn.LSTM(n_feat, n_hid, batch_first=True, bidirectional=True)
        self.attention = BahdanauAttention(n_hid * 2)
        self.dropout = nn.Dropout(drop_prob)
        self.out_location = nn.Linear(n_hid * 2, n_class)
        self.out_membrane = nn.Linear(n_hid * 2, n_membrane_class)

    def forward(self, x):
        x = self.drop_in(x)
        l_lstm, (h_n, _) = self.lstm(x)
        state_h = torch.cat([h_n[0], h_n[1]], dim=1)  # (batch, 2*n_hid)
        context, _ = self.attention(l_lstm, state_h)    # (batch, 2*n_hid)
        context = self.dropout(context)
        return self.out_location(context), self.out_membrane(context)


# ====== 架构 6：CNN + BiLSTM + Attention ======

class CNN_BLSTM_Attention(nn.Module):
    """
    CNN → BiLSTM(return_sequences=True) → LayerNorm → Attention → Dense → Dropout → 2×Dense

    P1 增强：BiLSTM 输出后增加 LayerNorm，沿特征维度归一化以稳定 Attention score 分布。
    v3.0 增强：640→256 线性瓶颈层 + 同方差不确定性多任务权重 (log_var_loc, log_var_mem)。
    """
    def __init__(self, seq_len, n_feat, n_hid, n_class, drop_prob, n_filt,
                 bottleneck_dim=256, n_membrane_class=3):
        super().__init__()
        hidden_dim = n_hid * 2

        # ---- v3.0: 信息瓶颈 — 640→256 可学习压缩 ----
        self.input_proj = nn.Sequential(
            nn.Linear(n_feat, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob)
        )
        # Conv 层输入通道从 n_feat(640) 改为 bottleneck_dim(256)
        self.conv_a = nn.Conv1d(bottleneck_dim, n_filt, 3, padding=1)
        self.conv_b = nn.Conv1d(bottleneck_dim, n_filt, 5, padding=2)
        self.conv_final = nn.Conv1d(n_filt * 2, n_filt * 2, 3, padding=1)

        self.drop_in = nn.Dropout(drop_prob)
        self.lstm = nn.LSTM(n_filt * 2, n_hid, batch_first=True, bidirectional=True)
        self.layernorm = nn.LayerNorm(hidden_dim)     # P1: 稳定 Attention 输入分布
        self.attention = BahdanauAttention(hidden_dim)
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_prob)
        self.out_location = nn.Linear(hidden_dim, n_class)
        self.out_membrane = nn.Linear(hidden_dim, n_membrane_class)

        # ---- v3.0: 同方差不确定性多任务权重 ----
        # log_var_loc = log(σ²_loc), log_var_mem = log(σ²_mem)
        # 初始 σ=1.0 → log_var=0.0，即初始双任务等权
        self.log_var_loc = nn.Parameter(torch.zeros(1))
        self.log_var_mem = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (batch, seq_len, n_feat)
        x = self.input_proj(x)                        # v3.0: (batch, seq_len, 256)
        x = x.permute(0, 2, 1)                        # (batch, 256, seq_len)
        a = F.relu(self.conv_a(x))
        b = F.relu(self.conv_b(x))
        x = torch.cat([a, b], dim=1)                 # (batch, 2*n_filt, seq_len)
        x = F.relu(self.conv_final(x))
        x = x.permute(0, 2, 1)                       # (batch, seq_len, 2*n_filt)
        x = self.drop_in(x)
        l_lstm, (h_n, _) = self.lstm(x)
        l_lstm = self.layernorm(l_lstm)               # P1: (batch, seq_len, 2*n_hid)
        state_h = torch.cat([h_n[0], h_n[1]], dim=1)  # (batch, 2*n_hid)
        state_h = self.layernorm(state_h)              # P1: 归一化 decoder hidden
        context, _ = self.attention(l_lstm, state_h)   # (batch, 2*n_hid)
        context = F.relu(self.dense(context))
        context = self.dropout(context)
        return self.out_location(context), self.out_membrane(context)


# ====== 架构 7：完整版（6 并行 Conv + BiLSTM + Attention）======

class CNN_BLSTM_Attention_Complete(nn.Module):
    """
    最终版架构（仅输出 location，无 membrane）：
      Input → Dropout → Permute →
      6×Conv1d(k=1,3,5,9,15,21, n_filt, Orthogonal, ReLU) → Concat →
      Permute → Conv1d(128, 3, ReLU) →
      BiLSTM(n_hid, return_seq=True, dropout=drop_hid) →
      Concat(forward_h, backward_h) → Attention → Dropout →
      Dense(n_hid*2, ReLU, Orthogonal) → Dropout → Dense(n_class, Orthogonal)
    """

    def __init__(self, seq_len, n_feat, n_hid, n_class, drop_prob, drop_hid, n_filt):
        super().__init__()
        kernels = [1, 3, 5, 9, 15, 21]

        # Input dropout
        self.drop_input = nn.Dropout(drop_prob)

        # 6 并行 Conv1d（channels_first 格式：输入 (n_feat, seq_len)）
        self.convs = nn.ModuleList([
            nn.Conv1d(n_feat, n_filt, k, padding=k // 2) for k in kernels
        ])

        # 中间 Conv1d：输入 (seq_len, 6*n_filt)，由 Permute 后按 channels_first 解释
        # → 输入维度 = seq_len channels，6*n_filt 个 spatial positions
        self.conv_mid = nn.Conv1d(seq_len, 128, 3, padding=1)

        # BiLSTM：输入 (batch, 128, 6*n_filt) → Keras LSTM 默认 channels_last
        #   → timesteps=128, features=6*n_filt
        self.drop_lstm = nn.Dropout(drop_hid)  # 模拟 Keras LSTM(dropout=drop_hid)
        self.lstm = nn.LSTM(6 * n_filt, n_hid, batch_first=True, bidirectional=True)

        # 可学习初始状态（等价 Keras layers[12].initial_states = Orthogonal()）
        self.h0 = nn.Parameter(torch.zeros(2, 1, n_hid))
        self.c0 = nn.Parameter(torch.zeros(2, 1, n_hid))

        # Attention
        self.attention = BahdanauAttention(n_hid * 2)

        # Dense layers
        self.drop1 = nn.Dropout(drop_hid)
        self.dense1 = nn.Linear(n_hid * 2, n_hid * 2)
        self.drop2 = nn.Dropout(drop_hid)
        self.out_location = nn.Linear(n_hid * 2, n_class)

        # 正交初始化
        self._init_weights()

    def _init_weights(self):
        for conv in self.convs:
            init_orthogonal(conv)
        init_orthogonal(self.dense1)
        init_orthogonal(self.out_location)
        init_orthogonal(self.h0)
        init_orthogonal(self.c0)

    def forward(self, x):
        # x: (batch, seq_len, n_feat)
        batch_size = x.size(0)
        s_len = x.size(1)

        x = self.drop_input(x)

        # Permute → (batch, n_feat, seq_len)
        x = x.permute(0, 2, 1)

        # 6 并行 Conv
        conv_outs = [F.relu(conv(x)) for conv in self.convs]  # each (batch, n_filt, seq_len)
        x = torch.cat(conv_outs, dim=1)                        # (batch, 6*n_filt, seq_len)

        # Permute → (batch, seq_len, 6*n_filt)，然后 Conv1d(channels_first 语义)
        x = x.permute(0, 2, 1)                                  # (batch, seq_len, 6*n_filt)
        x = F.relu(self.conv_mid(x))                            # (batch, 128, 6*n_filt)

        # BiLSTM
        x = self.drop_lstm(x)                                   # (batch, 128, 6*n_filt)
        h0 = self.h0.expand(-1, batch_size, -1).contiguous()
        c0 = self.c0.expand(-1, batch_size, -1).contiguous()
        l_lstm, (h_n, _) = self.lstm(x, (h0, c0))              # output (batch, 128, 2*n_hid)
        state_h = torch.cat([h_n[0], h_n[1]], dim=1)           # (batch, 2*n_hid)

        # Attention
        context, _ = self.attention(l_lstm, state_h)            # (batch, 2*n_hid)

        # Dense + Dropout
        context = self.drop1(context)
        context = F.relu(self.dense1(context))
        context = self.drop2(context)
        return self.out_location(context)                       # (batch, n_class)


# ====== 工厂函数 ======

def create_model(name, seq_len=1000, n_feat=640, n_hid=64, n_class=10,
                 drop_prob=0.3, n_filt=32, drop_hid=0.3, bottleneck_dim=256,
                 n_membrane_class=3):
    """根据名称创建模型，统一参数接口"""
    name_lower = name.lower().replace('-', '_').replace(' ', '_')
    if name_lower == 'ffn':
        return FFN(seq_len, n_feat, n_hid, n_class, drop_prob, n_membrane_class)
    elif name_lower == 'cnn':
        return CNN(seq_len, n_feat, n_hid, n_class, drop_prob, n_filt, n_membrane_class)
    elif name_lower == 'blstm' or name_lower == 'lstm':
        return BLSTM(seq_len, n_feat, n_hid, n_class, drop_prob, n_membrane_class)
    elif name_lower == 'cnn_blstm' or name_lower == 'cnn_lstm':
        return CNN_BLSTM(seq_len, n_feat, n_hid, n_class, drop_prob, n_filt, n_membrane_class)
    elif name_lower == 'blstm_attention' or name_lower == 'lstm_attention':
        return BLSTM_Attention(seq_len, n_feat, n_hid, n_class, drop_prob, n_membrane_class)
    elif name_lower == 'cnn_blstm_attention' or name_lower == 'cnn_lstm_attention':
        return CNN_BLSTM_Attention(seq_len, n_feat, n_hid, n_class, drop_prob, n_filt,
                                   bottleneck_dim, n_membrane_class)
    elif name_lower == 'complete' or name_lower == 'cnn_blstm_attention_complete':
        return CNN_BLSTM_Attention_Complete(seq_len, n_feat, n_hid, n_class, drop_prob, drop_hid, n_filt)
    else:
        raise ValueError(f"Unknown model: {name}. "
                         f"Choose from: FFN, CNN, BLSTM, CNN_BLSTM, BLSTM_Attention, "
                         f"CNN_BLSTM_Attention, Complete")


# ====== 单元测试 ======

if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("Testing all 7 architectures")
    print("=" * 60)

    batch, seq_len, n_feat, n_hid, n_class = 2, 1000, 640, 64, 10
    drop_prob, drop_hid, n_filt = 0.3, 0.3, 32

    x = torch.randn(batch, seq_len, n_feat)

    tests = []

    # 1. FFN
    m = FFN(seq_len, n_feat, n_hid, n_class, drop_prob)
    loc, mem = m(x)
    tests.append(("FFN (dual output)", loc.shape == (batch, n_class) and mem.shape == (batch, 3)))
    m.train(); loss = loc.sum() + mem.sum(); loss.backward()

    # 2. CNN
    m = CNN(seq_len, n_feat, n_hid, n_class, drop_prob, n_filt)
    loc, mem = m(x)
    tests.append(("CNN (dual output)", loc.shape == (batch, n_class) and mem.shape == (batch, 3)))
    m.train(); loss = loc.sum() + mem.sum(); loss.backward()

    # 3. BLSTM
    m = BLSTM(seq_len, n_feat, n_hid, n_class, drop_prob)
    loc, mem = m(x)
    tests.append(("BLSTM (dual output)", loc.shape == (batch, n_class) and mem.shape == (batch, 3)))
    m.train(); loss = loc.sum() + mem.sum(); loss.backward()

    # 4. CNN_BLSTM
    m = CNN_BLSTM(seq_len, n_feat, n_hid, n_class, drop_prob, n_filt)
    loc, mem = m(x)
    tests.append(("CNN_BLSTM (dual output)", loc.shape == (batch, n_class) and mem.shape == (batch, 3)))
    m.train(); loss = loc.sum() + mem.sum(); loss.backward()

    # 5. BLSTM_Attention
    m = BLSTM_Attention(seq_len, n_feat, n_hid, n_class, drop_prob)
    loc, mem = m(x)
    tests.append(("BLSTM_Attention (dual output)", loc.shape == (batch, n_class) and mem.shape == (batch, 3)))
    m.train(); loss = loc.sum() + mem.sum(); loss.backward()

    # 6. CNN_BLSTM_Attention
    m = CNN_BLSTM_Attention(seq_len, n_feat, n_hid, n_class, drop_prob, n_filt)
    loc, mem = m(x)
    tests.append(("CNN_BLSTM_Attention (dual output)", loc.shape == (batch, n_class) and mem.shape == (batch, 3)))
    m.train(); loss = loc.sum() + mem.sum(); loss.backward()

    # 7. Complete (single output — location only)
    m = CNN_BLSTM_Attention_Complete(seq_len, n_feat, n_hid, n_class, drop_prob, drop_hid, n_filt)
    out = m(x)
    tests.append(("Complete (single output)", out.shape == (batch, n_class)))
    m.train(); loss = out.sum(); loss.backward()

    # 工厂函数测试
    m2 = create_model("cnn_blstm_attention", seq_len, n_feat, n_hid)
    tests.append(("Factory function", m2 is not None))

    # 结果汇总
    print()
    all_pass = True
    for name, passed in tests:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")

    print()
    model_names = [
        "FFN", "CNN", "BLSTM", "CNN_BLSTM",
        "BLSTM_Attention", "CNN_BLSTM_Attention", "Complete"
    ]
    print(f"All 7 architectures + factory: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print()

    # 参数量统计
    print("Model parameter counts:")
    for name in model_names:
        model = create_model(name, seq_len, n_feat, n_hid)
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {name:30s}: {n_params:>10,d} params ({n_trainable:>10,d} trainable)")

    sys.exit(0 if all_pass else 1)
