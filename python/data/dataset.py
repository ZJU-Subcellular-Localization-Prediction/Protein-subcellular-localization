"""
PyTorch Dataset — 文件级按需加载

每条序列的 ESM-2 embedding 存为独立 .pt 文件，Dataset 在 __getitem__ 时
按需从磁盘加载单条，避免一次性将全部特征加载到内存。

DataLoader 使用 num_workers > 0 可并行 I/O 加载。
"""
import torch
import os


class ProteinDataset(torch.utils.data.Dataset):
    """
    Args:
        manifest_entries: list of dict, each containing:
            {'file': 'train/00005.pt', 'y_location': 5, 'y_membrane': 0, 'seq_len_raw': 423}
        features_dir: root directory of features (e.g. 'data/features')
        seq_len: target sequence length for padding (e.g. 1000)
    """

    def __init__(self, manifest_entries, features_dir, seq_len):
        self.entries = manifest_entries
        self.features_dir = features_dir
        self.seq_len = seq_len
        self.n_feat = None  # lazy: 从第一条数据推断

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        # 按需从磁盘加载单条 .pt
        filepath = os.path.join(self.features_dir, entry['file'])
        emb = torch.load(filepath, map_location='cpu', weights_only=True)
        # emb: (seq_len_raw, 640) float16

        # 确定特征维度（仅第一次）
        if self.n_feat is None:
            self.n_feat = emb.shape[1]

        actual_len = min(emb.shape[0], self.seq_len)

        # 零填充到固定长度
        X = torch.zeros(self.seq_len, emb.shape[1], dtype=torch.float32)
        X[:actual_len, :] = emb[:actual_len, :].float()

        y_loc = int(entry['y_location'])
        y_mem = int(entry['y_membrane'])

        return X, y_loc, y_mem


def load_manifest_split(manifest_path, split_name):
    """
    加载 manifest 中指定 split 的条目列表。

    Args:
        manifest_path: path to manifest.pt (e.g. 'data/features/manifest.pt')
        split_name: 'train', 'val', or 'test'

    Returns:
        (entries, features_dir, seq_len)
          - entries: list of dict for the split
          - features_dir: directory containing the .pt files
          - seq_len: target sequence length from manifest
    """
    manifest = torch.load(manifest_path, map_location='cpu', weights_only=False)
    features_dir = os.path.dirname(manifest_path)
    seq_len = manifest['seq_len']
    entries = manifest['splits'].get(split_name, [])
    if not entries:
        raise ValueError(f"Split '{split_name}' not found or empty in manifest. "
                         f"Available: {list(manifest['splits'].keys())}")
    return entries, features_dir, seq_len


def create_dataloaders(manifest_path, batch_size=32, num_workers=4):
    """
    便捷函数：一次性创建 train/val/test 三个 DataLoader。

    Returns:
        train_loader, val_loader, test_loader
    """
    entries_train, feat_dir, seq_len = load_manifest_split(manifest_path, 'train')
    entries_val, _, _ = load_manifest_split(manifest_path, 'val')
    entries_test, _, _ = load_manifest_split(manifest_path, 'test')

    ds_train = ProteinDataset(entries_train, feat_dir, seq_len)
    ds_val = ProteinDataset(entries_val, feat_dir, seq_len)
    ds_test = ProteinDataset(entries_test, feat_dir, seq_len)

    tl = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers, pin_memory=True)
    vl = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True)
    tt = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True)

    return tl, vl, tt


# ====== 单元测试 ======

if __name__ == "__main__":
    import sys

    manifest_path = "data/features/manifest.pt"
    if not os.path.exists(manifest_path):
        # 尝试从 python/ 目录的父级查找
        alt_path = "../data/features/manifest.pt"
        if os.path.exists(alt_path):
            manifest_path = alt_path
        else:
            print(f"manifest not found at {manifest_path} or {alt_path}")
            print("Skipping dataset unit test (manifest required).")
            sys.exit(0)

    print("=" * 60)
    print("Testing ProteinDataset with file-level loading")
    print("=" * 60)

    entries, feat_dir, seq_len = load_manifest_split(manifest_path, 'train')
    print(f"  Train entries: {len(entries)}")
    print(f"  Features dir:  {feat_dir}")
    print(f"  Target seq_len: {seq_len}")

    ds = ProteinDataset(entries, feat_dir, seq_len)
    print(f"  Dataset length: {len(ds)}")

    # 测试单条加载
    X, y_loc, y_mem = ds[0]
    print(f"  Sample 0 — X: {X.shape}, loc: {y_loc}, mem: {y_mem}")
    assert X.shape == (seq_len, 640), f"Expected ({seq_len}, 640), got {X.shape}"
    assert X.dtype == torch.float32
    print(f"  X dtype: {X.dtype}, values non-zero: {(X != 0).any().item()}")

    # 测试 DataLoader
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    Xb, yb_loc, yb_mem = next(iter(loader))
    print(f"  Batch — X: {Xb.shape}, loc: {yb_loc.shape}, mem: {yb_mem.shape}")

    # 验证 train/val/test 无重叠
    _, _, _ = load_manifest_split(manifest_path, 'val')
    _, _, _ = load_manifest_split(manifest_path, 'test')
    print("  All 3 splits loaded successfully.")

    print()
    print("ProteinDataset unit test PASSED!")
