"""
extract_features.py — ESM-2 特征提取脚本（文件级存储策略）

策略：每条序列的 ESM-2 embedding 存为独立 .pt 文件，Phase 2 训练时按需加载，
      避免一次性将全部特征加载到内存（~17GB → 单条仅 ~2.5MB）。

输出结构：
  python/data/features/
  ├── train/00000.pt ...    # 每条 (seq_len, 640) float16
  ├── val/00000.pt ...
  ├── test/00000.pt ...
  └── manifest.pt           # 包含 splits、labels、元信息
"""
import torch
import numpy as np
import math
import gc
import os
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO

# ====== 配置 ======
MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
SEQ_LEN = 1000
BATCH_SIZE = 4
INPUT_FILE = "D:/test/Protein-subcellular-localization-main/dataset/DeepLoc/DeepLoc.rtf"
FEATURES_DIR = "data/features"

# ====== 标签字典 ======
labels_dic_location = {
    'Cell.membrane': 0, 'Cytoplasm': 1, 'Endoplasmic.reticulum': 2,
    'Golgi.apparatus': 3, 'Lysosome/Vacuole': 4, 'Mitochondrion': 5,
    'Nucleus': 6, 'Peroxisome': 7, 'Plastid': 8, 'Extracellular': 9
}
labels_dic_membrane = {'M': 0, 'S': 1, 'U': 0}
# 注：membrane 标签 0/1/2 → 后续完整版模型只用 location，此处保留兼容


def parse_deeploc(filepath):
    """解析 DeepLoc.rtf，返回 records 列表"""
    fasta_sequences = list(SeqIO.parse(open(filepath), 'fasta-pearson'))
    records = []
    part = 1

    for fasta in fasta_sequences:
        description = fasta.description.rstrip('\\')
        sequence = str(fasta.seq).rstrip('\\').strip()
        split = description.split()

        if len(split) < 2 or split[1].startswith("Cytoplasm-Nucleus"):
            continue

        location = split[1].split("-")[0]
        membrane = split[1].split("-")[1][0]
        is_test = len(split) == 3

        if is_test:
            records.append((sequence, labels_dic_location[location],
                            labels_dic_membrane[membrane], True, 0))
        else:
            fold = ((part - 1) % 4) + 1
            records.append((sequence, labels_dic_location[location],
                            labels_dic_membrane[membrane], False, fold))
            part += 1

    return records


def center_truncate(seq, target_len):
    if len(seq) <= target_len:
        return seq
    extra = len(seq) - target_len
    idx_i = len(seq) // 2 - extra // 2
    idx_f = len(seq) // 2 + (extra + 1) // 2
    return seq[:idx_i] + seq[idx_f:]


def extract_and_save(sequences, records, split_name, split_indices,
                     model, tokenizer, device, batch_size, features_dir):
    """
    ESM-2 推理 → 按批处理 → 逐条保存为独立的 .pt 文件。
    每条文件仅包含 embedding tensor (seq_len_raw, 640) float16，
    不在此阶段做 padding（padding 由 Dataset 在加载时动态完成）。
    """
    os.makedirs(os.path.join(features_dir, split_name), exist_ok=True)
    manifest_entries = []

    # 按长度排序以提升 batch 效率
    idx_list = list(split_indices)
    idx_sorted = sorted(idx_list, key=lambda i: len(sequences[i]))
    sorted_seqs = [sequences[i] for i in idx_sorted]

    total = len(sorted_seqs)
    done_count = 0

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_seqs = sorted_seqs[batch_start:batch_end]
        batch_orig_indices = idx_sorted[batch_start:batch_end]

        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 1:-1, :]  # (B, max_seq, 640)

        for i, orig_idx in enumerate(batch_orig_indices):
            seq_len_raw = len(batch_seqs[i])
            emb = embeddings[i, :seq_len_raw, :].cpu().half()  # (seq_len, 640) float16

            # 保存为独立文件
            filepath = os.path.join(features_dir, split_name, f"{orig_idx:05d}.pt")
            torch.save(emb, filepath)

            _, loc_label, mem_label, _, _ = records[orig_idx]
            manifest_entries.append({
                'file': f"{split_name}/{orig_idx:05d}.pt",
                'orig_idx': orig_idx,
                'y_location': int(loc_label),
                'y_membrane': int(mem_label),
                'seq_len_raw': seq_len_raw,
            })

        done_count += len(batch_seqs)
        if done_count % 1000 == 0 or done_count == total:
            print(f"  [{split_name}] {done_count}/{total}")

    return manifest_entries


def main():
    print("=" * 60)
    print("Phase 1: ESM-2 Feature Extraction (file-level storage)")
    print("=" * 60)

    # Step 1 — 解析数据
    print(f"\n[1/4] Parsing {INPUT_FILE}...")
    records = parse_deeploc(INPUT_FILE)
    print(f"  Total: {len(records)} sequences")

    # Step 2 — 中心截断 + 划分 split
    print(f"\n[2/4] Preprocessing (center truncation) & splitting...")
    raw_sequences = [r[0] for r in records]
    processed_seqs = [center_truncate(seq, SEQ_LEN) for seq in raw_sequences]
    lengths = [len(s) for s in processed_seqs]
    print(f"  Seq lengths — min: {min(lengths)}, max: {max(lengths)}, mean: {np.mean(lengths):.0f}")

    # 确定各 split 的索引
    train_indices = []
    val_indices = []
    test_indices = []
    for i, r in enumerate(records):
        _, _, _, is_test, fold = r
        if is_test:
            test_indices.append(i)
        elif fold == 1:
            val_indices.append(i)
        else:
            train_indices.append(i)

    print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # Step 3 — 加载 ESM-2 + 逐批推理 + 直接存盘
    print(f"\n[3/4] Loading ESM-2 model: {MODEL_NAME}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(device)

    print("  Extracting features & saving to individual .pt files...")

    manifest = {
        'seq_len': SEQ_LEN,
        'n_feat': 640,
        'model_name': MODEL_NAME,
        'labels_location': list(labels_dic_location.keys()),
        'labels_membrane': list(labels_dic_membrane.keys()),
        'splits': {},
    }

    for split_name, split_indices in [('train', train_indices),
                                       ('val', val_indices),
                                       ('test', test_indices)]:
        if not split_indices:
            manifest['splits'][split_name] = []
            continue
        entries = extract_and_save(processed_seqs, records, split_name,
                                   split_indices, model, tokenizer, device,
                                   BATCH_SIZE, FEATURES_DIR)
        manifest['splits'][split_name] = entries

    del model, tokenizer
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Step 4 — 保存 manifest
    print(f"\n[4/4] Saving manifest...")
    manifest_path = os.path.join(FEATURES_DIR, "manifest.pt")
    torch.save(manifest, manifest_path)
    print(f"  Manifest saved to {manifest_path}")

    # 统计
    total_files = sum(len(v) for v in manifest['splits'].values())
    print(f"\n  Total .pt files created: {total_files}")
    print(f"  Per-file size: ~{640 * 500 * 2 / 1024:.0f} KB (avg seq_len=500)")

    print("=" * 60)
    print("Phase 1 complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
