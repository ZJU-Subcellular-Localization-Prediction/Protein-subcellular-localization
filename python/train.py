"""
训练脚本 — 7 种对比架构的训练 + 评估

优化特性 (Step 2.6):
  - Location 逆频率加权 CrossEntropyLoss（26x 类别不平衡反制）
  - Adam L2 正则化 (weight_decay=1e-4)
  - ReduceLROnPlateau 动态学习率调度
  - 双任务 Loss 解耦 (loss = loc_loss + mem_weight * mem_loss)

用法:
    python train.py --model FFN --epochs 60
    python train.py --model CNN_BLSTM_Attention --epochs 60 --use_class_weights --mem_weight 0.5
    python train.py --model Complete --epochs 120 --batch_size 64
"""

import os
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import matplotlib
matplotlib.use("Agg")  # 非 GUI 后端，适用于服务器/无显示器环境
import matplotlib.pyplot as plt

# 添加当前目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import create_dataloaders
from models.architectures import create_model

# ====== 标签映射（来自 CLAUDE.md） ======

LOCATION_CLASSES = [
    "Cell.membrane", "Cytoplasm", "Endoplasmic.reticulum",
    "Golgi.apparatus", "Lysosome/Vacuole", "Mitochondrion",
    "Nucleus", "Peroxisome", "Plastid", "Extracellular"
]
MEMBRANE_CLASSES = ["Membrane", "Soluble", "Unknown"]


def parse_args():
    p = argparse.ArgumentParser(description="Train protein subcellular localization model")
    p.add_argument("--model", default="FFN",
                   help="Model architecture (FFN, CNN, BLSTM, CNN_BLSTM, BLSTM_Attention, "
                        "CNN_BLSTM_Attention, Complete)")
    p.add_argument("--data", default="data/features/manifest.pt",
                   help="Path to manifest.pt")
    p.add_argument("--epochs", type=int, default=60, help="Max training epochs")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    p.add_argument("--n_hid", type=int, default=64, help="Hidden units for LSTM/Dense")
    p.add_argument("--n_filt", type=int, default=32, help="Conv1d filters")
    p.add_argument("--drop_prob", type=float, default=0.5, help="Dropout probability")
    p.add_argument("--drop_hid", type=float, default=0.3, help="Dropout for hidden layers (Complete only)")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    p.add_argument("--output", default="./", help="Output directory for checkpoints and plots")
    p.add_argument("--no_eval", action="store_true", help="Skip final evaluation on test set")
    # 训练优化参数 (Step 2.6)
    p.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization coefficient for Adam")
    p.add_argument("--mem_weight", type=float, default=0.5, help="Membrane loss weight relative to location")
    p.add_argument("--lr_patience", type=int, default=4, help="ReduceLROnPlateau patience (epochs)")
    p.add_argument("--lr_factor", type=float, default=0.5, help="ReduceLROnPlateau factor")
    p.add_argument("--use_class_weights", action="store_true", default=True,
                   help="Use inverse-frequency class weights for Location loss")
    p.add_argument("--no_class_weights", action="store_true", default=False,
                   help="Disable class weights (overrides --use_class_weights)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--monitor", default="val_acc", choices=["val_acc", "val_loss"],
                   help="Metric for early stopping / checkpoint / LR scheduler (default: val_acc)")
    return p.parse_args()


def compute_accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def validate(model, loader, crit_loc_weighted, crit_loc_unweighted, crit_mem,
             is_dual, device, mem_weight=0.5):
    """返回 dict: {total_loss_weighted, total_loss_unweighted, loc_loss_weighted,
                    loc_loss_unweighted, mem_loss, loc_acc, mem_acc}"""
    model.eval()
    total_w_sum, total_uw_sum = 0.0, 0.0
    loc_w_sum, loc_uw_sum, mem_sum = 0.0, 0.0, 0.0
    loc_correct, mem_correct, n_samples = 0, 0, 0

    for X, y_loc, y_mem in loader:
        X, y_loc = X.to(device), y_loc.to(device)
        y_mem = y_mem.to(device)

        out = model(X)
        if is_dual:
            loc_logits, mem_logits = out
            loc_w = crit_loc_weighted(loc_logits, y_loc).item()
            loc_uw = crit_loc_unweighted(loc_logits, y_loc).item()
            mem_l = crit_mem(mem_logits, y_mem).item()
            loc_w_sum += loc_w * X.size(0)
            loc_uw_sum += loc_uw * X.size(0)
            mem_sum += mem_l * X.size(0)
            total_w_sum += (loc_w + mem_weight * mem_l) * X.size(0)
            total_uw_sum += (loc_uw + mem_weight * mem_l) * X.size(0)
            loc_correct += (torch.argmax(loc_logits, dim=1) == y_loc).sum().item()
            mem_correct += (torch.argmax(mem_logits, dim=1) == y_mem).sum().item()
        else:
            loc_logits = out
            loc_w = crit_loc_weighted(loc_logits, y_loc).item()
            loc_uw = crit_loc_unweighted(loc_logits, y_loc).item()
            loc_w_sum += loc_w * X.size(0)
            loc_uw_sum += loc_uw * X.size(0)
            total_w_sum += loc_w * X.size(0)
            total_uw_sum += loc_uw * X.size(0)
            loc_correct += (torch.argmax(loc_logits, dim=1) == y_loc).sum().item()

        n_samples += X.size(0)

    n = n_samples
    result = {
        "total_loss_weighted": total_w_sum / n,
        "total_loss_unweighted": total_uw_sum / n,
        "loc_loss_weighted": loc_w_sum / n,
        "loc_loss_unweighted": loc_uw_sum / n,
        "loc_acc": loc_correct / n,
    }
    if is_dual:
        result["mem_loss"] = mem_sum / n
        result["mem_acc"] = mem_correct / n
    return result


# ====== 评估指标（Step 2.5） ======

@torch.no_grad()
def evaluate(model, loader, is_dual, device):
    """在给定 DataLoader 上收集全部预测和真实标签。"""
    model.eval()
    all_loc_preds, all_loc_true = [], []
    all_mem_preds, all_mem_true = [], []

    for X, y_loc, y_mem in loader:
        X = X.to(device)
        out = model(X)
        if is_dual:
            loc_logits, mem_logits = out
            all_mem_preds.append(torch.argmax(mem_logits, dim=1).cpu())
            all_mem_true.append(y_mem)
        else:
            loc_logits = out
        all_loc_preds.append(torch.argmax(loc_logits, dim=1).cpu())
        all_loc_true.append(y_loc)

    loc_preds = torch.cat(all_loc_preds).numpy()
    loc_true = torch.cat(all_loc_true).numpy()
    result = {"loc_true": loc_true, "loc_preds": loc_preds}

    if is_dual:
        mem_preds = torch.cat(all_mem_preds).numpy()
        mem_true = torch.cat(all_mem_true).numpy()
        result["mem_true"] = mem_true
        result["mem_preds"] = mem_preds

    return result


def gorodkin(y_true, y_pred):
    """多类 Matthews 相关系数（Gorodkin 度量）"""
    return matthews_corrcoef(y_true, y_pred)


def mcc_membrane(y_true, y_pred):
    """膜分类 Matthews 相关系数"""
    return matthews_corrcoef(y_true, y_pred)


def plot_confusion_matrix(cm, classes, title, save_path):
    """绘制混淆矩阵热力图并保存"""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=8)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_curves(history, save_path):
    """绘制 loss/acc 曲线并保存（P0: 同时显示 weighted 和 unweighted val_loss）"""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss (weighted + unweighted)
    axes[0, 0].plot(epochs, history["train_loss"], label="Train (weighted)")
    if "val_loss_weighted" in history:
        axes[0, 0].plot(epochs, history["val_loss_weighted"], label="Val (weighted)", alpha=0.7)
    if "val_loss_unweighted" in history:
        axes[0, 0].plot(epochs, history["val_loss_unweighted"], label="Val (unweighted)", alpha=0.7)
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Total Loss (weighted + unweighted)"); axes[0, 0].legend()

    # Location accuracy
    axes[0, 1].plot(epochs, history["train_loc_acc"], label="Train")
    axes[0, 1].plot(epochs, history["val_loc_acc"], label="Val")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Location Accuracy"); axes[0, 1].legend()

    # Membrane loss/acc (if available)
    if "train_mem_loss" in history:
        axes[1, 0].plot(epochs, history["train_mem_loss"], label="Train")
        axes[1, 0].plot(epochs, history["val_mem_loss"], label="Val")
        axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_title("Membrane Loss"); axes[1, 0].legend()

        axes[1, 1].plot(epochs, history["train_mem_acc"], label="Train")
        axes[1, 1].plot(epochs, history["val_mem_acc"], label="Val")
        axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Accuracy")
        axes[1, 1].set_title("Membrane Accuracy"); axes[1, 1].legend()
    else:
        # Single output: hide bottom row
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def print_measures(model_name, history, best_epoch, loc_mcc, mem_mcc):
    """仿照原项目 models.py 第 626-642 行的 print_measures"""
    idx = best_epoch - 1  # 0-indexed
    val_loss_w = history["val_loss_weighted"][idx]
    val_loss_uw = history["val_loss_unweighted"][idx]
    val_loc_acc = history["val_loc_acc"][idx]

    print(f"\n{'='*60}")
    print(f"Best values for Network {model_name}")
    print(f"{'='*60}")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation loss (weighted):   {val_loss_w:.6f}")
    print(f"Validation loss (unweighted): {val_loss_uw:.6f}")
    print(f"Validation accuracy (location): {val_loc_acc:.6f}")

    if "val_mem_acc" in history and history["val_mem_acc"][idx] >= 0:
        val_mem_acc = history["val_mem_acc"][idx]
        print(f"Validation accuracy (membrane): {val_mem_acc:.6f}")

    print(f"Gorodkin measure on test (location MCC): {loc_mcc:.6f}")
    if mem_mcc is not None:
        print(f"MCC measure on test (membrane): {mem_mcc:.6f}")
    print(f"{'='*60}\n")


# ====== 主训练函数 ======

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 固定随机种子 ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Random seed: {args.seed}")

    # ---- 判断是否双输出 -----
    model_name_lower = args.model.lower().replace("-", "_").replace(" ", "_")
    is_dual = model_name_lower != "complete" and "complete" not in model_name_lower

    # ---- 加载数据 ----
    manifest_path = args.data
    if not os.path.isabs(manifest_path):
        manifest_path = os.path.join(os.path.dirname(__file__), args.data)
    train_loader, val_loader, test_loader = create_dataloaders(
        manifest_path, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ---- 创建模型 ----
    model = create_model(
        args.model, seq_len=1000, n_feat=640,
        n_hid=args.n_hid, n_class=10, drop_prob=args.drop_prob,
        n_filt=args.n_filt, drop_hid=args.drop_hid
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} | Dual output: {is_dual} | Trainable params: {n_params:,}")

    # ---- 类别权重（Location 逆频率加权） ----
    train_counts = np.array([800, 1525, 517, 215, 192, 906, 2427, 93, 453, 1185])
    total = train_counts.sum()
    n_class = 10
    loc_weights_raw = total / (n_class * train_counts)
    # [1.04, 0.55, 1.61, 3.87, 4.33, 0.92, 0.34, 8.94, 1.84, 0.70]

    use_class_weights = args.use_class_weights and not args.no_class_weights
    if use_class_weights:
        loc_weights_tensor = torch.FloatTensor(loc_weights_raw).to(device)
        criterion_loc = nn.CrossEntropyLoss(weight=loc_weights_tensor)
        print(f"Location class weights (inverse frequency): {loc_weights_raw}")
    else:
        criterion_loc = nn.CrossEntropyLoss()
        print("Location class weights: DISABLED (using uniform)")

    criterion_mem = nn.CrossEntropyLoss()  # M:S=1.7:1, healthy distribution
    print(f"Membrane loss: unweighted CrossEntropyLoss (M:S=1.7:1 is healthy)")

    # P0: 无加权 Loss 专用于验证监控（不被类别权重扭曲）
    criterion_loc_unweighted = nn.CrossEntropyLoss()
    print("Validation monitor: unweighted CrossEntropyLoss for val_loss_raw")

    # ---- 优化器（含 L2 正则化） ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    print(f"Optimizer: Adam(lr={args.lr}, weight_decay={args.weight_decay})")

    # ---- 学习率调度器 ----
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_factor,
        patience=args.lr_patience, min_lr=1e-6, verbose=True
    )
    print(f"LR scheduler: ReduceLROnPlateau(patience={args.lr_patience}, "
          f"factor={args.lr_factor}, min_lr=1e-6)")
    print(f"Dual-task loss: loss = loc_loss + {args.mem_weight} * mem_loss")
    print(f"Monitor metric: {args.monitor} (used for early stopping + LR scheduler + checkpoint)")

    # ---- 训练状态 ----
    os.makedirs(args.output, exist_ok=True)
    best_metric = float("-inf") if args.monitor == "val_acc" else float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {
        "train_loss": [], "val_loss_weighted": [], "val_loss_unweighted": [],
        "train_loc_acc": [], "val_loc_acc": [],
    }
    if is_dual:
        history["train_mem_loss"] = []
        history["val_mem_loss"] = []
        history["train_mem_acc"] = []
        history["val_mem_acc"] = []

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # ---- 训练阶段 ----
        model.train()
        train_loss_sum, train_loc_correct, train_mem_correct, n_batches, n_samples = 0.0, 0, 0, 0, 0
        train_loc_loss_sum, train_mem_loss_sum = 0.0, 0.0

        for X, y_loc, y_mem in train_loader:
            X, y_loc = X.to(device), y_loc.to(device)
            y_mem = y_mem.to(device) if is_dual else None

            optimizer.zero_grad()
            out = model(X)

            if is_dual:
                loc_logits, mem_logits = out
                loc_loss = criterion_loc(loc_logits, y_loc)
                mem_loss = criterion_mem(mem_logits, y_mem)
                loss = loc_loss + args.mem_weight * mem_loss

                train_loc_loss_sum += loc_loss.item() * X.size(0)
                train_mem_loss_sum += mem_loss.item() * X.size(0)
                train_loc_correct += (torch.argmax(loc_logits, dim=1) == y_loc).sum().item()
                train_mem_correct += (torch.argmax(mem_logits, dim=1) == y_mem).sum().item()
            else:
                loc_logits = out  # Complete 模型只输出 location
                loss = criterion_loc(loc_logits, y_loc)
                train_loc_loss_sum += loss.item() * X.size(0)
                train_loc_correct += (torch.argmax(loc_logits, dim=1) == y_loc).sum().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()

            train_loss_sum += loss.item() * X.size(0)
            n_samples += X.size(0)
            n_batches += 1

        # ---- 验证阶段 ----
        val = validate(model, val_loader, criterion_loc, criterion_loc_unweighted,
                       criterion_mem, is_dual, device, mem_weight=args.mem_weight)
        val_loss_weighted = val["total_loss_weighted"]
        val_loss_unweighted = val["total_loss_unweighted"]

        # ---- 选择监控指标 ----
        if args.monitor == "val_acc":
            monitor_value = val["loc_acc"]
            is_better = monitor_value > best_metric
        else:
            monitor_value = val_loss_unweighted
            is_better = monitor_value < best_metric

        # ---- 记录 ----
        epoch_train_loss = train_loss_sum / n_samples
        epoch_train_loc_acc = train_loc_correct / n_samples

        history["train_loss"].append(epoch_train_loss)
        history["val_loss_weighted"].append(val_loss_weighted)
        history["val_loss_unweighted"].append(val_loss_unweighted)
        history["train_loc_acc"].append(epoch_train_loc_acc)
        history["val_loc_acc"].append(val["loc_acc"])

        log = (f"Epoch {epoch:3d}/{args.epochs} | "
               f"LR: {optimizer.param_groups[0]['lr']:.1e} | "
               f"T loss: {epoch_train_loss:.4f} | "
               f"V loss(w): {val_loss_weighted:.4f} | V loss(uw): {val_loss_unweighted:.4f} | "
               f"T loc acc: {epoch_train_loc_acc:.4f} | V loc acc: {val['loc_acc']:.4f}")

        if is_dual:
            epoch_train_mem_acc = train_mem_correct / n_samples
            history["train_mem_loss"].append(train_mem_loss_sum / n_samples)
            history["val_mem_loss"].append(val["mem_loss"])
            history["train_mem_acc"].append(epoch_train_mem_acc)
            history["val_mem_acc"].append(val["mem_acc"])
            log += f" | T mem acc: {epoch_train_mem_acc:.4f} | V mem acc: {val['mem_acc']:.4f}"

        print(log)

        # ---- 学习率调度（P0: 使用选定的监控指标） ----
        if args.monitor == "val_acc":
            scheduler.step(-monitor_value)  # ReduceLROnPlateau mode='min', 传入负值使"越大越好"变为"越小越好"
        else:
            scheduler.step(monitor_value)

        # ---- Early Stopping + Checkpoint ----
        if is_better:
            best_metric = monitor_value
            best_epoch = epoch
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_metrics": val,
                "args": vars(args),
                "history": history,
            }
            torch.save(checkpoint, os.path.join(args.output, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch} (patience={args.patience})")
                break

    train_time = time.time() - t_start
    metric_name = "val_acc" if args.monitor == "val_acc" else "val_loss_unweighted"
    print(f"\nTraining finished in {train_time/60:.1f} min. "
          f"Best epoch: {best_epoch} ({metric_name}={best_metric:.4f})")

    # ---- 保存训练曲线 ----
    plot_curves(history, os.path.join(args.output, "training_curves.png"))
    print(f"Training curves saved to {os.path.join(args.output, 'training_curves.png')}")

    # ====== Step 2.5: 评估（测试集） ======
    if not args.no_eval:
        print(f"\n{'='*60}")
        print("Evaluating best model on test set...")
        print(f"{'='*60}")

        ckpt = torch.load(os.path.join(args.output, "best_model.pt"),
                         map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        eval_result = evaluate(model, test_loader, is_dual, device)

        # Gorodkin / MCC
        loc_mcc = gorodkin(eval_result["loc_true"], eval_result["loc_preds"])
        mem_mcc = mcc_membrane(eval_result["mem_true"], eval_result["mem_preds"]) if is_dual else None

        print(f"Test Gorodkin (location MCC): {loc_mcc:.6f}")
        if mem_mcc is not None:
            print(f"Test MCC (membrane): {mem_mcc:.6f}")

        # 混淆矩阵 — Location
        cm_loc = confusion_matrix(eval_result["loc_true"], eval_result["loc_preds"])
        plot_confusion_matrix(cm_loc, LOCATION_CLASSES,
                              "Confusion Matrix — Location (Test Set)",
                              os.path.join(args.output, "cm_location.png"))

        # 混淆矩阵 — Membrane (dual only)
        if is_dual:
            cm_mem = confusion_matrix(eval_result["mem_true"], eval_result["mem_preds"])
            plot_confusion_matrix(cm_mem, MEMBRANE_CLASSES,
                                  "Confusion Matrix — Membrane (Test Set)",
                                  os.path.join(args.output, "cm_membrane.png"))

        # print_measures
        print_measures(args.model, history, best_epoch, loc_mcc, mem_mcc)

    return 0


if __name__ == "__main__":
    main()
