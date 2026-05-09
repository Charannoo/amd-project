"""
STEP 3: Fine-tune ESM2-650M on drug-target binding data
Run: python 3_train.py [--resume]
Expected runtime: 4-6 hours on AMD MI300X (with AMP)
Expected cost: ~$15-25

Outputs saved to ~/alchemy_training/checkpoints/:
  esm2_alchemy_best.pt     — best model checkpoint
  esm2_alchemy_last.pt     — latest checkpoint (for resume)
  training_curves.png      — loss + accuracy curves (use in submission!)
  training_log.json        — full metrics log
"""
import json
import os
import sys
import time
from pathlib import Path

# ── ROCm setup (must be before torch import) ────────────────────────────────
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.4.2")

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from transformers import (
    EsmModel,
    EsmTokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path.home() / "alchemy_training" / "data"
OUTPUT_DIR   = Path.home() / "alchemy_training" / "checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME   = "facebook/esm2_t33_650M_UR50D"
BATCH_SIZE   = 16
GRAD_ACCUM   = 2        # effective batch = 32
LR           = 2e-5
EPOCHS       = 3
MAX_SEQ_LEN  = 512
WARMUP_STEPS = 500
LOG_EVERY    = 50       # record step loss every N optimizer steps
USE_AMP      = False    # Automatic Mixed Precision disabled to prevent NaN
RESUME       = "--resume" in sys.argv

USE_WANDB = False
if USE_WANDB:
    import wandb
    wandb.init(project="alchemy-esm2", config={
        "model": MODEL_NAME, "batch_size": BATCH_SIZE, "lr": LR, "epochs": EPOCHS,
    })

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'='*62}")
print(f"  ALCHEMY — ESM2-650M Fine-tuning")
print(f"{'='*62}")
print(f"  Device:    {device}")
if torch.cuda.is_available():
    print(f"  GPU:       {torch.cuda.get_device_name(0)}")
    print(f"  VRAM:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  AMP:       {'Enabled' if USE_AMP else 'Disabled'}")
print(f"  Batch:     {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM} effective")
print(f"  Epochs:    {EPOCHS}")
print(f"  LR:        {LR}")
print(f"  Resume:    {RESUME}")
print(f"{'='*62}\n")

if not torch.cuda.is_available():
    print("⚠ WARNING: No GPU detected. Training on CPU will take days.")
    print("  Make sure you're running on the MI300X droplet.\n")

# ── Verify data exists ────────────────────────────────────────────────────────
train_file = DATA_DIR / "train.jsonl"
val_file = DATA_DIR / "val.jsonl"
if not train_file.exists() or not val_file.exists():
    print("❌ ERROR: Training data not found.")
    print(f"  Expected: {train_file}")
    print(f"  Expected: {val_file}")
    print("  Run 'python 2_prepare_data.py' first.")
    sys.exit(1)


# ── Dataset ───────────────────────────────────────────────────────────────────
class BindingDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.rows = []
        with open(jsonl_path) as f:
            for line in f:
                self.rows.append(json.loads(line))
        print(f"  Loaded {len(self.rows):,} rows from {jsonl_path.name}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(
            row["sequence"],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pki":            torch.tensor(row["pki"],   dtype=torch.float32),
            "label":          torch.tensor(row["label"], dtype=torch.float32),
        }

# ── Model ─────────────────────────────────────────────────────────────────────
class ESM2BindingPredictor(nn.Module):
    """
    ESM2-650M backbone + two prediction heads:
      1. pKi regression  (continuous binding affinity)
      2. Binary classifier (active / inactive binder)
    """
    def __init__(self, esm_model):
        super().__init__()
        self.esm     = esm_model
        hidden       = esm_model.config.hidden_size  # 1280 for 650M
        self.dropout = nn.Dropout(0.1)
        self.norm    = nn.LayerNorm(hidden)
        self.regressor = nn.Sequential(
            nn.Linear(hidden, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        out    = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = self.norm(self.dropout(pooled))
        return self.regressor(pooled).squeeze(-1), self.classifier(pooled).squeeze(-1)

# ── Graph generation ──────────────────────────────────────────────────────────
def save_training_graphs(history, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed — skipping graphs.")
        return None

    PURPLE = "#7c6ff7"
    TEAL   = "#00d4aa"
    ORANGE = "#ff8c42"
    GRID   = "#2a2a4a"
    TEXT   = "#e0e0e0"
    BG     = "#0a0a1a"
    PANEL  = "#12122a"

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

    def make_ax(row, col, title, ylabel, xlabel="Epoch"):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=TEXT, fontsize=11, pad=8)
        ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
        ax.set_xlabel(xlabel, color=TEXT, fontsize=9)
        ax.tick_params(colors=TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
        return ax

    epochs = list(range(1, len(history["val_loss"]) + 1))

    # 1. Total loss
    ax = make_ax(0, 0, "Total Loss", "Loss")
    ax.plot(epochs, history["train_loss_epoch"], color=PURPLE, marker="o", label="Train", lw=2)
    ax.plot(epochs, history["val_loss"],         color=TEAL,   marker="s", label="Val",   lw=2)
    ax.legend(facecolor="#1a1a2e", labelcolor=TEXT, fontsize=9)

    # 2. Regression loss (pKi)
    ax = make_ax(0, 1, "Regression Loss (pKi)", "MSE")
    ax.plot(epochs, history["train_reg_epoch"], color=PURPLE, marker="o", label="Train", lw=2)
    ax.plot(epochs, history["val_reg"],         color=TEAL,   marker="s", label="Val",   lw=2)
    ax.legend(facecolor="#1a1a2e", labelcolor=TEXT, fontsize=9)

    # 3. Classification loss
    ax = make_ax(0, 2, "Classification Loss (Active/Inactive)", "BCE")
    ax.plot(epochs, history["train_cls_epoch"], color=PURPLE, marker="o", label="Train", lw=2)
    ax.plot(epochs, history["val_cls"],         color=TEAL,   marker="s", label="Val",   lw=2)
    ax.legend(facecolor="#1a1a2e", labelcolor=TEXT, fontsize=9)

    # 4. Val accuracy
    ax = make_ax(1, 0, "Validation Accuracy", "Accuracy")
    ax.plot(epochs, history["val_acc"], color=ORANGE, marker="^", lw=2.5)
    ax.set_ylim(0, 1)
    for ep, acc in zip(epochs, history["val_acc"]):
        ax.annotate(f"{acc:.3f}", (ep, acc), textcoords="offset points",
                    xytext=(0, 9), ha="center", color=ORANGE, fontsize=10, fontweight="bold")

    # 5. Step-level loss + smoothed
    ax = make_ax(1, 1, "Train Loss per Step (smoothed)", "Loss", xlabel="Optimizer Step")
    steps  = history["step_numbers"]
    losses = history["step_losses"]
    if steps:
        ax.plot(steps, losses, color=PURPLE, lw=1, alpha=0.35, label="Raw")
        w = max(1, len(losses) // 30)
        smoothed = [
            sum(losses[max(0, i-w):i+1]) / len(losses[max(0, i-w):i+1])
            for i in range(len(losses))
        ]
        ax.plot(steps, smoothed, color=TEAL, lw=2, label=f"Smooth (w={w})")
        ax.legend(facecolor="#1a1a2e", labelcolor=TEXT, fontsize=8)

    # 6. LR schedule
    ax = make_ax(1, 2, "Learning Rate Schedule", "LR", xlabel="Optimizer Step")
    if history["step_lrs"]:
        ax.plot(history["step_numbers"], history["step_lrs"], color=ORANGE, lw=1.5)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_color(TEXT)

    best_acc  = max(history["val_acc"])  if history["val_acc"]  else 0
    best_loss = min(history["val_loss"]) if history["val_loss"] else 0
    fig.suptitle(
        f"ALCHEMY  —  ESM2-650M Fine-tuning on BindingDB Drug-Target Binding\n"
        f"Best Val Loss: {best_loss:.4f}   |   Best Val Accuracy: {best_acc:.4f}   |   AMD MI300X / ROCm",
        color=TEXT, fontsize=12, y=1.02,
    )

    out = output_dir / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n  📊 Curves saved → {out}")
    return str(out)


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    print("\n=== Loading tokenizer and model ===")
    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    esm_base  = EsmModel.from_pretrained(MODEL_NAME)
    model     = ESM2BindingPredictor(esm_base).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params/1e6:.1f}M")

    if torch.cuda.is_available():
        print(f"GPU Memory after model load: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    print("\n=== Loading datasets ===")
    train_ds = BindingDataset(train_file, tokenizer, MAX_SEQ_LEN)
    val_ds   = BindingDataset(val_file,   tokenizer, MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    scheduler   = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)
    mse_loss    = nn.MSELoss()
    bce_loss    = nn.BCEWithLogitsLoss()
    scaler      = GradScaler(enabled=USE_AMP)

    best_val_loss = float("inf")
    global_step   = 0
    start_epoch   = 1
    run_start     = time.time()

    history = {
        "train_loss_epoch": [], "train_reg_epoch": [], "train_cls_epoch": [],
        "val_loss": [], "val_reg": [], "val_cls": [], "val_acc": [],
        "step_losses": [], "step_numbers": [], "step_lrs": [],
    }

    # ── Resume from checkpoint ──
    last_ckpt = OUTPUT_DIR / "esm2_alchemy_last.pt"
    if RESUME and last_ckpt.exists():
        print(f"\n  Resuming from {last_ckpt} ...")
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch   = ckpt["epoch"] + 1
        global_step   = ckpt["global_step"]
        best_val_loss = ckpt["best_val_loss"]
        history       = ckpt["history"]
        if "scaler_state" in ckpt and USE_AMP:
            scaler.load_state_dict(ckpt["scaler_state"])
        print(f"  Resumed at epoch {start_epoch}, step {global_step}, best_val_loss={best_val_loss:.4f}\n")

    print(f"\n=== Training epochs {start_epoch}–{EPOCHS}  ({total_steps} total steps) ===\n")

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        optimizer.zero_grad()
        total_loss = reg_sum = cls_sum = 0.0
        n_batches  = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")
        for step, batch in enumerate(pbar):
            ids   = batch["input_ids"].to(device, non_blocking=True)
            mask  = batch["attention_mask"].to(device, non_blocking=True)
            pki_t = batch["pki"].to(device, non_blocking=True)
            lbl_t = batch["label"].to(device, non_blocking=True)

            with autocast(enabled=USE_AMP):
                pki_p, logit = model(ids, mask)
                l_reg = mse_loss(pki_p, pki_t)
                l_cls = bce_loss(logit, lbl_t)
                loss  = (2.0 * l_reg + l_cls) / GRAD_ACCUM

            scaler.scale(loss).backward()

            total_loss += loss.item() * GRAD_ACCUM
            reg_sum    += l_reg.item()
            cls_sum    += l_cls.item()
            n_batches  += 1

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % LOG_EVERY == 0:
                    history["step_losses"].append(total_loss / n_batches)
                    history["step_numbers"].append(global_step)
                    history["step_lrs"].append(scheduler.get_last_lr()[0])

            if (step + 1) % 100 == 0:
                pbar.set_postfix(
                    loss=f"{total_loss/n_batches:.4f}",
                    reg=f"{reg_sum/n_batches:.4f}",
                    cls=f"{cls_sum/n_batches:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

        e_loss = total_loss / max(n_batches, 1)
        e_reg  = reg_sum    / max(n_batches, 1)
        e_cls  = cls_sum    / max(n_batches, 1)

        # ── Validate ──
        model.eval()
        v_loss = v_reg = v_cls = 0.0
        n_val = correct = total = 0
        pki_preds, pki_trues = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]  "):
                ids   = batch["input_ids"].to(device, non_blocking=True)
                mask  = batch["attention_mask"].to(device, non_blocking=True)
                pki_t = batch["pki"].to(device, non_blocking=True)
                lbl_t = batch["label"].to(device, non_blocking=True)

                with autocast(enabled=USE_AMP):
                    pki_p, logit = model(ids, mask)
                    l_reg = mse_loss(pki_p, pki_t)
                    l_cls = bce_loss(logit, lbl_t)

                v_loss += (2.0 * l_reg + l_cls).item()
                v_reg  += l_reg.item()
                v_cls  += l_cls.item()
                n_val  += 1
                preds   = (torch.sigmoid(logit) >= 0.5).float()
                correct += (preds == lbl_t).sum().item()
                total   += lbl_t.size(0)
                pki_preds.extend(pki_p.float().cpu().tolist())
                pki_trues.extend(pki_t.float().cpu().tolist())

        v_loss /= max(n_val, 1)
        v_reg  /= max(n_val, 1)
        v_cls  /= max(n_val, 1)
        v_acc   = correct / max(total, 1)
        try:
            import numpy as np
            pearson_r = float(np.corrcoef(pki_preds, pki_trues)[0, 1])
        except Exception:
            pearson_r = float("nan")

        gpu_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        elapsed = time.time() - run_start

        print(f"\n{'='*62}")
        print(f"  Epoch {epoch}/{EPOCHS}  |  {(time.time()-t0)/60:.1f} min  |  GPU {gpu_gb:.1f} GB  |  Total {elapsed/3600:.1f} hrs")
        print(f"  Train   loss={e_loss:.4f}  reg={e_reg:.4f}  cls={e_cls:.4f}")
        print(f"  Val     loss={v_loss:.4f}  reg={v_reg:.4f}  cls={v_cls:.4f}")
        print(f"  Val     acc={v_acc:.4f}   pKi Pearson R={pearson_r:.4f}")
        print(f"{'='*62}\n")

        history["train_loss_epoch"].append(e_loss)
        history["train_reg_epoch"].append(e_reg)
        history["train_cls_epoch"].append(e_cls)
        history["val_loss"].append(v_loss)
        history["val_reg"].append(v_reg)
        history["val_cls"].append(v_cls)
        history["val_acc"].append(v_acc)

        # ── Save best checkpoint ──
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            ckpt = OUTPUT_DIR / "esm2_alchemy_best.pt"
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_loss": v_loss, "val_accuracy": v_acc,
                "val_pearson_r": pearson_r, "history": history,
                "config": {
                    "model_name": MODEL_NAME, "max_seq_len": MAX_SEQ_LEN,
                    "hidden_size": esm_base.config.hidden_size,
                    "batch_size": BATCH_SIZE, "lr": LR, "epochs": EPOCHS,
                },
            }, ckpt)
            print(f"  ✅ Best checkpoint saved → {ckpt}\n")

        # ── Save last checkpoint (for resume) ──
        torch.save({
            "epoch": epoch, "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if USE_AMP else {},
            "best_val_loss": best_val_loss,
            "history": history,
        }, OUTPUT_DIR / "esm2_alchemy_last.pt")

        # Save graph + log after every epoch
        save_training_graphs(history, OUTPUT_DIR)
        with open(OUTPUT_DIR / "training_log.json", "w") as f:
            json.dump(history, f, indent=2)

        if USE_WANDB:
            wandb.log({"val/loss": v_loss, "val/acc": v_acc,
                       "val/pki_pearson_r": pearson_r}, step=global_step)

    total_time = time.time() - run_start
    print(f"\n{'='*62}")
    print(f"  TRAINING COMPLETE")
    print(f"  Total time   : {total_time/3600:.2f} hours")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best val acc : {max(history['val_acc']):.4f}")
    print(f"  Checkpoint   : {OUTPUT_DIR / 'esm2_alchemy_best.pt'}")
    print(f"  Curves PNG   : {OUTPUT_DIR / 'training_curves.png'}")
    print(f"  Log JSON     : {OUTPUT_DIR / 'training_log.json'}")
    print(f"{'='*62}")
    print("\nNext: python 4_integrate.py")


if __name__ == "__main__":
    try:
        import matplotlib
    except ImportError:
        import subprocess
        print("Installing matplotlib...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "-q"])
    train()
