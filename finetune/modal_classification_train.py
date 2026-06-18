"""
finetune/modal_classification_train.py — Run GPT spam-classification fine-tuning on Modal.

Usage:
    uv run python -m modal run finetune/modal_classification_train.py              # train
    uv run python -m modal run finetune/modal_classification_train.py::download    # download latest checkpoint
    uv run python -m modal run finetune/modal_classification_train.py::list_checkpoints
"""

import os
from pathlib import Path
import modal

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT = Path.cwd()
REMOTE_CKPT = "/checkpoints"

# ── Modal app + persistent volume ─────────────────────────────────────────────

app = modal.App("sair-minigpt-classifier")

volume = modal.Volume.from_name(
    "sair-minigpt-classifier-checkpoints",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "tiktoken",
        "numpy",
        "pandas",
        "transformers",
        "wandb",
    )
    .add_local_dir(
        str(ROOT),
        remote_path="/app",
        ignore=[
            ".venv",
            "**__pycache__**",
            "*.pyc",
            "*.pyo",
            "checkpoints",
            "runs",
            ".git",
            "*.gif",
            "*.jpg",
            "*.png",
        ],
    )
)


# ── Local entrypoints ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main() -> None:
    """Launch classification fine-tuning on a Modal A100 GPU."""
    print("Launching classification fine-tuning on Modal A100...")
    train_fn.remote()


@app.local_entrypoint()
def download() -> None:
    """Download the latest and best checkpoints to the local machine."""
    local_ckpt_dir = ROOT / "checkpoints"
    local_ckpt_dir.mkdir(exist_ok=True)

    all_files = list(volume.listdir("/"))
    
    # Files to download
    target_filenames = ["best_model.pt"]
    
    # Also find the latest epoch checkpoint
    checkpoint_files = sorted(
        f.path
        for f in all_files
        if f.path.startswith("epoch_") and f.path.endswith(".pt")
    )
    if checkpoint_files:
        target_filenames.append(checkpoint_files[-1])
    
    # Also get any images
    target_filenames.extend([f.path for f in all_files if f.path.endswith(".png")])

    for filename in target_filenames:
        # Check if file actually exists in volume
        if not any(f.path == filename for f in all_files):
            continue
            
        dest = local_ckpt_dir / filename
        print(f"Downloading {filename} → {dest}")
        with open(dest, "wb") as fh:
            for chunk in volume.read_file(filename):
                fh.write(chunk)

    print(f"\nDone. Checkpoints saved to {local_ckpt_dir}")


@app.local_entrypoint()
def list_checkpoints() -> None:
    """List all files stored in the Modal volume with their sizes."""
    files = list(volume.listdir("/"))

    if not files:
        print("Modal volume is empty.")
        return

    print(f"{'File':<30} {'Size':>12}")
    print("-" * 44)
    for f in sorted(files, key=lambda x: x.path):
        size_mb: float = getattr(f, "size", 0) / (1024 * 1024)
        print(f"{f.path:<30} {size_mb:>10.1f} MB")


# ── Remote training function ───────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A100",
    volumes={REMOTE_CKPT: volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=8 * 3600,
)
def train_fn() -> None:
    """
    Fine-tune the miniGPT model for spam classification.
    """
    import sys
    sys.path.insert(0, "/app")

    import torch
    import wandb
    from config import VARIANT, data_dir_classification, MODEL_PRESET
    from data.dataset import get_classification_dataloaders
    from finetune.classification_finetuning import setup_classification_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # ── W&B Init ───────────────────────────────────────────────────────────────
    wandb.init(
        project="sair-minigpt-classifier",
        config={
            "variant": VARIANT,
            "preset": MODEL_PRESET,
            "epochs": 20,
            "lr": 5e-5,
            "batch_size": 8,
        }
    )

    # ── Config ─────────────────────────────────────────────────────────────────
    NUM_EPOCHS = 20
    LR = 5e-5

    # ── Data preparation ───────────────────────────────────────────────────────
    csv_remote_path = "/app/classification_data/SMSSpamCollection.csv"
    print(f"Loading data from: {csv_remote_path}")
    train_loader, val_loader, test_loader = get_classification_dataloaders(
        csv_path=csv_remote_path,
        output_dir=f"{REMOTE_CKPT}/data-splits"
    )

    # ── Model setup ───────────────────────────────────────────────────────────
    print(f"Setting up model for variant: {VARIANT}")
    model, config = setup_classification_model(VARIANT)
    model.to(device)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def calc_accuracy_loader(data_loader, num_batches=None):
        model.eval()
        correct, total = 0, 0
        limit = num_batches or len(data_loader)
        for i, (x, y) in enumerate(data_loader):
            if i >= limit:
                break
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                logits = model(x)[:, -1, :]
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.size(0)
        return correct / total

    # ── Training loop ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)[:, -1, :]
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(f"  epoch {epoch} | step {step + 1}/{len(train_loader)} | loss {loss.item():.4f}")
                wandb.log({"train/loss": loss.item(), "epoch": epoch, "step": step})

        train_acc = calc_accuracy_loader(train_loader, num_batches=10)
        val_acc   = calc_accuracy_loader(val_loader)
        avg_loss  = total_loss / len(train_loader)
        
        print(f"Epoch {epoch}/{NUM_EPOCHS}  avg_loss={avg_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")
        wandb.log({
            "epoch": epoch,
            "epoch/avg_loss": avg_loss,
            "epoch/train_acc": train_acc,
            "epoch/val_acc": val_acc,
        })

        # Save per-epoch checkpoint
        ckpt_path = f"{REMOTE_CKPT}/epoch_{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        
        # Best model logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = f"{REMOTE_CKPT}/best_model.pt"
            torch.save(model.state_dict(), best_path)
            print(f"  ⭐ New best model! accuracy: {val_acc:.4f} → {best_path}")
            wandb.run.summary["best_val_acc"] = best_val_acc

        volume.commit()

    # ── Final test evaluation ──────────────────────────────────────────────────
    print("\nLoading best model for final testing...")
    model.load_state_dict(torch.load(f"{REMOTE_CKPT}/best_model.pt"))
    test_acc = calc_accuracy_loader(test_loader)
    print(f"Final test accuracy (Best Model): {test_acc:.4f}")
    wandb.run.summary["test_acc"] = test_acc

    wandb.finish()
    volume.commit()
    print("Training complete. Checkpoints saved to Modal volume.")
