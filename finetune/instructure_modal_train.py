"""
train/modal_train.py — Run GPT fine-tuning on Modal cloud GPU.

Usage:
    uv run python -m modal run train/modal_train.py::main              # train
    uv run python -m modal run train/modal_train.py::download          # download latest checkpoint
    uv run python -m modal run train/modal_train.py::download_specific # download a specific checkpoint
    uv run python -m modal run train/modal_train.py::list_checkpoints  # list all checkpoint files

Prerequisites:
    uv run python -m modal token new    # one-time authentication
"""

import os
from pathlib import Path

import modal


# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT = Path.cwd()
DATA_DIR = ROOT / "instruction-follower-data"
DATA_DIR.mkdir(exist_ok=True)

REMOTE_DATA = "/data"
REMOTE_CKPT = "/checkpoints"


# ── Modal app + persistent volume ─────────────────────────────────────────────

app = modal.App("sair-minigpt-instruct")

volume = modal.Volume.from_name(
    "sair-minigpt-instruct-checkpoints",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "tiktoken",
        "numpy",
        "transformers",
        "pymupdf",
        "wandb",
        "matplotlib",
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
            "data/raw",
            "runs",
            "mlruns",
            ".git",
            "*.gif",
            "*.jpg",
            "*.png",
        ],
    )
    .add_local_dir(str(DATA_DIR), remote_path=REMOTE_DATA)
)


# ── Local entrypoints ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main() -> None:
    """Launch instruction fine-tuning on a Modal A100 GPU."""
    print("Launching fine-tuning on Modal A100...")
    train_fn.remote()


@app.local_entrypoint()
def download() -> None:
    """Download the latest checkpoint and loss-curve PNG to the local machine."""
    local_ckpt_dir = ROOT / "checkpoints"
    local_ckpt_dir.mkdir(exist_ok=True)

    all_files = list(volume.listdir("/"))
    checkpoint_files: list[str] = sorted(
        f.path
        for f in all_files
        if f.path.startswith("epoch_") and f.path.endswith(".pt")
    )
    extra_files: list[str] = [
        f.path for f in all_files if not f.path.startswith("epoch_")
    ]

    if not checkpoint_files:
        print("No checkpoints found in Modal volume.")
        return

    # Download the latest checkpoint + any PNG loss curves
    to_download: list[str] = [checkpoint_files[-1]] + [
        f for f in extra_files if f.endswith(".png")
    ]

    for filename in to_download:
        dest = local_ckpt_dir / filename
        print(f"Downloading {filename} → {dest}")
        with open(dest, "wb") as fh:
            for chunk in volume.read_file(filename):
                fh.write(chunk)

    print(f"\nDone. Latest checkpoint: checkpoints/{checkpoint_files[-1]}")
    print("Run inference with:  uv run python cli.py generate 'Harry Potter'")
    print("Or launch the UI:    uv run python cli.py ui")


@app.local_entrypoint()
def download_specific() -> None:
    """Download a single hard-coded checkpoint from the Modal volume."""
    filename = "epoch_26.pt"
    local_ckpt_dir = ROOT / "checkpoints"
    local_ckpt_dir.mkdir(exist_ok=True)

    dest = local_ckpt_dir / filename
    print(f"Downloading {filename} from Modal volume → {dest}")
    with open(dest, "wb") as fh:
        for chunk in volume.read_file(filename):
            fh.write(chunk)

    print(f"Done. Saved to checkpoints/{filename}")


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
    Fine-tune the miniGPT model on instruction-following data.

    Runs remotely on a Modal A100. Loads the model variant defined in
    ``config.VARIANT``, builds train/val data loaders from the tokenised
    ``.bin`` files mounted at ``REMOTE_DATA``, and trains for 30 epochs.
    The final checkpoints are committed to the persistent Modal volume.
    """
    import sys

    sys.path.insert(0, "/app")

    import torch

    from config import MODEL_PRESET, MODELS, VARIANT
    from finetune.instructure_follower_finetuning import (
        loading_model,
        split_and_get_loaders,
    )
    from train.trainer import train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    model, config = loading_model(VARIANT)
    param_count: int = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    context_size: int = config["context_length"]  # e.g. 256 (tiny) or 1024 (medium)
    train_loader, val_loader, _ = split_and_get_loaders(REMOTE_DATA)

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=REMOTE_CKPT,
        num_epochs=30,
        context_size=context_size,
        model_config=config,
    )

    volume.commit()
    print("Training complete. Checkpoints saved to Modal volume.")