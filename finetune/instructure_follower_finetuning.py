from pathlib import Path

import tiktoken
import torch

from config import HF_MODELS, INSTRUCTION_DATA_DIR, MODEL_CONFIG, VARIANT, CKPT_DIR
from data.dataset import data_split, get_instruction_loaders
from inference.load_weights import load_from_hf
from model.gpt import GPTModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_and_get_loaders(data_dir: str | Path) -> tuple:
    """Load instruction data, split it, and create DataLoaders."""
    user_file = Path(data_dir) / "instrution-data.json"
    data_split(user_file, output_dir=data_dir)
    tokenizer = tiktoken.get_encoding("gpt2")
    return get_instruction_loaders(data_dir=data_dir, tokenizer=tokenizer)


def loading_model(variant: str) -> tuple[torch.nn.Module, dict]:
    """Load a model from HuggingFace or a local checkpoint."""
    if variant in HF_MODELS:
        print(f"Loading pretrained model from HuggingFace: {variant}")
        model, config = load_from_hf(variant)
    else:
        print(f"Loading model from checkpoint: {variant}")
        model = GPTModel(MODEL_CONFIG).to(device)
        model.load_state_dict(torch.load(variant, map_location=device))
        config = MODEL_CONFIG
    return model.to(device), config


if __name__ == "__main__":
    from train.trainer import train
    from inference.generate      import generate, load_model, load_finetuned_model
    from tiktoken import get_encoding

    train_loader, val_loader, test_loader = split_and_get_loaders(INSTRUCTION_DATA_DIR)
    model, config = loading_model(VARIANT)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Training locally on: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train(model, train_loader, val_loader, device)
    print("model trained")
    # model = load_finetuned_model(VARIANT, CKPT_DIR)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # 
    # tokenizer = get_encoding("gpt2")
    # out = generate(
    #     model          = model,
    #     prompt         = """
    # ### Instruction:
    # Solve the following problem. (the answer is two)
    #
    # ### Input:
    # What is 1 + 1 ?
    # ###Response
    # """,
    #     max_new_tokens = 100,
    #     context_size   = config["context_length"],
    #     tokenizer      = tokenizer,
    #     device         = device,
    #     temperature    = 0.8,
    #     top_k          = 50,
    #     top_p          = 0.9,
    #     beams          = 1,
    #     method         = "greedy",
    # )
    # print(out)

