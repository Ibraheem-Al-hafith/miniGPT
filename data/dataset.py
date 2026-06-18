"""
data/dataset.py — GPT2Dataset and DataLoader factory.
Identical to Notebook 1 — just made importable.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from functools import partial # for collate_fn customization

from config import DATA_DIR, MAX_LEN, STRIDE, BATCH_SIZE, PAD_TOKEN_ID, INGNORE_INDEX, ALLOWED_MAX_LENGTH

#instruction follower imports
import json 
import os
import urllib.request
import random
import pandas as pd

"""
WE  MUST ADD INSTRUCTION FINETUNING CONFIGURATION
"""

class GPT2Dataset(Dataset):
    def __init__(self, file_path: Path, max_length: int, stride: int):
        self.data       = np.fromfile(file_path, dtype=np.int32)
        self.max_length = max_length
        self.stride     = stride

    def __len__(self):
        return (len(self.data) - self.max_length) // self.stride

    def __getitem__(self, idx):
        start = idx * self.stride
        x = torch.tensor(self.data[start : start + self.max_length],         dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.max_length + 1], dtype=torch.long)
        return x, y


def get_loaders(
    data_dir:   Path = DATA_DIR,
    max_length: int  = MAX_LEN,
    stride:     int  = STRIDE,
    batch_size: int  = BATCH_SIZE,
):
    def _make(split, shuffle):
        path = Path(data_dir) / f"{split}_ids.bin"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run: python cli.py prepare"
            )
        ds = GPT2Dataset(path, max_length, stride)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    return _make("train", True), _make("val", False), _make("test", False)

#-----------------------------------------------------------
#               Instruction Follower utilities
#-----------------------------------------------------------

def download_and_load_file(file_path, url) -> dict:
    # If the URL is a GitHub blob URL, convert it to a raw URL
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
    except json.JSONDecodeError:
        # If it fails, maybe the file was already downloaded incorrectly (e.g. as HTML)
        # Delete it and try one more time if we just used the original URL
        if os.path.exists(file_path):
            os.remove(file_path)
        
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
        
        with open(file_path, "r") as file:
            data = json.load(file)
            
    return data


def data_split(file_path, output_dir: str | Path="."):
    # 1. Load the original data
    with open(file_path, "r") as file:
        data = json.load(file)

    # 2. Calculate split indices
    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)    # 10% for testing
    
    # 3. Slice the data
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]  
    val_data = data[train_portion + test_portion:]

    # 4. Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 5. Save the splits to separate JSON files
    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(train_data, f, indent=4)
        
    with open(os.path.join(output_dir, "test.json"), "w") as f:
        json.dump(test_data, f, indent=4)
        
    with open(os.path.join(output_dir, "val.json"), "w") as f:
        json.dump(val_data, f, indent=4)

    print(f"Splits saved: train.json, test.json, val.json in '{output_dir}'")
    
    # Returning the data just in case you still need to use it right away
    return train_data, test_data, val_data


def format_input(entry: dict):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = (
        f"\n\n### Input: \n{entry['input']}" if entry['input'] else ""
    )
    return instruction_text + input_text
# ------------------Instruction follower dataset  ---------------------------
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenized_text = []
        for entry in data:
            formated_text = format_input(entry=entry)
            response_text = f"\n\n### Response: \n{entry['output']}"
            full_text = formated_text + response_text
            self.tokenized_text.append(
                tokenizer.encode(full_text)
            )
        
    def __getitem__(self, index):
        return self.tokenized_text[index]
    def __len__(self):
        return len(self.tokenized_text)



# --------------------- Custom collate function --------------------------
def custom_collate_fn(
        batch,
        pad_token_id=PAD_TOKEN_ID,
        ignore_index=INGNORE_INDEX,
        allowed_max_length=ALLOWED_MAX_LENGTH,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets==pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

# ---------------- instruction follower dataloaders ------------------



def get_instruction_loaders(
    data_dir,
    tokenizer,
    batch_size = BATCH_SIZE,
):
    def _make(split, shuffle):
        path = Path(data_dir) / f"{split}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run: python cli.py prepare"
            )
        with open(path, "r") as file:
            data = json.load(file)
        ds = InstructionDataset(data=data, tokenizer=tokenizer)
        return DataLoader(
            ds, batch_size=batch_size,
            shuffle=shuffle, drop_last=True,
            collate_fn=custom_collate_fn
            )

    return _make("train", True), _make("val", False), _make("test", False)

# ---------------------------------------------------------
#                       Classification
# ---------------------------------------------------------


def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    # Counts the instances
    ham_subset = df[df["Label"] == "ham"].sample(
    #of “spam”
    num_spam, random_state=123
    )
    # Randomly samples “ham”
    balanced_df = pd.concat([
    # instances to match the number
    ham_subset, df[df["Label"] == "spam"]
    # of “spam” instances
    ])
    balanced_df = balanced_df["Label"].map({"ham": 0, "spam": 1})
    return balanced_df

def random_split(df, train_frac, validation_frac, output_dir="."):
    # Shuffles the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    
    # Calculates split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    
    # Splits the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_path / "train.csv", index=None)
    validation_df.to_csv(out_path / "validation.csv", index=None)
    test_df.to_csv(out_path / "test.csv", index=None)
    
    return train_df, validation_df, test_df

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]
        self.encoded_texts = [
            encoded_text + [pad_token_id] *
            (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        # Convert label to 1 if spam, 0 if ham if it's still text
        if isinstance(label, str):
            label = 1 if label == "spam" else 0
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

def get_classification_dataloaders(csv_path: str, output_dir: str = "."):
    """
    steps: 
        1. read the csv file.
        2. balance the dataset (equal ham and spam).
        3. split the data into train, val, test.
        4. save the splitted data (the data class accepts only csv paths).
        5. initialize datasets.
        6. create dataloaders.
        7. return dataloaders.
    """
    import tiktoken
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Classification data not found at {csv_path}")

    # The SMSSpamCollection is actually a TSV with no header
    df = pd.read_csv(csv_path, sep="\t", names=["Label", "Text"])
    
    # ── BALANCE DATASET ──────────────────────────────────────────────────────
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(n=num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    # ─────────────────────────────────────────────────────────────────────────

    out_dir = Path(output_dir)
    random_split(balanced_df, 0.7, 0.1, output_dir=out_dir)
    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = SpamDataset(
        csv_file=out_dir / "train.csv",
        max_length=None,
        tokenizer=tokenizer
    )
    val_dataset = SpamDataset(
        csv_file=out_dir / "validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    test_dataset = SpamDataset(
        csv_file=out_dir / "test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    num_workers = 0
    batch_size = 8

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
    )
    return train_loader, val_loader, test_loader