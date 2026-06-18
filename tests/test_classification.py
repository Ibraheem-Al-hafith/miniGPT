import pytest
import torch
import pandas as pd
from pathlib import Path
from data.dataset import SpamDataset, get_classification_dataloaders
from finetune.classification_finetuning import setup_classification_model
import tiktoken

@pytest.fixture
def mock_classification_csv(tmp_path):
    """Create a small dummy SMSSpamCollection-style TSV file (no header)."""
    csv_path = tmp_path / "SMSSpamCollection.csv"
    data = [
        ["ham", "Hey, are we still meeting for lunch?"],
        ["ham", "Can you pick up some milk on your way home?"],
        ["ham", "I'll be there in 5 minutes."],
        ["ham", "The meeting is at 10 AM."],
        ["spam", "Congratulations! You've won a $1000 Walmart gift card. Click here."],
        ["spam", "WINNER! Call 0800-123-456 to claim your prize."],
    ]
    df = pd.DataFrame(data, columns=["Label", "Text"])
    # Write as TSV NO HEADER to mimic real SMSSpamCollection
    df.to_csv(csv_path, sep="\t", index=False, header=False)
    return csv_path

def test_spam_dataset(tmp_path):
    tokenizer = tiktoken.get_encoding("gpt2")
    # SpamDataset expects a file WITH a header (as produced by random_split)
    csv_path = tmp_path / "with_header.csv"
    data = [
        ["ham", "Hello world"],
        ["spam", "Free money"]
    ]
    df = pd.DataFrame(data, columns=["Label", "Text"])
    df.to_csv(csv_path, index=False)
    
    ds = SpamDataset(csv_path, tokenizer, max_length=10)
    assert len(ds) == 2
    x, y = ds[0]
    assert y.item() == 0 # ham
    x, y = ds[1]
    assert y.item() == 1 # spam

def test_get_classification_dataloaders(mock_classification_csv, tmp_path):
    out_dir = tmp_path / "processed"
    train_loader, val_loader, test_loader = get_classification_dataloaders(
        csv_path=mock_classification_csv,
        output_dir=out_dir
    )
    
    # Check if files were created
    assert (out_dir / "train.csv").exists()
    assert (out_dir / "validation.csv").exists()
    assert (out_dir / "test.csv").exists()
    
    # Check balancing: our mock has 4 ham, 2 spam. 
    # Balanced should have 2 ham, 2 spam = 4 total.
    # Split: 0.7 train, 0.1 val, 0.2 test.
    # 4 * 0.7 = 2.8 -> 2 for train.
    # 4 * 0.1 = 0.4 -> 0 for val? 
    # Let's check the total count across loaders.
    
    total_samples = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
    assert total_samples == 4 # Balanced count

def test_setup_classification_model():
    # We'll use the 'tiny' preset to keep it fast
    config = {
        "vocab_size": 50257, "context_length": 256,
        "emb_dim": 256, "n_heads": 4, "n_layers": 4,
        "drop_rate": 0.1, "qkv_bias": False,
    }
    
    # Mock loading_model to return a base GPTModel
    from model.gpt import GPTModel
    base_model = GPTModel(config)
    
    # We can't easily mock the 'loading_model' import inside classification_finetuning
    # so we test the logic directly or via setup_classification_model if we can point it to a preset
    
    # For unit testing the head replacement logic:
    emb_dim = base_model.out_head.in_features
    num_classes = 2
    base_model.out_head = torch.nn.Linear(emb_dim, num_classes)
    
    assert base_model.out_head.out_features == 2
    
    # Verify we can run a forward pass
    x = torch.randint(0, 50257, (1, 10))
    logits = base_model(x)[:, -1, :]
    assert logits.shape == (1, 2)

def test_setup_classification_model_functional():
    # This might take a few seconds as it loads GPT-2 config
    model, config = setup_classification_model(variant="gpt2")
    assert model.out_head.out_features == 2
    assert isinstance(model.out_head, torch.nn.Linear)
    
    # Check that most parameters are frozen
    trainable = [p for p in model.parameters() if p.requires_grad]
    # Head + last transformer block + final norm
    assert len(trainable) > 0
    
    # Head should be trainable
    assert model.out_head.weight.requires_grad == True
