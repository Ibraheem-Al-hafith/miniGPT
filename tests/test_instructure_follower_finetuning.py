"""test_instructure_follower_finetuning.py — instruction data loaders and model loading."""
import json
import torch
import pytest


def test_split_and_get_loaders_returns_three_loaders(tmp_path):
    from finetune.instructure_follower_finetuning import split_and_get_loaders

    data = [
        {"instruction": f"Question {i}", "input": "", "output": f"Answer {i}"}
        for i in range(20)
    ]
    json_path = tmp_path / "instrution-data.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    train_loader, val_loader, test_loader = split_and_get_loaders(tmp_path)

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

    train_batch = next(iter(train_loader))
    assert isinstance(train_batch, tuple) and len(train_batch) == 2
    assert train_batch[0].shape[0] == 8
    assert train_batch[0].shape == train_batch[1].shape

    assert (tmp_path / "train.json").exists()
    assert (tmp_path / "test.json").exists()
    assert (tmp_path / "val.json").exists()


def test_split_and_get_loaders_raises_on_missing_json(tmp_path):
    from finetune.instructure_follower_finetuning import split_and_get_loaders

    with pytest.raises(FileNotFoundError):
        split_and_get_loaders(tmp_path)


def test_loading_model_returns_model_and_config_from_local_checkpoint(tmp_path, monkeypatch):
    from finetune.instructure_follower_finetuning import loading_model
    from model.gpt import GPTModel

    tiny_config = {
        "vocab_size": 256, "context_length": 16,
        "emb_dim": 32, "n_heads": 2, "n_layers": 2,
        "drop_rate": 0.0, "qkv_bias": False,
    }

    monkeypatch.setattr(
        "finetune.instructure_follower_finetuning.MODEL_CONFIG", tiny_config
    )

    model = GPTModel(tiny_config)
    ckpt_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), ckpt_path)

    loaded_model, config = loading_model(str(ckpt_path))

    assert isinstance(loaded_model, torch.nn.Module)
    assert isinstance(config, dict)
    assert config["context_length"] == tiny_config["context_length"]

    B, T = 2, tiny_config["context_length"]
    x = torch.randint(0, tiny_config["vocab_size"], (B, T))
    with torch.no_grad():
        logits = loaded_model(x)
    assert logits.shape == (B, T, tiny_config["vocab_size"])


def test_loading_model_from_hf_variant(monkeypatch):
    from finetune.instructure_follower_finetuning import loading_model

    mock_model = torch.nn.Linear(10, 10)
    mock_config = {"context_length": 128}

    def mock_load_from_hf(variant):
        return mock_model, mock_config

    monkeypatch.setattr(
        "finetune.instructure_follower_finetuning.load_from_hf", mock_load_from_hf
    )

    model, config = loading_model("gpt2")

    assert model is mock_model
    assert config is mock_config
