"""
ui_finetune/server.py — FastAPI backend for Fine-tuned models (Instruction & Classification).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import tiktoken
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from config import MODEL_CONFIG, CKPT_DIR, GEN_MAX_TOKENS, GEN_TEMPERATURE, GEN_TOP_K, GEN_TOP_P, GEN_BEAMS, VARIANT
from inference.generate import generate

# ── Load model ────────────────────────────────────────────────────────────────

device    = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = tiktoken.encoding_for_model("gpt2")
model     = None
config    = MODEL_CONFIG
mode      = "instruction"  # "instruction" or "classification"
loaded_ckpt = None

def load_instruction(hf_variant=None):
    global model, config, mode, loaded_ckpt
    mode = "instruction"
    loaded_ckpt = None
    from finetune.instructure_follower_finetuning import loading_model
    v = hf_variant or VARIANT
    
    print(f"Loading base model: {v}")
    model, config = loading_model(v)
    
    # Try to load local checkpoint if not explicitly loading from HF
    if hf_variant is None:
        ckpts = sorted(Path(CKPT_DIR).glob("epoch_*.pt"))
        if ckpts:
            latest_ckpt = ckpts[-1]
            try:
                state_dict = torch.load(latest_ckpt, map_location=device)
                # Check for size mismatch to avoid loading classification head into generation head
                if "out_head.weight" in state_dict:
                    ckpt_vocab_size = state_dict["out_head.weight"].shape[0]
                    model_vocab_size = model.out_head.weight.shape[0]
                    if ckpt_vocab_size == model_vocab_size:
                        model.load_state_dict(state_dict)
                        loaded_ckpt = latest_ckpt.name
                        print(f"Loaded instruction weights from {latest_ckpt}")
            except Exception as e:
                print(f"Could not load {latest_ckpt.name}: {e}")
            
    model = model.to(device)
    model.eval()
    if not loaded_ckpt:
        print("Warning: No instruction checkpoint found. Using base model.")

def load_classification(hf_variant=None):
    global model, config, mode, loaded_ckpt
    mode = "classification"
    loaded_ckpt = None
    from finetune.classification_finetuning import setup_classification_model
    v = hf_variant or VARIANT
    
    print(f"Setting up classification model based on: {v}")
    model, config = setup_classification_model(v)
    
    # Load best_model.pt or latest epoch if it exists
    ckpt_path = Path(CKPT_DIR) / "best_model.pt"
    if not ckpt_path.exists():
        # Fallback to latest epoch if best_model doesn't exist
        ckpts = sorted(Path(CKPT_DIR).glob("epoch_*.pt"))
        if ckpts:
            ckpt_path = ckpts[-1]
            
    if ckpt_path.exists():
        try:
            state_dict = torch.load(ckpt_path, map_location=device)
            # Check for size mismatch (e.g. 2 classes for Spam/Ham)
            if "out_head.weight" in state_dict:
                ckpt_classes = state_dict["out_head.weight"].shape[0]
                model_classes = model.out_head.weight.shape[0]
                if ckpt_classes == model_classes:
                    model.load_state_dict(state_dict)
                    loaded_ckpt = ckpt_path.name
                    print(f"Loaded classification weights from {ckpt_path}")
        except Exception as e:
            print(f"Error loading {ckpt_path.name}: {e}")
        
    model = model.to(device)
    model.eval()
    if not loaded_ckpt:
        print("Warning: No classification checkpoint found. Using base model with random head.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="SAIR miniGPT Fine-tune UI")

UI_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(UI_DIR / "index.html")


# ── Endpoints ─────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt:      str
    max_tokens:  int   = GEN_MAX_TOKENS
    temperature: float = GEN_TEMPERATURE
    top_k:       int   = GEN_TOP_K
    top_p:       float = GEN_TOP_P
    beams:       int   = GEN_BEAMS
    method:      str   = "nucleus"

@app.post("/generate")
def generate_endpoint(req: GenerateRequest):
    if model is None:
        return {"error": "Model not loaded"}
    if mode != "instruction":
        return {"error": "Model is in classification mode."}
        
    text = generate(
        model          = model,
        prompt         = req.prompt,
        max_new_tokens = req.max_tokens,
        context_size   = config["context_length"],
        tokenizer      = tokenizer,
        device         = device,
        temperature    = req.temperature,
        top_k          = req.top_k,
        top_p          = req.top_p,
        beams          = req.beams,
        method         = req.method,
    )
    return {"text": text}


class ClassifyRequest(BaseModel):
    text: str

@app.post("/classify")
def classify_endpoint(req: ClassifyRequest):
    if model is None:
        return {"error": "Model not loaded"}
    if mode != "classification":
        return {"error": "Model is in instruction mode."}

    # Tokenize
    encoded = tokenizer.encode(req.text)
    # Truncate to context length
    encoded = encoded[:config["context_length"]]
    x = torch.tensor([encoded]).to(device)
    
    with torch.no_grad():
        logits = model(x)[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=-1).item()
        
    # Verified label mapping: 0=Spam, 1=Ham
    labels = ["Spam", "Ham"]
    label = labels[pred] if pred < len(labels) else str(pred)
    
    return {
        "label": label,
        "confidence": probs[0, pred].item(),
        "probabilities": probs[0].tolist()
    }


@app.get("/info")
def info():
    return {
        "device" : device,
        "params" : sum(p.numel() for p in model.parameters()) if model else 0,
        "config" : config,
        "mode"   : mode,
        "checkpoint": loaded_ckpt
    }


class SwitchRequest(BaseModel):
    task: str


@app.post("/switch")
def switch_task(req: SwitchRequest):
    global model
    try:
        # Clear memory
        model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if req.task == "classification":
            load_classification()
        else:
            load_instruction()
            
        return {"status": "success", "mode": mode}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}


# ── Entry point ───────────────────────────────────────────────────────────────

def run(task="instruction", hf_variant=None, host="0.0.0.0", port=7861):
    if task == "classification":
        load_classification(hf_variant)
    else:
        load_instruction(hf_variant)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["instruction", "classification"], default="instruction")
    parser.add_argument("--hf", type=str, default=None)
    args = parser.parse_args()
    run(task=args.task, hf_variant=args.hf)
