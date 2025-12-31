from fastapi import FastAPI, Query
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

app = FastAPI()

MODEL_NAME = "distilgpt2"

# Load model ONCE at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.eval()

@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "DistilGPT-2 API is running. Use /ask?prompt=Hi"
    }

@app.get("/ask")
def ask(prompt: str = Query(..., min_length=1, max_length=200)):
    start = time.time()

    inputs = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=50,      # keep SMALL
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "prompt": prompt,
        "response": text,
        "time_sec": round(time.time() - start, 2)
    }
