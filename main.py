import os
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import httpx
import asyncio

app = FastAPI()

# Model setup
MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

class Update(BaseModel):
    update_id: int
    message: dict = None

def generate_response(prompt: str, max_length: int = 100) -> str:
    """Generate text using DistilGPT2."""
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            inputs,
            max_length=max_length,
            num_beams=2,
            top_p=0.95,
            temperature=0.7,
            do_sample=True
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

async def send_telegram_message(chat_id: int, text: str):
    """Send message back to Telegram."""
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{TELEGRAM_API_URL}/sendMessage",
            json={"chat_id": chat_id, "text": text}
        )

@app.post("/webhook")
async def webhook(update: Update):
    """Handle incoming Telegram messages."""
    if update.message is None:
        return {"ok": True}
    
    chat_id = update.message["chat"]["id"]
    user_message = update.message.get("text", "")
    
    if not user_message:
        return {"ok": True}
    
    # Generate response
    response_text = generate_response(user_message)
    
    # Send response back
    await send_telegram_message(chat_id, response_text)
    
    return {"ok": True}

@app.get("/")
def read_root():
    return {"status": "Telegram DistilGPT2 Bot is running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
