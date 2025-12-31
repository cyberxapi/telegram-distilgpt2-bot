import os
import logging
import time
from typing import Optional
from functools import lru_cache

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DistilGPT2 AI Chat API",
    description="Production-level API for DistilGPT2 text generation",
    version="1.0.0"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model caching
_tokenizer = None
_model = None
_device = None

def get_device():
    """Get the device (CPU/CUDA) to run the model on."""
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {_device}")
    return _device

def load_model():
    """Load tokenizer and model with lazy loading."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        try:
            logger.info("Loading DistilGPT2 model...")
            _tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            _model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            _model.to(get_device())
            _model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    return _tokenizer, _model

def generate_response(
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.95,
    num_beams: int = 1
) -> str:
    """
    Generate text using DistilGPT2 model.
    
    Args:
        prompt: Input text to generate from
        max_length: Maximum length of generated text
        temperature: Controls randomness (0.0 = deterministic, 1.0+ = random)
        top_p: Nucleus sampling parameter
        num_beams: Number of beams for beam search (1 = greedy)
    
    Returns:
        Generated text response
    """
    tokenizer, model = load_model()
    device = get_device()
    
    try:
        # Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate with constraints
        with torch.no_grad():
            output = model.generate(
                inputs,
                max_length=min(max_length, 200),  # Cap at 200 tokens
                temperature=max(0.1, min(temperature, 2.0)),  # Clamp temperature
                top_p=max(0.1, min(top_p, 1.0)),  # Clamp top_p
                num_beams=max(1, min(num_beams, 4)),  # Clamp beams
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=torch.ones_like(inputs),
                do_sample=True if num_beams == 1 else False,
            )
        
        # Decode output
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response
    
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Pre-load model on startup."""
    try:
        load_model()
        logger.info("Startup: Model pre-loaded successfully")
    except Exception as e:
        logger.warning(f"Startup: Model pre-loading failed: {str(e)}")

@app.get("/")
async def root():
    """Health check and API information."""
    return {
        "status": "online",
        "api": "DistilGPT2 Chat API",
        "version": "1.0.0",
        "usage": "GET /chat?ask=Your+question+here",
        "parameters": {
            "ask": "(required) Your prompt/question",
            "max_length": "(optional, default=100) Max response length",
            "temperature": "(optional, default=0.7) Randomness (0.1-2.0)",
            "top_p": "(optional, default=0.95) Diversity (0.1-1.0)",
            "num_beams": "(optional, default=1) Beam search (1-4)"
        }
    }

@app.get("/chat")
async def chat(
    ask: str = Query(..., min_length=1, max_length=500, description="Your question or prompt"),
    max_length: int = Query(100, ge=10, le=200, description="Max response length"),
    temperature: float = Query(0.7, ge=0.1, le=2.0, description="Response randomness"),
    top_p: float = Query(0.95, ge=0.1, le=1.0, description="Nucleus sampling"),
    num_beams: int = Query(1, ge=1, le=4, description="Beam search width")
):
    """
    Chat endpoint with DistilGPT2.
    
    Example: /chat?ask=Hello, how are you?
    """
    start_time = time.time()
    
    try:
        # Generate response
        response = generate_response(
            prompt=ask.strip(),
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams
        )
        
        inference_time = time.time() - start_time
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "prompt": ask,
                "response": response,
                "inference_time_ms": round(inference_time * 1000, 2),
                "model": "DistilGPT2",
                "parameters": {
                    "max_length": max_length,
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_beams": num_beams
                }
            }
        )
    
    except Exception as e:
        logger.error(f"Error in /chat: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "prompt": ask
            }
        )

@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    try:
        load_model()
        return {
            "status": "healthy",
            "model_loaded": True,
            "device": get_device(),
            "timestamp": time.time()
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "error": str(e)
            }
        )

@app.get("/models")
async def get_available_models():
    """List available models."""
    return {
        "available_models": ["distilgpt2"],
        "current_model": "distilgpt2",
        "device": get_device(),
        "description": "DistilGPT2 - Lightweight, fast text generation model"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
