# DistilGPT2 AI Chat API

A production-level REST API for DistilGPT2 text generation, deployed on Render.com free plan.

## Features

- **Fast & Lightweight**: Uses DistilGPT2 (~82M parameters) for efficient text generation
- **REST API**: Simple HTTP GET endpoint with query parameters
- **Production Ready**: Includes CORS, error handling, logging, and health checks
- **Configurable Generation**: Control temperature, diversity, and beam search
- **Performance Metrics**: Returns inference time for each request

## API Endpoints

### GET `/chat`
Generate text using DistilGPT2.

**Query Parameters:**
- `ask` (required): Your question or prompt
- `max_length` (optional, default=100): Max response length
- `temperature` (optional, default=0.7): Randomness (0.1-2.0)
- `top_p` (optional, default=0.95): Diversity (0.1-1.0)
- `num_beams` (optional, default=1): Beam search (1-4)

**Example:**
```bash
curl "http://localhost:8000/chat?ask=Hello, how are you?"
```

## Installation

1. Clone repository: `git clone https://github.com/cyberxapi/telegram-distilgpt2-bot`
2. Install deps: `pip install -r requirements.txt`
3. Run: `python main.py`

## Deploy on Render.com

1. Go to Render Dashboard
2. Create New Web Service
3. Connect GitHub repository
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Click Deploy

## Usage Examples

### JavaScript
```javascript
const ask = "What is AI?";
const res = await fetch(`https://api.onrender.com/chat?ask=${encodeURIComponent(ask)}`);
const data = await res.json();
console.log(data.response);
```

### Python
```python
import requests
res = requests.get("https://api.onrender.com/chat", params={"ask": "Hello"})
print(res.json()["response"])
```

## Model Info

- **Model**: DistilGPT2
- **Size**: ~336 MB
- **Parameters**: ~82M
- **Language**: English

## License

MIT License
