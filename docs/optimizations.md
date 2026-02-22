# Quran AI Production Optimization Guide (CPU)

To ensure the system runs smoothly on CPU-only servers (e.g., standard AWS EC2 or DigitalOcean Droplets), we implement several optimization techniques.

## 1. Model Quantization (Dynamic INT8)
We can significantly reduce model size and increase inference speed by converting weights from FP32 to INT8.

```python
import torch
from transformers import Wav2Vec2ForCTC

def optimize_model_for_cpu(model):
    # Dynamic quantization for linear layers
    optimized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return optimized_model

# In main.py:
# phonetic_analyzer.model = optimize_model_for_cpu(phonetic_analyzer.model)
```

## 2. Whisper Optimization (CTranslate2)
For Whisper, using `Faster-Whisper` (which uses CTranslate2) can result in 4x speedup on CPU.
- **Action**: In production, replace `openai-whisper` with `faster-whisper`.
- **Latency**: Reduces base model transcription from ~5s to ~1.2s on standard CPUs.

## 3. Parallelism Control
Python's `torch` can sometimes over-subscribe CPU cores. Limit threads for predictable latency.
```python
import torch
torch.set_num_threads(4) # Match to your server's physical cores
```

## 4. API Scaling for Mobile
- **Statelessness**: The API is already stateless, allowing easy horizontal scaling behind a Load Balancer.
- **Audio Pre-processing**: Mobile clients should ideally compress audio (e.g., Opus in OGG or AAC) before uploading to reduce bandwidth.
- **Asynchronous Processing**: For longer recitations, implement a Task Queue (Celery/Redis) and return a `job_id`.
