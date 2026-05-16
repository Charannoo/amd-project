# ALCHEMY — runnable container (CPU). For MI300X, install ROCm PyTorch on the host instead.
FROM python:3.11-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# fair-esm + torch are heavy; adjust torch index in Dockerfile.*gpu if needed
COPY . .

ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0

EXPOSE 7860

# Optional: smina binary must be on PATH if BINDING_USE_SMINA=1
CMD ["python", "main.py"]
