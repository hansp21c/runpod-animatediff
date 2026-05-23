FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
       python3-pip python3-dev build-essential \
       ffmpeg \
       libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ldconfig /usr/local/cuda-12.1/compat/ || true

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 torchvision==0.18.1

COPY requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir -r /requirements.txt

COPY . .

ENV ANIMATEDIFF_BASE=emilianJR/epiCRealism \
    ANIMATEDIFF_MOTION=guoyww/animatediff-motion-adapter-v1-5-3 \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

CMD ["python3", "-u", "handler.py"]
