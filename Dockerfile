# Download models
FROM alpine/git:2.43.0 as download

RUN apk add --no-cache wget curl

# Download INIVerse-MIX_Pony (using correct model ID)
RUN \
    echo "Downloading INIVerse-MIX_Pony model..." && \
    curl -L -H "Authorization: Bearer 71986aa96b44dfb5c0d1fcdeebde7a73" -o /INIVerse_Max.safetensors "https://civitai.com/api/download/models/1150354?type=Model&format=SafeTensor&size=full&fp=fp16" && \
    echo "Model downloaded. File size: $(wc -c < /INIVerse_Max.safetensors) bytes"

# Download Pony VAE
RUN \
    echo "Downloading Pony VAE..." && \
    curl -L -H "Authorization: Bearer 71986aa96b44dfb5c0d1fcdeebde7a73" -o /pony.vae.pt "https://civitai.com/api/download/models/290640?type=VAE&format=Other" && \
    echo "VAE downloaded. File size: $(wc -c < /pony.vae.pt) bytes"

# Download Detail Tweaker XL LoRA
RUN \
    echo "Downloading Detail Tweaker XL LoRA..." && \
    curl -L -H "Authorization: Bearer 71986aa96b44dfb5c0d1fcdeebde7a73" -o /add-detail-xl.safetensors "https://civitai.com/api/download/models/135867?type=Model&format=SafeTensor" && \
    echo "LoRA downloaded. File size: $(wc -c < /add-detail-xl.safetensors) bytes"

# Download 4x-Ultrasharp
RUN \
    echo "Downloading 4x-UltraSharp upscaler..." && \
    wget -q -O /4x-UltraSharp.pth "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth?download=true" && \
    echo "Upscaler downloaded: $(wc -c < /4x-UltraSharp.pth) bytes"

# Download ADetailer models
RUN \
    echo "Downloading ADetailer models..." && \
    mkdir -p /adetailer_models && \
    wget -q -O /adetailer_models/face_yolov8n.pt "https://huggingface.co/Bing-su/adetailer/resolve/main/models/face_yolov8n.pt" && \
    wget -q -O /adetailer_models/hand_yolov8n.pt "https://huggingface.co/Bing-su/adetailer/resolve/main/models/hand_yolov8n.pt" && \
    echo "ADetailer models downloaded"

# Build image
FROM python:3.10.14-slim

ARG A1111_RELEASE=v1.9.3
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    ROOT=/stable-diffusion-webui \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt install -y \
    fonts-dejavu-core rsync git jq moreutils aria2 wget libgoogle-perftools-dev libtcmalloc-minimal4 procps libgl1 libglib2.0-0 && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean -y

RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd stable-diffusion-webui && \
    git reset --hard ${A1111_RELEASE} && \
    pip install xformers && \
    pip install -r requirements_versions.txt && \
    python -c "from launch import prepare_environment; prepare_environment()" --skip-torch-cuda-test

# Install ADetailer extension
RUN git clone https://github.com/Bing-su/adetailer.git /stable-diffusion-webui/extensions/adetailer

# Create folders and copy models
RUN mkdir -p ${ROOT}/models/Stable-diffusion
RUN mkdir -p ${ROOT}/models/VAE
RUN mkdir -p ${ROOT}/models/Lora
RUN mkdir -p ${ROOT}/models/ESRGAN
RUN mkdir -p ${ROOT}/extensions/adetailer/models

COPY --from=download /INIVerse_Max.safetensors ${ROOT}/models/Stable-diffusion/
COPY --from=download /pony.vae.pt ${ROOT}/models/VAE/
COPY --from=download /add-detail-xl.safetensors ${ROOT}/models/Lora/
COPY --from=download /4x-UltraSharp.pth ${ROOT}/models/ESRGAN/
COPY --from=download /adetailer_models/*.pt ${ROOT}/extensions/adetailer/models/

# Verify models are present
RUN echo "Model verification:" && \
    ls -la ${ROOT}/models/Stable-diffusion/ && \
    ls -la ${ROOT}/models/VAE/ && \
    ls -la ${ROOT}/models/Lora/ && \
    ls -la ${ROOT}/models/ESRGAN/ && \
    ls -la ${ROOT}/extensions/adetailer/models/ && \
    echo "INIVerse_Max size: $(wc -c < ${ROOT}/models/Stable-diffusion/INIVerse_Max.safetensors) bytes" && \
    echo "Pony VAE size: $(wc -c < ${ROOT}/models/VAE/pony.vae.pt) bytes" && \
    echo "Detail Tweaker LoRA size: $(wc -c < ${ROOT}/models/Lora/add-detail-xl.safetensors) bytes"

# Install app dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY test_input.json .

# Copy application files from src directory to root
COPY src/ .
RUN chmod +x /start.sh

CMD /start.sh