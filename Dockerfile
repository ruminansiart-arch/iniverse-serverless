# Download models
FROM alpine/git:2.43.0 as download
RUN apk add --no-cache wget curl

# Download INIVerse_Max
RUN curl -L -H "Authorization: Bearer 71986aa96b44dfb5c0d1fcdeebde7a73" -o /INIVerse_Max.safetensors "https://civitai.com/api/download/models/1150354?type=Model&format=SafeTensor&size=full&fp=fp16" && \
    echo "INIVerse_Max downloaded: $(wc -c < /INIVerse_Max.safetensors) bytes" 

# Download 4x-Ultrasharp
RUN wget -q -O /4x-UltraSharp.pth "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth?download=true" && \
    echo "4x-UltraSharp downloaded: $(wc -c < /4x-UltraSharp.pth) bytes"

# Download Detail Tweaker XL LoRA
RUN curl -L -H "Authorization: Bearer 71986aa96b44dfb5c0d1fcdeebde7a73" -o /detail_tweaker_xl.safetensors "https://civitai.com/api/download/models/135867?type=Model&format=SafeTensor" && \
    echo "Detail Tweaker XL LoRA downloaded: $(wc -c < /detail_tweaker_xl.safetensors) bytes"

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
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install xformers && \
    pip install -r requirements_versions.txt && \
    python -c "from launch import prepare_environment; prepare_environment()" --skip-torch-cuda-test

# Install ADetailer extension
RUN git clone https://github.com/Bing-su/adetailer.git /stable-diffusion-webui/extensions/adetailer

# Create folders and copy models
RUN mkdir -p ${ROOT}/models/Stable-diffusion
RUN mkdir -p ${ROOT}/models/ESRGAN
RUN mkdir -p ${ROOT}/models/Lora

COPY --from=download /INIVerse_Max.safetensors ${ROOT}/models/Stable-diffusion/
COPY --from=download /4x-UltraSharp.pth ${ROOT}/models/ESRGAN/
COPY --from=download /detail_tweaker_xl.safetensors ${ROOT}/models/Lora/

# Verify models are present
RUN echo "Model verification:" && \
    ls -la ${ROOT}/models/Stable-diffusion/ && \
    ls -la ${ROOT}/models/ESRGAN/ && \
    ls -la ${ROOT}/models/Lora/ && \
    echo "INIVerse_Max size: $(wc -c < ${ROOT}/models/Stable-diffusion/INIVerse_Max.safetensors) bytes" && \
    echo "4x-UltraSharp size: $(wc -c < ${ROOT}/models/ESRGAN/4x-UltraSharp.pth) bytes" && \
    echo "Detail Tweaker XL LoRA size: $(wc -c < ${ROOT}/models/Lora/detail_tweaker_xl.safetensors) bytes"

# Install app dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY test_input.json .
ADD src .
RUN chmod +x /start.sh
CMD /start.sh
