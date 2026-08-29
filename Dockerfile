FROM arm64v8/ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libgl1 libglib2.0-0 libsm6 libxext6 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    opencv-python-headless \
    pyyaml \
    numpy \
    psutil \
    "ruamel.yaml"

WORKDIR /app
COPY stream_detector/ ./stream_detector/
COPY docker-entrypoint.sh /docker-entrypoint.sh

ENV CONFIG_PATH=/app/config.yaml

ENTRYPOINT ["/docker-entrypoint.sh"]
