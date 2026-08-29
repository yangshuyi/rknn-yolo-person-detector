FROM arm64v8/ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl \
    libgl1 libglib2.0-0 libsm6 libxext6 \
    libstdc++6 \
    && curl -fL -o /tmp/jellyfin-ffmpeg.deb \
       https://repo.jellyfin.org/files/ffmpeg/ubuntu/latest-7.x/arm64/jellyfin-ffmpeg7_7.1.4-3-noble_arm64.deb \
    && apt-get install -y /tmp/jellyfin-ffmpeg.deb \
    && rm -f /tmp/jellyfin-ffmpeg.deb \
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
