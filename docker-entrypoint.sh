#!/bin/bash
set -e

# rknnlite 需要 /proc/device-tree/compatible 来识别 RK3588 平台
# 容器内 /proc 是独立 namespace，通过 SYS_ADMIN + mount bind 注入此文件
COMPAT_SRC="/rknn/compatible"
COMPAT_DST="/proc/device-tree/compatible"

if [ -f "$COMPAT_SRC" ] && [ ! -s "$COMPAT_DST" ]; then
    mkdir -p /proc/device-tree 2>/dev/null || true
    # 尝试 bind mount（需要 SYS_ADMIN cap）
    mount --bind "$COMPAT_SRC" "$COMPAT_DST" 2>/dev/null && \
        echo "INFO: /proc/device-tree/compatible mounted" || \
        echo "WARN: mount failed, rknnlite may not find compatible"
fi

exec python3 -m stream_detector
