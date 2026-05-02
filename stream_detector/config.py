from dataclasses import dataclass, field
from typing import List
import yaml


@dataclass
class ModelConfig:
    path: str = 'models/yolo11n.rknn'
    input_size: int = 640
    conf_thresh: float = 0.25
    nms_thresh: float = 0.45
    target_classes: List[str] = field(default_factory=lambda: ['person'])


@dataclass
class ProcessingConfig:
    process_fps: int = 10
    max_delay_ms: int = 1000


@dataclass
class OutputConfig:
    mjpeg_port: int = 8090
    jpeg_quality: int = 85


@dataclass
class StreamConfig:
    name: str
    input: str


@dataclass
class Config:
    model: ModelConfig
    processing: ProcessingConfig
    output: OutputConfig
    streams: List[StreamConfig]


def load_config(path: str) -> Config:
    with open(path) as f:
        d = yaml.safe_load(f)
    return Config(
        model=ModelConfig(**d.get('model', {})),
        processing=ProcessingConfig(**d.get('processing', {})),
        output=OutputConfig(**d.get('output', {})),
        streams=[StreamConfig(**s) for s in d['streams']],
    )
