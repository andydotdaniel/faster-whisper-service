from faster_whisper import WhisperModel
import numpy as np
from pydantic import BaseModel

from typing import List
import logging

model_size = "small"

logger = logging.getLogger('uvicorn.error')

class Segment():
    text: str
    start: float
    end: float

    def __init__(self, text: str, start: float, end: float) -> None:
        self.text = text
        self.start = start
        self.end = end

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")

def transcribe(audio: np.ndarray) -> List[Segment]:
    segments, info = model.transcribe(audio, beam_size=5)
    logger.info("[TRANSCRIBE] Detected language '%s' with probability %f" % (info.language, info.language_probability))

    return segments