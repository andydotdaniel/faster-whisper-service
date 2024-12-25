from faster_whisper import WhisperModel
import numpy as np
from pydantic import BaseModel

from typing import List

model_size = "small"

class Segment(BaseModel):
    text: str
    start: float
    end: float

    def __init__(self, text: str, start: float, end: float) -> None:
        super().__init__(text=text, start=start, end=end)

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")

def format_segments(segments) -> List[Segment]:
    return [Segment(segment.text, segment.start, segment.end) for segment in segments]

def transcribe(audio: np.ndarray) -> List[Segment]:
    segments, info = model.transcribe(audio, beam_size=5)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    return format_segments(segments)