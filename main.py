from typing import Dict, List, Optional
import numpy as np

from fastapi import FastAPI, UploadFile, BackgroundTasks, File

from uuid import uuid4
from io import BytesIO
from pydub import AudioSegment
from pydantic import BaseModel

import uvicorn
import logging

from whisper import transcribe, Segment

app = FastAPI()
logger = logging.getLogger('uvicorn.error')

# In-memory storage for task status
tasks = {}

def create_task_id() -> str:
    return str(uuid4())

class TranscribeTask(BaseModel):
    status: str
    error: Optional[str] = None
    segments: List[Segment] = []

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

@app.get("/transcribe/status/{task_id}")
async def check_status(task_id: str) -> TranscribeTask:
    # Retrieve the task by its ID
    task = tasks.get(task_id)

    # Check if the task exists and return its status
    if task:
        return task

    return {"status": "not found"}

async def process_audio_file(task_id: str, audio_bytes: bytes) -> None:
    try:
        # Use pydub to handle different audio formats and convert audio
        audio = AudioSegment.from_file(BytesIO(audio_bytes))
        audio = audio.set_channels(1).set_frame_rate(16000)

        # Convert the audio to a raw data byte string
        raw_data = audio.raw_data

        # Convert the raw data into a NumPy array for processing
        audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
        audio_np /= np.iinfo(np.int16).max  # Normalize to range [-1.0, 1.0]

        # Transcribe the audio using the Whisper model
        result = transcribe(audio_np)
        
        tasks[task_id] = TranscribeTask(status="completed", segments=result)
    except Exception as e:
        # Handle any errors that occur during the processing
        tasks[task_id] = TranscribeTask(status="failed", error=str(e))

@app.post("/transcribe")
async def transcribe_file(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
) -> Dict[str, str]:
    task_id = create_task_id()
    tasks[task_id] = TranscribeTask(status="processing")

    # Read the file into memory (bytes)
    audio_bytes = await file.read()

    background_tasks.add_task(process_audio_file, task_id, audio_bytes)

    return {"task_id": task_id}