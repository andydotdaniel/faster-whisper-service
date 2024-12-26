from typing import Dict, List, Optional
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from io import BytesIO
from pydub import AudioSegment

import uvicorn

from whisper import transcribe, Segment
from utilities import seconds_to_timestamp

app = FastAPI()

def process_audio_file(audio_bytes: bytes):
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
        segments = transcribe(audio_np)

        for segment in segments:
            start_timestamp = seconds_to_timestamp(segment.start)
            end_timestamp = seconds_to_timestamp(segment.end)
            result = f"{start_timestamp} -> {end_timestamp}:\n{segment.text.strip()}"

            yield f"data: {result}\n\n"

    except Exception as e:
        # Handle any errors that occur during the processing
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    # Read the file into memory (bytes)
    audio_bytes = await file.read()

    return StreamingResponse(process_audio_file(audio_bytes), media_type="text/event-stream")