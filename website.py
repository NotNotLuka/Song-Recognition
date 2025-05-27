from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from song_processing import Classifier, Song
import numpy as np
import io
import subprocess
import asyncio
from config import Config


cfg = Config("config.yaml")


def decode_chunk(to_decode):
    p = subprocess.run(
        ["ffmpeg", "-loglevel", "quiet", "-hide_banner", "-f", "webm", "-i", "pipe:", "-vn", "-ar", str(cfg.SR), "-ac", "1", "-f", "wav", "-"],
        input = to_decode,
        stdout=subprocess.PIPE,
    )

    return p.stdout


def process_chunk(data, header, classifier):
    if header is not None:
        data = header[0] + data
    decoded = decode_chunk(data)
    signal, sr = Song.load_data(io.BytesIO(decoded))

    if header is None:
        header = (data, len(signal))
        return [], header
    
    offset = header[1]
    signal = signal[offset:]

    if len(signal) == 0: 
        return [], header
    
    classifier.add_data(signal)
    scores = classifier.get_current_score()

    return scores, header


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r") as f:
        content = f.read()
    return content


@app.websocket("/ws/upload")
async def websocket_receive_bytes(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    classifier = Classifier()
    classifier.start_thread()
    header = None

    try:
        while True:
            data = await websocket.receive_bytes()
            scores, header = process_chunk(data, header, classifier) 

            out = []
            for song in scores:
                title = classifier.database.get_song_name(song[0])
                score = str(round(song[1] * 100, 1)) + "%"
                out.append((title, score))

            await websocket.send_json(out)

    except WebSocketDisconnect:
        pass
    finally:
        await loop.run_in_executor(None, classifier.stop_thread)


if __name__ == "__main__":
    uvicorn.run("website:app", host="0.0.0.0", port=8000, reload=True)
