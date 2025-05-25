from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from song_processing import Classifier
import numpy as np
import scipy, io, av


def decode(audio, header=None, header_len=-1, sr=22050):
    if header is None:
        header = audio
    else:
        audio = header + audio
    buffer = io.BytesIO(audio)

    container = av.open(buffer)
    pcm = np.concatenate(
        [frame.to_ndarray().mean(axis=0) for frame in container.decode(audio=0)]
    )

    if header_len != -1:
        pcm = pcm[header_len:]

    pcm = scipy.signal.resample(pcm, sr)
    signal = pcm / np.max(np.abs(pcm))

    return signal


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r") as f:
        content = f.read()
    return content


@app.websocket("/ws/upload")
async def websocket_receive_bytes(websocket: WebSocket):
    classificator = Classifier()
    header = None
    header_len = -1
    full_signal = []
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            print("New data")
            signal = decode(data, header, header_len)
            full_signal = np.concatenate((full_signal, signal))

            if header is None:
                header = data
                header_len = len(signal)
            scores = classificator.add_data(signal)
            print(scores)

            await websocket.send_text(f"Received {len(data)} bytes")
    except WebSocketDisconnect:
        print("connection closed")
        pass
    scipy.io.wavfile.write(
        "out.wav", 22050, np.array(full_signal * 32767).astype(np.int16)
    )


if __name__ == "__main__":
    uvicorn.run("website:app", host="0.0.0.0", port=8000, reload=True)
