from mistralai import Mistral
from mistralai.extra.realtime import UnknownRealtimeEvent
from mistralai.models import AudioFormat, RealtimeTranscriptionError, RealtimeTranscriptionSessionCreated, TranscriptionStreamDone, TranscriptionStreamTextDelta

import asyncio
import sys
import time
from typing import AsyncIterator

import numpy as np

# Try to import sounddevice; it may be unavailable (e.g. no PortAudio in WSL)
try:
    import sounddevice as sd
    _sd_available = sd.query_devices(kind="input") is not None
except Exception:
    sd = None
    _sd_available = False

api_key = "ybeqehOv7gWxBX5mycJqUycR0U7Wfgdv"
client = Mistral(api_key=api_key)

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 4096  # frames per callback (~256ms at 16kHz)
CHUNK_SIZE = BLOCK_SIZE * 2  # bytes (int16 = 2 bytes per sample)

audio_format = AudioFormat(encoding="pcm_s16le", sample_rate=SAMPLE_RATE)


async def microphone_stream() -> AsyncIterator[bytes]:
    """Capture PCM s16le from the default microphone via sounddevice."""
    loop = asyncio.get_event_loop()
    q: asyncio.Queue[bytes] = asyncio.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(f"[sounddevice] {status}", file=sys.stderr)
        loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=BLOCK_SIZE,
        callback=callback,
    ):
        print("Microphone open — speak now. Press Ctrl+C to stop.", flush=True)
        while True:
            yield await q.get()


async def stdin_stream() -> AsyncIterator[bytes]:
    """Read raw PCM s16le chunks from stdin (piped from ffmpeg)."""
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)
    print("Reading audio from stdin — speak now. Press Ctrl+C to stop.", flush=True)
    while True:
        chunk = await reader.read(CHUNK_SIZE)
        if not chunk:
            break
        yield chunk


async def silence_gated_stream(
    source: AsyncIterator[bytes],
    rms_threshold: float = 150.0,
    silence_timeout: float = 2.0,
) -> AsyncIterator[bytes]:
    """Wrap an audio stream and stop it after `silence_timeout` seconds of silence."""
    last_sound = time.monotonic()
    started = False  # don't time out before the user has spoken at all
    async for chunk in source:
        rms = np.sqrt(np.mean(np.frombuffer(chunk, dtype=np.int16).astype(np.float32) ** 2))
        if rms >= rms_threshold:
            last_sound = time.monotonic()
            started = True
        elif started and (time.monotonic() - last_sound) >= silence_timeout:
            print(f"\n[silence > {silence_timeout}s — stopping]", flush=True)
            break
        yield chunk


def audio_stream() -> AsyncIterator[bytes]:
    if _sd_available:
        return silence_gated_stream(microphone_stream())
    print("No microphone detected via sounddevice; reading from stdin.", file=sys.stderr)
    print("Run: ffmpeg -f dshow -i audio=\"<mic name>\" -ar 16000 -ac 1 -f s16le - 2>/dev/null | python SpeechToText.py", file=sys.stderr)
    return silence_gated_stream(stdin_stream())


async def main():
    try:
        async for event in client.audio.realtime.transcribe_stream(
            audio_stream=audio_stream(),
            model="voxtral-mini-transcribe-realtime-2602",
            audio_format=audio_format,
        ):
            if isinstance(event, RealtimeTranscriptionSessionCreated):
                print(f"\nSession created.", flush=True)
            elif isinstance(event, TranscriptionStreamTextDelta):
                print(event.text, end="", flush=True)
            elif isinstance(event, TranscriptionStreamDone):
                print("\nTranscription done.", flush=True)
            elif isinstance(event, RealtimeTranscriptionError):
                print(f"\nError: {event}", flush=True)
            elif isinstance(event, UnknownRealtimeEvent):
                print(f"\nUnknown event: {event}", flush=True)
    except KeyboardInterrupt:
        print("\nStopping...")

sys.exit(asyncio.run(main()))