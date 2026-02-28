"""
SpeechToTextDaemon.py

Runs as a background daemon alongside BAR.
The Lua widget `gui_speech_to_chat.lua` communicates via two files:

  stt_trigger.flag  – widget writes this to start a recording
  stt_stop.flag     – widget writes this (key released) to stop recording and finalise
  stt_cancel.flag   – widget writes this to abort and discard the recording
  stt_result.txt    – daemon writes the transcription here when done
  stt_log.txt       – shared log file (daemon appends to it)

Usage (WSL with ffmpeg audio pipe):
  ffmpeg.exe -f dshow -i audio="<mic name>" -ar 16000 -ac 1 -f s16le - 2>/dev/null \
    | python ./tools/SpeechToTextDaemon.py

Usage (normal OS with a working mic):
  python ./tools/SpeechToTextDaemon.py
"""

from mistralai import Mistral
from mistralai.extra.realtime import UnknownRealtimeEvent
from mistralai.models import (
    AudioFormat,
    RealtimeTranscriptionError,
    RealtimeTranscriptionSessionCreated,
    TranscriptionStreamDone,
    TranscriptionStreamTextDelta,
)

import asyncio
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

# Try to import sounddevice; may be unavailable in WSL
try:
    import sounddevice as sd
    _sd_available = sd.query_devices(kind="input") is not None
except Exception:
    sd = None
    _sd_available = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent

def resolve_ipc_dir(override: str | None) -> Path:
    """
    Return the directory used for IPC files.
    Default: the Spring write-data LuaUI/ folder, which sits 3 levels above
    this script:  tools/ -> BAR.sdd/ -> games/ -> data/ -> LuaUI/
    Override with --ipc-dir if your setup differs.
    """
    if override:
        return Path(override)
    # Derive from script location: tools → BAR.sdd → games → data
    spring_data_dir = SCRIPT_DIR.parent.parent.parent
    return spring_data_dir / "LuaUI"

# Resolved after arg parsing (see __main__)
IPC_DIR      = None   # set in main
TRIGGER_FILE = None
STOP_FILE    = None
CANCEL_FILE  = None
DONE_FILE    = None
RESULT_FILE  = None
LOG_FILE     = None

# ---------------------------------------------------------------------------
# Asyncio coordination (initialised in main())
# ---------------------------------------------------------------------------
_audio_q    = None   # asyncio.Queue[bytes | None] — shared audio ring
_start_evt  = None   # set when trigger file detected (key pressed)
_stop_evt   = None   # set when stop file detected (key released)
_cancel_evt = None   # set when cancel file detected

# ---------------------------------------------------------------------------
# API / audio config
# ---------------------------------------------------------------------------
API_KEY     = "ybeqehOv7gWxBX5mycJqUycR0U7Wfgdv"
SAMPLE_RATE = 16000
CHANNELS    = 1
BLOCK_SIZE  = 4096
CHUNK_SIZE  = BLOCK_SIZE * 2   # bytes (int16 = 2 bytes per sample)

audio_format = AudioFormat(encoding="pcm_s16le", sample_rate=SAMPLE_RATE)
client       = Mistral(api_key=API_KEY)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ---------------------------------------------------------------------------
# Audio streams
# ---------------------------------------------------------------------------
async def microphone_stream() -> AsyncIterator[bytes]:
    """Capture PCM s16le from the default microphone via sounddevice."""
    loop = asyncio.get_event_loop()
    q: asyncio.Queue[bytes] = asyncio.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            log(f"[sounddevice] {status}")
        loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=BLOCK_SIZE,
        callback=callback,
    ):
        log("Microphone open via sounddevice.")
        while True:
            yield await q.get()


async def stdin_stream() -> AsyncIterator[bytes]:
    """Read raw PCM s16le chunks from stdin (piped from ffmpeg)."""
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)
    log("Reading audio from stdin (ffmpeg pipe).")
    while True:
        chunk = await reader.read(CHUNK_SIZE)
        if not chunk:
            break
        yield chunk


async def audio_reader() -> None:
    """
    Background task: continuously reads from mic or stdin and pushes chunks
    into the shared _audio_q.  Runs for the entire lifetime of the daemon.
    """
    if _sd_available:
        source = microphone_stream()
    else:
        source = stdin_stream()
    log("Audio reader started.")
    async for chunk in source:
        await _audio_q.put(chunk)
    await _audio_q.put(None)   # EOF sentinel
    log("Audio reader finished (EOF).")


async def gated_audio_stream() -> AsyncIterator[bytes]:
    """
    Yields audio chunks to the Mistral API only while the user is holding the
    push-to-talk key.

    Lifecycle per call (one WebSocket session):
      1. Block here until _start_evt fires  (WebSocket is already warm at this
         point — it was opened before the user pressed the key).
      2. Drain any queued chunks that arrived before the key press.
      3. Yield live chunks until _stop_evt (graceful) or _cancel_evt (abort).
    """
    # Wait for the trigger — WebSocket is already connected at this point
    await _start_evt.wait()
    _start_evt.clear()

    # Discard stale audio that built up between sessions
    drained = 0
    while not _audio_q.empty():
        try:
            _audio_q.get_nowait()
            drained += 1
        except asyncio.QueueEmpty:
            break
    if drained:
        log(f"Drained {drained} stale audio chunk(s) before recording.")

    log("Recording — streaming live audio to API.")

    while True:
        try:
            chunk = await asyncio.wait_for(_audio_q.get(), timeout=0.5)
        except asyncio.TimeoutError:
            if _cancel_evt.is_set():
                _cancel_evt.clear()
                log("Cancel during wait — aborting.")
                return
            continue

        if chunk is None:   # reader EOF
            return

        if _stop_evt.is_set():
            _stop_evt.clear()
            yield chunk     # include last chunk so the API gets a clean ending
            # Drain leftovers produced between stop and here
            while not _audio_q.empty():
                try: _audio_q.get_nowait()
                except asyncio.QueueEmpty: break
            return

        if _cancel_evt.is_set():
            _cancel_evt.clear()
            while not _audio_q.empty():
                try: _audio_q.get_nowait()
                except asyncio.QueueEmpty: break
            log("Cancel signal — aborting session.")
            return

        yield chunk


# ---------------------------------------------------------------------------
# Session loop  (persistent warm WebSocket)
# ---------------------------------------------------------------------------
async def session_loop() -> None:
    """
    Maintains one Mistral WebSocket session at all times.
    Each iteration:
      1. Opens a new WebSocket connection (pre-connect).
      2. gated_audio_stream() blocks until the user presses the key,
         so the connection sits warm but idle.
      3. Audio flows as soon as _start_evt fires — zero handshake delay.
      4. When the stream ends the transcription result is written and the loop
         immediately opens the next WebSocket ready for the next key press.
    """
    log("Session loop started — pre-connecting WebSocket.")

    while True:
        text_parts: list[str] = ["@agent "]
        aborted = False

        try:
            async for event in client.audio.realtime.transcribe_stream(
                audio_stream=gated_audio_stream(),
                model="voxtral-mini-transcribe-realtime-2602",
                audio_format=audio_format,
            ):
                if isinstance(event, RealtimeTranscriptionSessionCreated):
                    log("WebSocket warm — waiting for key press.")
                elif isinstance(event, TranscriptionStreamTextDelta):
                    text_parts.append(event.text)
                    print(event.text, end="", flush=True)
                elif isinstance(event, TranscriptionStreamDone):
                    log("Transcription complete.")
                elif isinstance(event, RealtimeTranscriptionError):
                    log(f"Transcription error: {event}")
                    aborted = True
                elif isinstance(event, UnknownRealtimeEvent):
                    log(f"Unknown event: {event}")
        except Exception as exc:
            log(f"WebSocket error: {exc}")
            aborted = True
            await asyncio.sleep(1.0)   # back-off before reconnect on error

        if not aborted:
            text = "".join(text_parts).strip()
            if text:
                log(f"Writing result: {text!r}")
                RESULT_FILE.write_text(text, encoding="utf-8")
            else:
                log("Empty transcription.")
            DONE_FILE.write_text("done", encoding="utf-8")
            log("Done flag written — pre-connecting next WebSocket.")

        # Tiny pause then immediately open the next warm connection
        await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Trigger loop  (file polling at 50 ms)
# ---------------------------------------------------------------------------
async def trigger_loop() -> None:
    log(f"Trigger loop started (50 ms poll): {TRIGGER_FILE}")

    while True:
        await asyncio.sleep(0.05)

        if CANCEL_FILE.exists():
            CANCEL_FILE.unlink(missing_ok=True)
            _cancel_evt.set()
            log("Cancel file detected.")

        if STOP_FILE.exists():
            STOP_FILE.unlink(missing_ok=True)
            _stop_evt.set()
            log("Stop file detected — key released.")

        if TRIGGER_FILE.exists():
            TRIGGER_FILE.unlink(missing_ok=True)
            DONE_FILE.unlink(missing_ok=True)
            RESULT_FILE.unlink(missing_ok=True)
            _start_evt.set()
            log("Trigger file detected — key pressed.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def main() -> None:
    global _audio_q, _start_evt, _stop_evt, _cancel_evt

    _audio_q    = asyncio.Queue()
    _start_evt  = asyncio.Event()
    _stop_evt   = asyncio.Event()
    _cancel_evt = asyncio.Event()

    log("=== SpeechToTextDaemon started ===")
    log(f"IPC dir  : {IPC_DIR}")
    log(f"Log file : {LOG_FILE}")
    log(f"Mic mode : {'sounddevice' if _sd_available else 'stdin (ffmpeg pipe)'}")

    for f in [TRIGGER_FILE, STOP_FILE, CANCEL_FILE, DONE_FILE, RESULT_FILE]:
        f.unlink(missing_ok=True)

    await asyncio.gather(
        audio_reader(),
        session_loop(),
        trigger_loop(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BAR Speech-to-Text daemon")
    parser.add_argument(
        "--ipc-dir",
        default=None,
        help="Directory for IPC files (trigger/result/log). "
             "Defaults to the Spring write-data LuaUI/ folder "
             "(3 levels above this script: tools → BAR.sdd → games → data → LuaUI).",
    )
    args = parser.parse_args()

    # Resolve and set global paths
    IPC_DIR      = resolve_ipc_dir(args.ipc_dir)
    IPC_DIR.mkdir(parents=True, exist_ok=True)
    TRIGGER_FILE = IPC_DIR / "stt_trigger.flag"
    STOP_FILE    = IPC_DIR / "stt_stop.flag"
    CANCEL_FILE  = IPC_DIR / "stt_cancel.flag"
    DONE_FILE    = IPC_DIR / "stt_done.flag"
    RESULT_FILE  = IPC_DIR / "stt_result.txt"
    LOG_FILE     = IPC_DIR / "stt_log.txt"

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Daemon stopped by user.")
