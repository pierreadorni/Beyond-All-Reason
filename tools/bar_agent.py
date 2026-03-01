#!/usr/bin/env python3
"""
bar_agent.py  –  Beyond All Reason AI Agent
============================================
Connects to the in-game AgentBridge HTTP server, listens to in-game chat,
and responds with real commands via a Strands + Mistral LLM agent.

Usage
-----
    export MISTRAL_API_KEY="your_key_here"
    export ELEVENLABS_API_KEY="your_key_here"   # optional, enables TTS
    python tools/bar_agent.py

Optional env vars:
    BAR_HOST         (default: 127.0.0.1)
    BAR_PORT         (default: 7654)
    MISTRAL_MODEL    (default: mistral-large-latest)
    AGENT_PREFIX     – chat prefix that triggers the agent (default: @agent)
    AGENT_PREFIX2    – alternative prefix (default: !)
    ELEVENLABS_API_KEY – enables TTS voice output via ElevenLabs
    EL_VOICE_ID      – ElevenLabs voice ID (default: Adam)
    EL_MODEL_ID      – ElevenLabs model  (default: eleven_turbo_v2_5)
    EL_AMP_SCALE     – RMS→0-1 multiplier for shake effect (default: 5.0)
    TTS_SFX_PATH     – background crackle path (default: sounds/voice-soundeffects/AMBSci-Soft,...)
    TTS_PREFIX_PATH  – prefix beep path (default: sounds/voice-soundeffects/prefix_communication_sound.mp3)
    TTS_SUFFIX_PATH  – suffix sound path (default: sounds/voice-soundeffects/suffix_transmission.mp3)
    TTS_SFX_VOLUME_DB    – crackle gain in dB (default: 10)
    TTS_PREFIX_VOLUME_DB – prefix gain in dB (default: 0)
    TTS_SUFFIX_VOLUME_DB – suffix gain in dB (default: 0)

Requirements
------------
    pip install 'strands-agents[mistral]' strands-agents-tools
    pip install elevenlabs sounddevice numpy   # optional, for TTS
    pip install pydub                          # optional, for SFX overlay
"""

import asyncio
import json
import os
import datetime
import queue as _queue
import subprocess
import sys
import time
import threading
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, AsyncIterator

# ---------------------------------------------------------------------------
# Strands / Mistral imports
# ---------------------------------------------------------------------------
try:
    from strands import Agent, tool
    from strands.models.mistral import MistralModel
except ImportError:
    sys.exit(
        "Missing dependency. Please run:\n" "  pip install 'strands-agents[mistral]'\n"
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HOST = os.environ.get("BAR_HOST", "127.0.0.1")
PORT = int(os.environ.get("BAR_PORT", "7654"))
BASE_URL = f"http://{HOST}:{PORT}"
MODEL_ID = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
AGENT_PREFIX = os.environ.get("AGENT_PREFIX", "@agent")
AGENT_PREFIX2 = os.environ.get("AGENT_PREFIX2", "!")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "1.0"))  # seconds

# ---------------------------------------------------------------------------
# STT / Voxtral configuration
# ---------------------------------------------------------------------------
# IPC files: default to the Spring write-data LuaUI/ folder
# (tools/ → BAR.sdd/ → games/ → data/ → LuaUI/)
_SCRIPT_DIR = Path(__file__).parent
_IPC_DIR = Path(
    os.environ.get("STT_IPC_DIR", str(_SCRIPT_DIR.parent.parent.parent / "LuaUI"))
)
_IPC_DIR.mkdir(parents=True, exist_ok=True)
STT_TRIGGER_FILE = _IPC_DIR / "stt_trigger.flag"
STT_STOP_FILE = _IPC_DIR / "stt_stop.flag"
STT_CANCEL_FILE = _IPC_DIR / "stt_cancel.flag"
STT_DONE_FILE = _IPC_DIR / "stt_done.flag"
STT_RESULT_FILE = _IPC_DIR / "stt_result.txt"  # written for Lua HUD display

STT_SAMPLE_RATE = 16000
STT_CHANNELS = 1
STT_BLOCK_SIZE = 4096
STT_CHUNK_SIZE = STT_BLOCK_SIZE * 2  # bytes (int16)
STT_MODEL = "voxtral-mini-transcribe-realtime-2602"

# Check for sounddevice input/output (both may be unavailable in WSL)
_sd_output_available = False  # updated below and again inside ElevenLabs block
try:
    import sounddevice as _sd

    _sd_available = _sd.query_devices(kind="input") is not None
    try:
        _sd_output_available = _sd.query_devices(kind="output") is not None
    except Exception:
        _sd_output_available = False
except Exception:
    _sd = None
    _sd_available = False
    _sd_output_available = False

# ElevenLabs TTS (all optional — TTS silently disabled when unset/missing)
EL_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
EL_VOICE_ID = os.environ.get("EL_VOICE_ID", "HhHF7uHMEx2kpTutnYTv")  # Adam
EL_MODEL_ID = os.environ.get("EL_MODEL_ID", "eleven_v3")
EL_AMP_SCALE = float(os.environ.get("EL_AMP_SCALE", "5.0"))  # RMS → 0–1
# Sound-effect overlay paths for TTS (pydub required; set to empty string to disable)
_BAR_ROOT = _SCRIPT_DIR.parent
TTS_SFX_PATH = os.environ.get(
    "TTS_SFX_PATH",
    str(_BAR_ROOT / "sounds/voice-soundeffects/AMBSci-Soft,_crackling_old_-Elevenlabs.mp3"),
)
TTS_PREFIX_PATH = os.environ.get(
    "TTS_PREFIX_PATH",
    str(_BAR_ROOT / "sounds/voice-soundeffects/prefix_communication_sound.mp3"),
)
TTS_SUFFIX_PATH = os.environ.get(
    "TTS_SUFFIX_PATH",
    str(_BAR_ROOT / "sounds/voice-soundeffects/suffix_transmission.mp3"),
)
TTS_SFX_VOLUME_DB    = float(os.environ.get("TTS_SFX_VOLUME_DB",    "10"))
TTS_PREFIX_VOLUME_DB = float(os.environ.get("TTS_PREFIX_VOLUME_DB",  "10"))
TTS_SUFFIX_VOLUME_DB = float(os.environ.get("TTS_SUFFIX_VOLUME_DB",  "10"))
# Fallback TTS player when sounddevice output is unavailable (e.g. WSL).
# Raw PCM s16le 16 kHz mono is written to its stdin.
TTS_PLAYER_CMD = os.environ.get(
    "TTS_PLAYER_CMD",
    "ffplay.exe -f s16le -ar 16000 -nodisp -autoexit -",
)
# Optional: index or substring of the output device name for sounddevice.
# Leave unset to use the system default output device.
# List devices: python -c "import sounddevice; print(sounddevice.query_devices())"
SD_OUTPUT_DEVICE: "int | str | None" = os.environ.get("SD_OUTPUT_DEVICE") or None
try:
    if SD_OUTPUT_DEVICE is not None:
        SD_OUTPUT_DEVICE = int(SD_OUTPUT_DEVICE)  # type: ignore[assignment]
except ValueError:
    pass  # keep as string (name substring)

# ---------------------------------------------------------------------------
# ElevenLabs TTS engine  (optional)
# ---------------------------------------------------------------------------
_el_client = None  # ElevenLabs client (None when unavailable)
_tts_lock = threading.Lock()  # serialises concurrent speak() calls

# pydub for SFX mixing (optional)
_pydub_available = False
try:
    from pydub import AudioSegment as _AudioSegment
    _pydub_available = True
except ImportError:
    pass

try:
    from elevenlabs import ElevenLabs as _ElevenLabs
    import numpy as _np
    import sounddevice as _sd

    # Re-check output availability now that numpy/_sd are imported for TTS use
    try:
        _sd.query_devices(kind="output")
        _sd_output_available = True
    except Exception:
        _sd_output_available = False

    if EL_API_KEY:
        _el_client = _ElevenLabs(api_key=EL_API_KEY)
        _out_mode = (
            "sounddevice"
            if _sd_output_available
            else f"pipe→ {TTS_PLAYER_CMD.split()[0]}"
        )
        _sfx_mode = "pydub SFX" if _pydub_available else "no SFX (pip install pydub)"
        print(
            f"[tts] ElevenLabs ready  voice={EL_VOICE_ID}  model={EL_MODEL_ID}  output={_out_mode}  sfx={_sfx_mode}"
        )
    else:
        print("[tts] ELEVENLABS_API_KEY not set — TTS disabled.")
except ImportError as _tts_import_err:
    print(f"[tts] Optional deps missing ({_tts_import_err}) — TTS disabled.")
    print("      pip install elevenlabs sounddevice numpy")

# ---------------------------------------------------------------------------
# Pre-load SFX arrays once at startup (avoid disk I/O on every _speak call)
# ---------------------------------------------------------------------------
_sfx_prefix_f32: "_np.ndarray | None" = None  # type: ignore[name-defined]
_sfx_suffix_f32: "_np.ndarray | None" = None  # type: ignore[name-defined]
_sfx_crackle_f32: "_np.ndarray | None" = None  # type: ignore[name-defined]

def _preload_sfx() -> None:
    """Load SFX MP3 files → float32 numpy arrays. Called once after imports succeed."""
    global _sfx_prefix_f32, _sfx_suffix_f32, _sfx_crackle_f32
    if not _pydub_available:
        return
    _SR = 16000
    def _load(path: str, gain_db: float):
        try:
            seg = _AudioSegment.from_file(path)
            seg = seg.set_channels(1).set_frame_rate(_SR).set_sample_width(2)
            arr = _np.frombuffer(seg.raw_data, dtype=_np.int16).astype(_np.float32) / 32768.0
            return arr * (10.0 ** (gain_db / 20.0))
        except Exception as _e:
            print(f"[tts] SFX preload failed ({path}): {_e}")
            return None
    _sfx_prefix_f32  = _load(TTS_PREFIX_PATH, TTS_PREFIX_VOLUME_DB)
    _sfx_suffix_f32  = _load(TTS_SUFFIX_PATH, TTS_SUFFIX_VOLUME_DB)
    _sfx_crackle_f32 = _load(TTS_SFX_PATH,   TTS_SFX_VOLUME_DB)
    print("[tts] SFX arrays pre-loaded.")

if _pydub_available and _el_client is not None:
    _preload_sfx()


def _speak(text: str, pose: str = "calm") -> None:
    """
    Generate TTS audio via ElevenLabs (pcm_16000, streamed) and play it through
    sounddevice or a fallback pipe, feeding per-chunk RMS to the game UI portrait.

    When pydub is installed:
      - Prefix and suffix audio files are pre-loaded as numpy arrays and played
        immediately before / after the streamed TTS — with no gap.
      - A looping background crackle (SFX) is mixed into each TTS chunk in
        real-time. The SFX stops the instant the TTS stream ends; it never bleeds
        into the suffix or beyond.
    Without pydub the raw PCM stream is played directly (no SFX).
    """
    if _el_client is None:
        return
    import re as _re

    clean = _re.sub(r"[*_`#]+", "", text).strip()
    if not clean:
        return
    try:
        SAMPLE_RATE = 16000
        CHUNK = int(SAMPLE_RATE * 0.05)  # 800 samples = 50 ms blocks
        CHUNK_BYTES = CHUNK * 2  # 2 bytes per s16le sample

        portrait_started = False

        def _ensure_portrait_started() -> None:
            nonlocal portrait_started
            if not portrait_started:
                portrait_started = True
                try:
                    _post("/tts/start", {"duration": 30.0, "pose": pose})
                except Exception:
                    pass

        # Use pre-loaded SFX arrays (loaded once at startup to avoid disk I/O here)
        prefix_f32 = _sfx_prefix_f32
        suffix_f32 = _sfx_suffix_f32
        sfx_f32    = _sfx_crackle_f32
        sfx_offset: int = 0  # current read position in the looping SFX array

        def _mix_sfx(block: "_np.ndarray") -> "_np.ndarray":
            """Add one chunk of looping SFX to block (in-place safe). SFX stops here."""
            nonlocal sfx_offset
            if sfx_f32 is None or len(sfx_f32) == 0:
                return block
            n = len(block)
            sfx_chunk = _np.empty(n, dtype=_np.float32)
            pos = 0
            while pos < n:
                avail = len(sfx_f32) - sfx_offset
                take = min(avail, n - pos)
                sfx_chunk[pos : pos + take] = sfx_f32[sfx_offset : sfx_offset + take]
                pos += take
                sfx_offset = (sfx_offset + take) % len(sfx_f32)
            return _np.clip(block + sfx_chunk, -1.0, 1.0)

        def _f32_to_bytes(arr: "_np.ndarray") -> bytes:
            return (arr * 32767.0).astype(_np.int16).tobytes()

        # ------------------------------------------------------------------
        # Request streaming PCM from ElevenLabs (same for both paths)
        # ------------------------------------------------------------------
        raw_gen = _el_client.text_to_speech.convert(
            voice_id=EL_VOICE_ID,
            text=clean,
            model_id=EL_MODEL_ID,
            output_format="pcm_16000",
        )
        if isinstance(raw_gen, (bytes, bytearray)):
            raw_gen = iter([raw_gen])

        leftover = b""

        if _sd_output_available:
            # Feed raw_gen via a producer thread so the audio consumer can write
            # silence while waiting for the first network chunk — preventing the
            # stream buffer underrun that causes the audible click/interruption.
            _pcm_q: "_queue.Queue[bytes | None]" = _queue.Queue(maxsize=64)

            def _producer() -> None:
                try:
                    for api_chunk in raw_gen:
                        _pcm_q.put(api_chunk)
                finally:
                    _pcm_q.put(None)  # sentinel: stream is done

            prod_thread = threading.Thread(target=_producer, daemon=True, name="tts-producer")
            prod_thread.start()

            _silence = _np.zeros(CHUNK, dtype=_np.float32)

            with _sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=SD_OUTPUT_DEVICE,
                latency="high",
            ) as stream:

                # ── 1. Prefix ──────────────────────────────────────────────
                if prefix_f32 is not None and len(prefix_f32):
                    _ensure_portrait_started()
                    for i in range(0, len(prefix_f32), CHUNK):
                        blk = prefix_f32[i : i + CHUNK]
                        if len(blk) < CHUNK:
                            blk = _np.pad(blk, (0, CHUNK - len(blk)))
                        stream.write(blk)

                # ── 2. Streamed TTS + looping SFX overlay ──────────────────
                # Silence blocks are written while the producer thread waits for
                # the first network chunk, keeping the hardware buffer full.
                _tts_done = False
                while not _tts_done:
                    try:
                        api_chunk = _pcm_q.get(timeout=0.04)  # 40 ms silence granularity
                    except _queue.Empty:
                        stream.write(_silence)  # keep stream alive during network wait
                        continue

                    if api_chunk is None:
                        _tts_done = True
                        break

                    leftover += api_chunk
                    while len(leftover) >= CHUNK_BYTES:
                        block_bytes = leftover[:CHUNK_BYTES]
                        leftover = leftover[CHUNK_BYTES:]
                        block_i16 = _np.frombuffer(block_bytes, dtype=_np.int16)
                        block_f32 = block_i16.astype(_np.float32) / 32768.0
                        mixed = _mix_sfx(block_f32)
                        _ensure_portrait_started()
                        stream.write(mixed)
                        rms = float(_np.sqrt(_np.mean(mixed**2)))
                        amp_val = min(1.0, rms * EL_AMP_SCALE)
                        threading.Thread(
                            target=lambda v=amp_val: _post("/tts/amplitude", {"value": v}),
                            daemon=True,
                        ).start()

                # Flush final partial chunk
                if leftover:
                    block_i16 = _np.frombuffer(leftover, dtype=_np.int16)
                    block_f32 = block_i16.astype(_np.float32) / 32768.0
                    mixed = _mix_sfx(block_f32)
                    if len(mixed) < CHUNK:
                        mixed = _np.pad(mixed, (0, CHUNK - len(mixed)))
                    _ensure_portrait_started()
                    stream.write(mixed)

                prod_thread.join()

                # ── 3. Suffix (SFX is NOT mixed in here) ───────────────────
                if suffix_f32 is not None and len(suffix_f32):
                    for i in range(0, len(suffix_f32), CHUNK):
                        blk = suffix_f32[i : i + CHUNK]
                        if len(blk) < CHUNK:
                            blk = _np.pad(blk, (0, CHUNK - len(blk)))
                        stream.write(blk)

        else:
            # WSL / no-output-device fallback: pipe raw s16le PCM to ffplay.exe
            cmd = TTS_PLAYER_CMD.split()
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            try:
                # ── 1. Prefix ──────────────────────────────────────────────
                if prefix_f32 is not None and len(prefix_f32):
                    _ensure_portrait_started()
                    proc.stdin.write(_f32_to_bytes(prefix_f32))

                # ── 2. Streamed TTS + looping SFX overlay ──────────────────
                for api_chunk in raw_gen:
                    leftover += api_chunk
                    while len(leftover) >= CHUNK_BYTES:
                        block_bytes = leftover[:CHUNK_BYTES]
                        leftover = leftover[CHUNK_BYTES:]
                        block_i16 = _np.frombuffer(block_bytes, dtype=_np.int16)
                        block_f32 = block_i16.astype(_np.float32) / 32768.0
                        mixed = _mix_sfx(block_f32)
                        _ensure_portrait_started()
                        proc.stdin.write(_f32_to_bytes(mixed))
                        rms = float(_np.sqrt(_np.mean(mixed**2)))
                        amp_val = min(1.0, rms * EL_AMP_SCALE)
                        threading.Thread(
                            target=lambda v=amp_val: _post("/tts/amplitude", {"value": v}),
                            daemon=True,
                        ).start()

                # Flush final partial chunk
                if leftover:
                    block_i16 = _np.frombuffer(leftover, dtype=_np.int16)
                    block_f32 = block_i16.astype(_np.float32) / 32768.0
                    mixed = _mix_sfx(block_f32)
                    proc.stdin.write(_f32_to_bytes(mixed))

                # ── 3. Suffix (SFX is NOT mixed in here) ───────────────────
                if suffix_f32 is not None and len(suffix_f32):
                    proc.stdin.write(_f32_to_bytes(suffix_f32))

            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                proc.wait()

    except Exception as exc:
        print(f"[tts] ERROR: {exc}")
    finally:
        try:
            _post("/tts/stop", {})
        except Exception:
            pass


def _speak_async(text: str, pose: str = "calm") -> None:
    """
    Fire-and-forget TTS in a background daemon thread.
    Calls are serialised by _tts_lock so they never overlap.
    pose: "attack" or "calm" — forwarded to the game UI portrait.
    """
    if _el_client is None:
        return

    def _run() -> None:
        with _tts_lock:
            _speak(text, pose=pose)
        # STT trigger is written inside _speak's finally block;
        # _tts_lock ensures we don't overlap with another _speak call.

    threading.Thread(target=_run, daemon=True, name="tts").start()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _http(
    method: str, path: str, body: Optional[dict] = None, timeout: float = 5.0
) -> dict:
    """Minimal HTTP helper (no requests dependency)."""
    url = BASE_URL + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(
                resp.read().decode("utf-8", errors="replace"), strict=False
            )
    except urllib.error.HTTPError as e:
        # Server responded with an HTTP error code (e.g. 404 = unknown endpoint)
        body_txt = e.read().decode(errors="replace")
        raise RuntimeError(
            f"AgentBridge HTTP {e.code} on {method} {path}: {body_txt}"
        ) from e
    except urllib.error.URLError as e:
        # Network-level failure (connection refused, timeout, …)
        raise ConnectionError(f"AgentBridge unreachable ({url}): {e}") from e


def _get(path: str) -> dict:
    return _http("GET", path)


def _post(path: str, body: dict) -> dict:
    return _http("POST", path, body)


# ---------------------------------------------------------------------------
# Strands tools
# (Each function decorated with @tool becomes a callable tool for the LLM.
#  The docstring is shown to the model as the tool description.)
# ---------------------------------------------------------------------------


@tool
def get_game_state() -> str:
    """
    Returns full current game state as JSON, including:
    - allyTeams: list of teams with their units (id, defName, position,
      health, category, buildOptions for factories)
    - visibleEnemies: enemy units currently visible on the map
    - mapInfo: map name, dimensions, wind speed, tidal strength
    - gameFrame: current simulation frame (30 fps)

    Call this to understand the overall battlefield situation before
    planning any actions.
    """
    try:
        state = _get("/state")
        return json.dumps(state, indent=2)
    except ConnectionError as e:
        return f"ERROR: {e}"


@tool
def get_build_catalog() -> str:
    """
    Returns a catalog of all unit definitions grouped by category:
    - commanders, factories, constructors, extractors, generators,
      converters, turrets, other

    Each factory entry includes a 'buildOptions' list (unit defNames the
    factory can produce). Use this to know what a factory can build before
    sending a 'build' command.
    """
    try:
        return json.dumps(_get("/defs"), indent=2)
    except ConnectionError as e:
        return f"ERROR: {e}"


@tool
def get_new_chat_messages() -> str:
    """
    Drains and returns all chat messages that have arrived since the last
    call. Returns a JSON list of objects: [{text, frame}, ...].
    'text' is the raw console line (e.g. '[All] PlayerName: hello').
    'frame' is the game frame when it was received.

    Returns an empty list if no new messages are available.
    """
    try:
        msgs = _get("/chat")
        return json.dumps(msgs)
    except ConnectionError as e:
        return f"ERROR: {e}"


@tool
def send_chat_message(message: str, pose: str = "calm") -> str:
    """
    Sends a chat message visible to all players in-game and speaks it via TTS.
    Use this to acknowledge commands, give status updates, or communicate
    with teammates.

    Args:
        message: The text to broadcast in the in-game 'All' chat channel.
        pose:    Which commander portrait to show during TTS playback.
                 Use "attack" when delivering aggressive orders, warnings, or
                 battle reports (e.g. enemy spotted, launching attack).
                 Use "calm" (default) for routine status updates, confirmations,
                 or strategic advice.
    """
    try:
        # Split messages longer than 250 chars into multiple in-game sends
        results = []
        parts = [message[i : i + 250] for i in range(0, len(message), 250)]
        for part in parts:
            resp = _post("/chat/send", {"message": part})
            results.append(resp)
            print(f"[send_chat] sent: {part!r} → {resp}")
        # Speak the full message with the chosen portrait pose
        _speak_async(message, pose=pose)
        return json.dumps(results if len(results) > 1 else results[0])
    except Exception as e:
        print(f"\n[send_chat] ERROR ({type(e).__name__}): {e}")
        return f"ERROR: {e}"


@tool
def map_ping(x: float, z: float, label: str = "") -> str:
    """
    Places a visible map marker (ping) at world coordinates (x, z) with an
    optional text label. The marker appears on the minimap and on the main
    view for all players, like a player clicking on the map.

    Use this to:
    - Point out a threat or a location of interest to the player
    - Highlight where a new building will be constructed
    - Mark enemy positions that need attention
    - Confirm the location of a completed order

    Args:
        x:     World X coordinate (east-west axis).
        z:     World Z coordinate (north-south axis).
        label: Short text label shown next to the marker (max ~40 chars).
    """
    try:
        resp = _post("/ping", {"x": x, "z": z, "label": label})
        print(f"[map_ping] @ ({x},{z}) label={label!r} → {resp}")
        return json.dumps(resp)
    except (ConnectionError, RuntimeError) as e:
        print(f"[map_ping] ERROR: {e}")
        return f"ERROR: {e}"


@tool
def command_unit(
    unit_id: int,
    cmd: str,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    target_id: int = 0,
    unit_type: str = "",
    facing: int = 0,
    shift: bool = False,
) -> str:
    """
    Issues a single order to an allied unit. Supported commands:

    Movement & combat:
      move    – move unit to (x, y, z)
      attack  – attack target_id (unit), or attack ground at (x, y, z)
      patrol  – patrol to (x, y, z)
      fight   – fight-move to (x, y, z) (attack-move)
      stop    – cancel all orders
      selfd   – self-destruct

    Economy & support:
      reclaim – reclaim feature or unit by target_id
      repair  – repair allied unit by target_id
      guard   – guard (escort) allied unit by target_id

    Production (factories & constructors):
      build      – queue unit_type (defName string, e.g. 'armlab')
                   Provide x, z coordinates for placement (y is calculated
                   automatically from terrain height — do NOT pass y).
                   facing 0-3: 0=South 1=East 2=North 3=West.
      set_rally  – set factory rally point to (x, y, z)

    Args:
        unit_id:   The Spring unit ID to command.
        cmd:       One of the verb strings above.
        x, y, z:   World coordinates (used by move/patrol/fight/build/set_rally).
        target_id: Target unit/feature ID (used by attack/reclaim/repair/guard).
        unit_type: Unit defName to build (used by 'build' only).
        facing:    Build facing 0-3 (0=South, 1=East, 2=North, 3=West).
        shift:     If True, append to order queue instead of replacing.
    """
    payload: dict = {"unitID": unit_id, "cmd": cmd, "shift": shift}
    if cmd in ("move", "patrol", "fight", "set_rally"):
        payload.update({"x": x, "y": y, "z": z})
    elif cmd == "attack":
        if target_id:
            payload["targetID"] = target_id
        else:
            payload.update({"x": x, "y": y, "z": z})
    elif cmd in ("reclaim", "repair", "guard"):
        payload["targetID"] = target_id
    elif cmd == "build":
        # Do NOT include y — the relay gadget calculates terrain height automatically.
        # Passing y=0 causes Spring to silently ignore the build order.
        payload.update({"x": x, "z": z, "facing": facing})
        if unit_type:
            payload["unitType"] = unit_type
    print(f"[command_unit] → POST /command payload={payload}")
    try:
        resp = _post("/command", payload)
        print(f"[command_unit] ← response: {resp}")
        return json.dumps(resp)
    except ConnectionError as e:
        print(f"[command_unit] ERROR: {e}")
        return f"ERROR: {e}"


@tool
def command_units_batch(commands: list) -> str:
    """
    Issues multiple orders in sequence. Each element of 'commands' must be a
    dict with the same fields accepted by command_unit:
      {"unit_id": <int>, "cmd": "<verb>", ...extra fields...}

    This is more efficient than calling command_unit multiple times when you
    want to coordinate a group of units simultaneously.

    Args:
        commands: List of command dicts. Each must include 'unit_id' and 'cmd'.
    """
    results = []
    for c in commands:
        results.append(
            command_unit(
                unit_id=c.get("unit_id", 0),
                cmd=c.get("cmd", "stop"),
                x=c.get("x", 0.0),
                y=c.get("y", 0.0),
                z=c.get("z", 0.0),
                target_id=c.get("target_id", 0),
                unit_type=c.get("unit_type", ""),
                facing=c.get("facing", 0),
                shift=c.get("shift", False),
            )
        )
    return json.dumps(results)


@tool
def find_allied_units(
    category: str = "", name_filter: str = "", owner: str = "bot"
) -> str:
    """
    Returns a filtered list of allied units from the current game state.

    Args:
        category:    Filter by unit category. Supported values:
                     'commander'   – the commander unit (isCommander)
                     'factory'     – unit-producing buildings (isFactory)
                     'constructor' – mobile builder units, NOT factories or commanders
                                     (isBuilder AND canMove AND NOT isFactory AND NOT isCommander)
                     'builder'     – all builders incl. commander (isBuilder AND NOT isFactory)
                     'combat'      – mobile fighting units (canMove AND canAttack AND NOT isBuilder)
                     'defense'     – static weapons/turrets (NOT canMove AND canAttack)
                     'extractor'   – metal extractors (isExtractor)
                     'generator'   – energy producers: solar, wind, tidal (isGenerator)
                     'structure'   – other static non-combat buildings
                     ''            – return all units (no filter)
        name_filter: Filter by defName substring, e.g. 'arm'. Empty = all.
        owner:       'bot' (default) – AI teams only
                     'human'        – human player's units only
                     'all'          – all allied units

    IMPORTANT: Always use owner='bot' unless the player explicitly asks you
    to command their own units.

    Returns a JSON list: [{unitID, name, humanName, isCommander, isFactory,
    isBuilder, canMove, canAttack, isExtractor, isGenerator, x, y, z,
    health, maxHealth, buildOptions, teamID, isBot}, ...]
    """
    try:
        state = _get("/state")
    except ConnectionError as e:
        return f"ERROR: {e}"

    def _matches_category(unit: dict, cat: str) -> bool:
        is_cmd = unit.get("isCommander", False)
        is_fact = unit.get("isFactory", False)
        is_bld = unit.get("isBuilder", False)
        can_mv = unit.get("canMove", False)
        can_atk = unit.get("canAttack", False)
        is_ext = unit.get("isExtractor", False)
        is_gen = unit.get("isGenerator", False)
        if cat == "commander":
            return is_cmd
        if cat == "factory":
            return is_fact
        if cat == "constructor":
            return is_bld and can_mv and not is_fact and not is_cmd
        if cat == "builder":
            return is_bld and not is_fact
        if cat == "combat":
            return can_mv and can_atk and not is_bld
        if cat == "defense":
            return not can_mv and can_atk
        if cat == "extractor":
            return is_ext
        if cat == "generator":
            return is_gen
        if cat == "structure":
            return (
                not can_mv and not can_atk and not is_fact and not is_ext and not is_gen
            )
        return False  # unknown category

    result = []
    for team in state.get("teams", []):
        is_bot = team.get("isBot", False)
        is_my_team = team.get("isMyTeam", False)
        if owner == "bot" and not is_bot:
            continue
        if owner == "human" and not is_my_team:
            continue
        for unit in team.get("units", []):
            if category and not _matches_category(unit, category.lower()):
                continue
            if name_filter and name_filter.lower() not in unit.get("name", "").lower():
                continue
            result.append({**unit, "teamID": team["teamID"], "isBot": is_bot})
    return json.dumps(result, indent=2)


@tool
def get_enemy_intel() -> str:
    """
    Returns all enemy units currently visible to the local player (not hidden
    in fog of war). Includes radar blips (blip=true, no defName) and fully
    visible units with position and estimated health.

    Use this for target acquisition, threat assessment, or planning strikes.
    """
    try:
        state = _get("/state")
        enemies = state.get("visibleEnemies", [])
        return json.dumps(enemies, indent=2)
    except ConnectionError as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Unit reservation & event-watch tools
# ---------------------------------------------------------------------------


@tool
def reserve_units(unit_ids: list) -> str:
    """
    Reserve one or more units so the native BAR AI cannot override their orders
    while the LLM agent is executing a multi-step task.

    Reserved units will only accept orders issued by this agent until you call
    unreserve_units().

    Args:
        unit_ids: List of integer Spring unit IDs to reserve.
    """
    try:
        resp = _post("/reserve", {"unitIDs": unit_ids})
        print(f"[reserve] {unit_ids} → {resp}")
        return json.dumps(resp)
    except (ConnectionError, RuntimeError) as e:
        return f"ERROR: {e}"


@tool
def unreserve_units(unit_ids: list) -> str:
    """
    Release the reservation on one or more units, allowing the native AI to
    issue orders to them again.

    Always call this when a multi-step task is complete or has failed.

    Args:
        unit_ids: List of integer Spring unit IDs to unreserve.
    """
    try:
        resp = _post("/unreserve", {"unitIDs": unit_ids})
        print(f"[unreserve] {unit_ids} → {resp}")
        return json.dumps(resp)
    except (ConnectionError, RuntimeError) as e:
        return f"ERROR: {e}"


@tool
def watch_unit(
    unit_id: int, event_type: str, task_id: str, description: str = ""
) -> str:
    """
    Register an event listener on a unit.  When the specified event fires,
    the agent loop will automatically wake this task and call the LLM again
    with the event details so it can proceed to the next step.

    CRITICAL — which event to use for each situation:

      "idle"         – BUILDER / COMMANDER finished building something and is now
                       standing still.  Use this on the builder unit to detect
                       when a construction order is complete.
      "finished"     – A NEWLY BUILT unit or structure just appeared on the map.
                       Use this on the STRUCTURE/UNIT that was just built, NOT
                       on the builder.  You usually won't know the new unit's ID
                       in advance, so prefer "idle" on the builder instead.
      "from_factory" – FACTORY just produced a new unit (newUnitID in the event).
                       Use this on the FACTORY unit, not on the produced unit.
      "destroyed"    – Unit was killed.
      "any"          – Fire on any of the above events.

    Typical usage patterns:
      • Wait for commander to finish building a structure:
            watch_unit(commander_id, "idle", task_id)
      • Wait for factory to produce a new combat unit:
            watch_unit(factory_id, "from_factory", task_id)

    Args:
        unit_id:     Spring unit ID to watch.
        event_type:  One of the event types above.
        task_id:     Unique string identifier for the current multi-step task.
                     Use a short descriptive slug, e.g. "build_mex_chain".
        description: Optional human-readable description of what this task is doing.
    """
    # Register locally so the polling loop can formulate a good continuation prompt
    if task_id not in activeTasks:
        activeTasks[task_id] = {
            "description": description or task_id,
            "reserved_units": [],
            "steps_done": [],
        }
    try:
        resp = _post(
            "/watch", {"unitID": unit_id, "event": event_type, "taskID": task_id}
        )
        print(f"[watch] unitID={unit_id} event={event_type} taskID={task_id} → {resp}")
        return json.dumps(resp)
    except (ConnectionError, RuntimeError) as e:
        return f"ERROR: {e}"


@tool
def unwatch_unit(unit_id: int, task_id: str = "") -> str:
    """
    Stop watching a unit for events (cancels a previously registered watch).
    Optionally, if task_id is provided and all units for that task have been
    unwatched, also removes the task from the registry.

    Args:
        unit_id: Spring unit ID to stop watching.
        task_id: Task ID to clean up from the registry (optional).
    """
    if task_id and task_id in activeTasks:
        del activeTasks[task_id]
        print(f"[unwatch] task {task_id!r} removed from registry")
    try:
        resp = _post("/watch", {"unitID": unit_id, "unwatch": True})
        print(f"[unwatch] unitID={unit_id} → {resp}")
        return json.dumps(resp)
    except (ConnectionError, RuntimeError) as e:
        return f"ERROR: {e}"


@tool
def reserve_and_build(
    unit_id: int,
    build_type: str,
    x: float,
    z: float,
    task_id: str,
    facing: int = 0,
    description: str = "",
) -> str:
    """
    Atomically: reserve a unit, issue a build order, then watch for idle.
    Use this INSTEAD of calling reserve_units + command_unit + watch_unit
    separately when you want a constructor/commander to build something and
    be notified when it finishes.

    IMPORTANT: build_type MUST be a defName that appears in the unit's
    'buildOptions' list (returned by get_game_state() or find_allied_units()).
    Always verify with get_game_state() first. If the unit cannot build the
    requested structure, this call will return an error listing valid names.

    This single call guarantees the reservation is in place before the build
    order reaches the game, preventing the native AI from overriding it.

    Args:
        unit_id:     Spring unit ID of the builder (commander or constructor).
        build_type:  DefName of the structure to build — MUST be in the unit's
                     buildOptions (e.g. 'corlab', not 'coralab').
        x, z:        World coordinates for placement (y calculated automatically).
        task_id:     Unique slug for this task (e.g. 'build_bot_lab').
        facing:      Build facing 0-3 (0=South 1=East 2=North 3=West).
        description: Short description for the task registry.
    """
    # ── Validate build_type against the unit's actual build options ──────────
    try:
        state = _get("/state")
        builder_unit = None
        for team in state.get("teams", []):
            for unit in team.get("units", []):
                if unit.get("unitID") == unit_id:
                    builder_unit = unit
                    break
            if builder_unit:
                break
        if builder_unit:
            build_opts = builder_unit.get("buildOptions") or []
            valid_names = [o["name"] for o in build_opts]
            if valid_names and build_type not in valid_names:
                return (
                    f"ERROR: unit {unit_id} cannot build '{build_type}'. "
                    f"Its build options are: {valid_names}. "
                    f"Use one of those names instead."
                )
        else:
            print(
                f"[reserve_and_build] WARNING: unit {unit_id} not found in state, skipping validation"
            )
    except Exception as e:
        print(
            f"[reserve_and_build] WARNING: validation failed ({e}), proceeding anyway"
        )
    # ──────────────────────────────────────────────────────────────────────────
    steps = []
    try:
        r = _post("/reserve", {"unitIDs": [unit_id]})
        steps.append(f"reserve → {r}")
        print(f"[reserve_and_build] reserved {unit_id}")
    except (ConnectionError, RuntimeError) as e:
        return f"ERROR reserving: {e}"
    try:
        payload = {
            "unitID": unit_id,
            "cmd": "build",
            "shift": False,
            "x": x,
            "z": z,
            "facing": facing,
            "unitType": build_type,
        }
        r = _post("/command", payload)
        steps.append(f"build → {r}")
        print(f"[reserve_and_build] build order sent: {payload}")
    except (ConnectionError, RuntimeError) as e:
        _post("/unreserve", {"unitIDs": [unit_id]})
        return f"ERROR building: {e}"
    try:
        if task_id not in activeTasks:
            activeTasks[task_id] = {
                "description": description or task_id,
                "reserved_units": [unit_id],
                "steps_done": [],
            }
        r = _post("/watch", {"unitID": unit_id, "event": "idle", "taskID": task_id})
        steps.append(f"watch → {r}")
        print(f"[reserve_and_build] watching {unit_id} idle for task {task_id!r}")
    except (ConnectionError, RuntimeError) as e:
        return f"WARNING: build ordered but watch failed: {e}"
    return json.dumps({"status": "ok", "steps": steps})


@tool
def get_build_queue(unit_ids: list) -> str:
    """
    Returns the current command queue for one or more units or factories.

    Each entry is one of:
    - Build order:    {"type": "build",   "defName": "<name>", "x": <n>, "z": <n>, "tag": <n>}
    - Other command:  {"type": "command", "id": <n>, "params": [...],    "tag": <n>}

    An empty "queue" list means the unit is idle (no pending orders).

    Use this to:
    - Confirm a factory or constructor is still executing a build order you issued.
    - Count how many items remain in a factory's production backlog.
    - Check whether a builder has an empty queue before assigning it a new task.
    - Diagnose why a build task continuation has not fired yet (unit may have
      been given other orders or the build may have already finished).

    Args:
        unit_ids: List of integer Spring unit IDs (factories, constructors, etc.)
    """
    try:
        result = _post("/buildqueue", {"unitIDs": unit_ids})
        print(f"[buildqueue] {result}")
        return json.dumps(result)
    except (ConnectionError, RuntimeError) as e:
        print(f"[buildqueue] ERROR: {e}")
        return f"ERROR: {e}"


@tool
def gift_units(unit_ids: list) -> str:
    """
    Transfers one or more bot-owned units or structures to the human player.
    Use this to hand over units you have built or controlled — the player
    will receive direct control of them immediately.

    Typical uses:
    - Give a freshly built constructor or combat unit to the player so they
      can micro it themselves.
    - Hand over a building (factory, defense turret, etc.) to let the player
      manage its production queue.
    - Reward the player with units after completing an objective.

    IMPORTANT: only gift bot-owned units (isBot=true). Gifting units that
    already belong to the human player has no effect and wastes a tool call.
    Always unreserve the units BEFORE gifting them so the transfer succeeds.

    Args:
        unit_ids: List of integer Spring unit IDs to transfer to the player.
    """
    try:
        resp = _post("/gift", {"unitIDs": unit_ids})
        print(f"[gift] {unit_ids} → {resp}")
        return json.dumps(resp)
    except (ConnectionError, RuntimeError) as e:
        print(f"[gift] ERROR: {e}")
        return f"ERROR: {e}"


@tool
def get_pending_events() -> str:
    """
    Poll and drain all pending unit events (idle, finished, destroyed,
    from_factory) that have been queued by the relay gadget since the last call.

    Returns a JSON list of event objects:
      {"type": "idle"|"finished"|"destroyed"|"from_factory",
       "unitID": <n>, "taskID": "<str>", "frame": <n>,
       "newUnitID": <n>   (only for from_factory events)}

    Returns an empty list if no events are pending.
    """
    try:
        evts = _get("/events")
        return json.dumps(evts)
    except ConnectionError as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Active-task registry  (task_id → context for event-driven continuation)
# ---------------------------------------------------------------------------
activeTasks: dict = {}

# ---------------------------------------------------------------------------
# Module-level work queue  (shared by chat-loop thread AND STT callback)
# ---------------------------------------------------------------------------
_work_queue: _queue.PriorityQueue = _queue.PriorityQueue()
_seq_counter: int = 0
_seq_lock: threading.Lock = threading.Lock()
_queued_tasks: set = set()
_queued_lock: threading.Lock = threading.Lock()


def _next_seq() -> int:
    global _seq_counter
    with _seq_lock:
        _seq_counter += 1
        return _seq_counter


def _with_pings(msg: str) -> str:
    """
    Append any unread player map-pings to msg.
    Example: "Build a turret here ! [Player pinged (1024, 512)]"
    """
    try:
        pings = _get("/pings")
        if pings:
            parts = [f"({p['x']}, {p['z']})" for p in pings]
            noun = "pinged" if len(parts) == 1 else "pinged"
            suffix = " [Player " + noun + " " + ", ".join(parts) + "]"
            print(f"[ping] attaching {len(parts)} ping(s) to message")
            return msg + suffix
    except Exception:
        pass
    return msg


def _enqueue(inp: str, task_id: "str | None", priority: int) -> None:
    """Thread-safe enqueue for agent work items."""
    if task_id:
        with _queued_lock:
            if task_id in _queued_tasks:
                print(f"[skip] task {task_id!r} already queued")
                return
            _queued_tasks.add(task_id)
    _work_queue.put((priority, _next_seq(), inp, task_id))
    print(
        f"[queue] enqueued priority={priority} task={task_id!r} len={_work_queue.qsize()}"
    )


# ---------------------------------------------------------------------------
# STT coroutines  (Voxtral real-time transcription, merged from SpeechToTextDaemon)
# ---------------------------------------------------------------------------
_stt_audio_q: asyncio.Queue  # filled by audio_reader, consumed by gated_audio_stream
_stt_start: asyncio.Event  # key pressed
_stt_stop: asyncio.Event  # key released
_stt_cancel: asyncio.Event  # key cancelled


async def _stt_mic_stream() -> AsyncIterator[bytes]:
    """Capture PCM s16le from the default mic via sounddevice."""
    loop = asyncio.get_event_loop()
    q: asyncio.Queue[bytes] = asyncio.Queue()

    def _cb(indata, frames, time_info, status):
        loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

    with _sd.InputStream(
        samplerate=STT_SAMPLE_RATE,
        channels=STT_CHANNELS,
        dtype="int16",
        blocksize=STT_BLOCK_SIZE,
        callback=_cb,
    ):
        print("[stt] Microphone open via sounddevice.")
        while True:
            yield await q.get()


async def _stt_stdin_stream() -> AsyncIterator[bytes]:
    """Read raw PCM s16le chunks from stdin (piped from ffmpeg)."""
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)
    print("[stt] Reading audio from stdin (ffmpeg pipe).")
    while True:
        chunk = await reader.read(STT_CHUNK_SIZE)
        if not chunk:
            break
        yield chunk


async def _stt_audio_reader() -> None:
    """Background task: pump mic/stdin into _stt_audio_q."""
    source = _stt_mic_stream() if _sd_available else _stt_stdin_stream()
    print("[stt] Audio reader started.")
    async for chunk in source:
        await _stt_audio_q.put(chunk)
    await _stt_audio_q.put(None)
    print("[stt] Audio reader finished (EOF).")


async def _stt_gated_stream() -> AsyncIterator[bytes]:
    """Yield audio only while the push-to-talk key is held."""
    await _stt_start.wait()
    _stt_start.clear()

    # Discard stale chunks
    drained = 0
    while not _stt_audio_q.empty():
        try:
            _stt_audio_q.get_nowait()
            drained += 1
        except asyncio.QueueEmpty:
            break
    if drained:
        print(f"[stt] Drained {drained} stale audio chunk(s).")

    print("[stt] Recording — streaming to Voxtral.")
    while True:
        try:
            chunk = await asyncio.wait_for(_stt_audio_q.get(), timeout=0.5)
        except asyncio.TimeoutError:
            if _stt_cancel.is_set():
                _stt_cancel.clear()
                return
            continue
        if chunk is None:
            return
        if _stt_stop.is_set():
            _stt_stop.clear()
            yield chunk
            while not _stt_audio_q.empty():
                try:
                    _stt_audio_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
            return
        if _stt_cancel.is_set():
            _stt_cancel.clear()
            while not _stt_audio_q.empty():
                try:
                    _stt_audio_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
            print("[stt] Cancelled.")
            return
        yield chunk


async def _stt_session_loop() -> None:
    """
    Maintain a warm Voxtral WebSocket. When key is pressed, stream audio;
    on completion, enqueue the transcription directly into the agent queue
    (bypassing game chat entirely).
    """
    try:
        from mistralai import Mistral as _Mistral
        from mistralai.extra.realtime import UnknownRealtimeEvent as _URE
        from mistralai.models import (
            AudioFormat as _AF,
            RealtimeTranscriptionError as _RTE,
            RealtimeTranscriptionSessionCreated as _RTSC,
            TranscriptionStreamDone as _TSD,
            TranscriptionStreamTextDelta as _TSTD,
        )
    except ImportError:
        print("[stt] mistralai not installed — STT disabled. pip install mistralai")
        return

    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        print("[stt] MISTRAL_API_KEY not set — STT disabled.")
        return

    _mc = _Mistral(api_key=api_key)
    _af = _AF(encoding="pcm_s16le", sample_rate=STT_SAMPLE_RATE)
    print("[stt] Session loop started — pre-connecting WebSocket.")

    while True:
        text_parts: list[str] = []
        aborted = False
        try:
            async for event in _mc.audio.realtime.transcribe_stream(
                audio_stream=_stt_gated_stream(),
                model=STT_MODEL,
                audio_format=_af,
            ):
                if isinstance(event, _RTSC):
                    print("[stt] WebSocket warm — waiting for key press.")
                elif isinstance(event, _TSTD):
                    text_parts.append(event.text)
                    print(event.text, end="", flush=True)
                elif isinstance(event, _TSD):
                    print("\n[stt] Transcription complete.")
                elif isinstance(event, _RTE):
                    print(f"[stt] Transcription error: {event}")
                    aborted = True
        except Exception as exc:
            print(f"[stt] WebSocket error: {exc}")
            aborted = True
            await asyncio.sleep(1.0)

        if not aborted:
            text = "".join(text_parts).strip()
            if text:
                # Write result for Lua widget HUD display (no chat send needed)
                STT_RESULT_FILE.write_text(text, encoding="utf-8")
                enriched = _with_pings(text)
                print(f"[stt] Routing to agent: {enriched!r}")
                # Directly enqueue — no game-chat round-trip
                _enqueue(f"[Voice] {enriched}", task_id=None, priority=0)
            else:
                print("[stt] Empty transcription.")
            # Write done flag so Lua widget resets its UI to idle
            STT_DONE_FILE.write_text("done", encoding="utf-8")

        await asyncio.sleep(0.1)


async def _stt_trigger_loop() -> None:
    """Poll IPC flag files at 50 ms to detect key press/release."""
    print(f"[stt] Trigger loop started (IPC dir: {_IPC_DIR})")
    while True:
        await asyncio.sleep(0.05)
        if STT_CANCEL_FILE.exists():
            STT_CANCEL_FILE.unlink(missing_ok=True)
            _stt_cancel.set()
        if STT_STOP_FILE.exists():
            STT_STOP_FILE.unlink(missing_ok=True)
            _stt_stop.set()
            print("[stt] Stop flag — key released.")
        if STT_TRIGGER_FILE.exists():
            STT_TRIGGER_FILE.unlink(missing_ok=True)
            STT_DONE_FILE.unlink(missing_ok=True)
            STT_RESULT_FILE.unlink(missing_ok=True)
            _stt_start.set()
            print("[stt] Trigger flag — key pressed.")


def _run_stt_loop() -> None:
    """Entry point for the STT daemon thread."""
    global _stt_audio_q, _stt_start, _stt_stop, _stt_cancel

    # Clean stale flag files from previous runs
    for f in [
        STT_TRIGGER_FILE,
        STT_STOP_FILE,
        STT_CANCEL_FILE,
        STT_DONE_FILE,
        STT_RESULT_FILE,
    ]:
        f.unlink(missing_ok=True)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _stt_audio_q = asyncio.Queue()
    _stt_start = asyncio.Event()
    _stt_stop = asyncio.Event()
    _stt_cancel = asyncio.Event()

    print(f"[stt] STT thread started. IPC dir: {_IPC_DIR}")
    print(
        f"[stt] Mic mode: {'sounddevice' if _sd_available else 'stdin (ffmpeg pipe)'}"
    )
    try:
        loop.run_until_complete(
            asyncio.gather(
                _stt_audio_reader(),
                _stt_session_loop(),
                _stt_trigger_loop(),
            )
        )
    except Exception as exc:
        print(f"[stt] Fatal error in STT loop: {exc}")
    finally:
        loop.close()


last_exec = None


def time_since_last_execution():
    global last_exec
    now = time.time()
    if last_exec is None:
        last_exec = now
        return None
    elapsed = now - last_exec
    last_exec = now
    # format elapsed as human-readable string
    return str(datetime.timedelta(seconds=elapsed))


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an AI co-commander for the real-time strategy game "Beyond All Reason"
(BAR), a Spring-engine game inspired by Total Annihilation.

Game basics:
- Two resources: Metal (M) and Energy (E). Extractors mine Metal from deposits;
  solar/wind/tidal generators produce Energy; converters trade E→M or M→E.
- Units are grouped by faction (Armada = arm_, Cortex = cor_).
- Commanders (com) are powerful, slow-moving units that can build and must not die.
- Factories (lab) produce combat and construction units.
- Constructors (con) can build structures anywhere on the map.
- Each game frame = 1/30 second. A frame value of 9000 ≈ 5 minutes in.

Your role:
- You watch in-game chat and respond when a player addresses you with
  "@agent" or starts a message with "!".
- You can read the game state, issue unit orders, queue factory production,
  send chat replies, and coordinate multi-unit actions.
- Always call get_game_state before taking any action so you have up-to-date info.
- ALWAYS use send_chat_message() to communicate with the player. It is the ONLY
  way your messages appear in-game. Your text response is NOT shown to the player.
- Use map_ping(x, z, label) to highlight important locations on the map — threats,
  build sites, completed structures, enemy positions, etc.
- Be concise (1-2 sentences max per send_chat_message call).
- NEVER use markdown formatting (no **bold**, no `backticks`, no # headings) —
  messages appear as plain text in-game.
- Never attack or reclaim allied units.
- When building units in a factory, use the exact defName from get_build_catalog.
- When the player says "build X", ALWAYS issue the build order, even if X already
  exists on the map. The player wants ANOTHER one built unless they say otherwise.
- Always obey the player's commands as literally as possible. If they say to attack an allied unit or something that seems suboptimal,
  just do it and don't question it. The player is the commander, you are the co-commander.,
- Address the player like a military subordinate, e.g. "Yes, Commander. Building additional metal extractor at (x, z)."
- When you start thinking, the player will have been waiting alread for a couple of seconds, 
  so respond as quickly as possible before starting your actions, then do the rest of your
  thinking and planning while the first message is in transit.  You can send multiple messages
  in a row if needed to acknowledge the command, then give updates as you execute it.
- I repeat, ALWAYS use send_chat_message() to talk to the player. Do NOT expect any text response you 
  generate to be seen by the player, those are only for your thinking out loud.

CRITICAL — which units to command:
- The game state contains teams with two flags: isBot (AI team) and isMyTeam (human player).
- You are the AI co-commander. ONLY command units belonging to BOT teams (isBot=true).
- NEVER command units belonging to the human player (isMyTeam=true) unless the player
  EXPLICITLY says something like "move my commander" or "use my units".
- Always use find_allied_units(owner='bot') to find units to command.
- When the player says "build X" or "attack Y", they mean: use the AI/bot units to do it.

Example interactions:
  Player: "@agent build a bot lab"
  → find_allied_units(category='commander', owner='bot') to get the AI commander,
    reserve_and_build(commander_id, 'coralab', x, z, 'build_bot_lab', description='build bot lab'),
    send_chat_message('Building bot lab, will notify when done.').
    When continuation fires (commander idle): unreserve_units + unwatch_unit, confirm.

  Player: "! we need more metal"
  → find_allied_units(category='constructor', owner='bot'), queue metal extractor builds.

  Player: "@agent attack the east with everything we have"
  → get enemy positions to the east, issue fight-move to BOT combat units.

Multi-step tasks with unit reservation:
- Use reserve_and_build() to atomically reserve + order + watch a builder in one call.
  This prevents the native AI from overriding the order between separate API calls.
- For non-build orders, call reserve_units() FIRST, then command_unit(), then watch_unit()
  in strict sequence (do not call them in parallel).
- The agent loop will automatically wake you when the event fires and call the LLM again
  with a continuation prompt containing the original request + event details.
- Always call unreserve_units() + unwatch_unit() when the task is complete or fails.
- Use a descriptive task_id slug (e.g. "build_bot_lab", "produce_5_tanks"). Keep it
  short and unique per concurrent task.
- Do NOT re-explain the task in the continuation — just execute the next step.
- When a TASK CONTINUATION fires with an "idle" event, ALWAYS call get_build_queue([unit_id])
  first to confirm the queue is empty and the build is truly done before calling
  unreserve_units() or unwatch_unit(). A non-empty queue means the unit is still working.
- You can gift bot units to the player with gift_units(). Do so when the player asks for
  units, when a build task is complete and the player should receive the result, or as a
  tactical gesture. Always unreserve_units() before calling gift_units().
"""

BAR_TOOLS = [
    get_game_state,
    get_build_catalog,
    get_new_chat_messages,
    send_chat_message,
    command_unit,
    command_units_batch,
    find_allied_units,
    get_enemy_intel,
    reserve_units,
    unreserve_units,
    watch_unit,
    unwatch_unit,
    get_pending_events,
    reserve_and_build,
    map_ping,
    gift_units,
    get_build_queue,
]


def build_agent() -> Agent:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        sys.exit("Set the MISTRAL_API_KEY environment variable before running.")

    model = MistralModel(
        model_id=MODEL_ID,
        api_key=api_key,
    )

    def _cb(**kwargs):
        """Log every tool invocation so we can see what the LLM decides to call."""
        if "current_tool_use" in kwargs:
            tu = kwargs["current_tool_use"]
            name = tu.get("name", "?")
            inp = tu.get("input", {})
            # Print on its own line even if a streaming chunk left no newline
            print(f"\n[tool_call] {name}({inp})")
        # Also stream text chunks to stdout so we can follow reasoning
        elif "data" in kwargs and isinstance(kwargs["data"], str):
            print(kwargs["data"], end="", flush=True)

    return Agent(
        model=model,
        tools=BAR_TOOLS,
        system_prompt=SYSTEM_PROMPT
        + f"\n [LAST EXECUTION] \n {time_since_last_execution()} ago.",
        callback_handler=_cb,
    )


# ---------------------------------------------------------------------------
# Chat polling loop
# ---------------------------------------------------------------------------


def _should_route(text: str) -> bool:
    """Return True if this chat line should be forwarded to the agent."""
    # Spring player chat format: "<PlayerName> message text"
    import re

    m = re.match(r"^<([^>]+)>\s+(.*)", text)
    if m:
        msg = m.group(2).strip()
        result = msg.lower().startswith(AGENT_PREFIX.lower()) or msg.startswith(
            AGENT_PREFIX2
        )
        if result:
            print(f"[route] MATCH — player='{m.group(1)}' msg='{msg}'")
        return result
    return False


def _strip_prefix(text: str) -> str:
    """Return 'PlayerName: command' with the trigger prefix removed."""
    import re

    m = re.match(r"^<([^>]+)>\s+(.*)", text)
    if m:
        sender = m.group(1)
        msg = m.group(2).strip()
        for prefix in (AGENT_PREFIX, AGENT_PREFIX2):
            if msg.lower().startswith(prefix.lower()):
                msg = msg[len(prefix) :].strip()
                break
        return f"[{sender}] {msg}"
    return text


def run_chat_loop(agent: Agent) -> None:
    """
    Main loop: poll /chat every POLL_INTERVAL seconds.
    Also polls /events and re-invokes the LLM for any active watched tasks.

    A single worker thread consumes the module-level _work_queue so Strands is
    never called concurrently.  Priority 0 = chat/voice input, 1 = task continuation.
    """
    print(
        f"AgentBridge chat loop started (polling {BASE_URL}/chat every {POLL_INTERVAL}s)"
    )
    print(f"Trigger prefixes: '{AGENT_PREFIX}'  '{AGENT_PREFIX2}'")
    print("Press Ctrl-C to stop.\n")

    def _worker() -> None:
        while True:
            priority, seq, inp, task_id = _work_queue.get()
            try:
                print(
                    f"[agent] calling LLM (priority={priority} task={task_id!r}): {inp[:120]!r}"
                )
                last_exc = None
                for attempt in range(2):
                    try:
                        response = agent(inp)
                        break
                    except TypeError as e:
                        if "concatenate str" in str(e) and attempt == 0:
                            print(
                                f"[agent] Strands streaming glitch, retrying... ({e})"
                            )
                            last_exc = e
                            time.sleep(1.0)
                            continue
                        raise
                else:
                    raise last_exc
                reply = str(response).strip()
                print(f"[agent] response ({len(reply)} chars): {reply[:300]!r}")
                # The LLM communicates via send_chat_message() tool — no auto-post here.
            except Exception as exc:
                print(f"[agent error] {type(exc).__name__}: {exc}")
                try:
                    _post("/chat/send", {"message": f"[AI] Error: {exc}"})
                except Exception as e2:
                    print(f"[agent error] also failed to send error msg: {e2}")
            finally:
                if task_id:
                    with _queued_lock:
                        _queued_tasks.discard(task_id)
                _work_queue.task_done()  # type: ignore[attr-defined]

    # Start the single worker thread
    threading.Thread(target=_worker, daemon=True, name="agent-worker").start()

    while True:
        # ── 1. Process incoming chat ────────────────────────────────────────
        try:
            messages = _get("/chat")
        except ConnectionError as e:
            print(f"[warn] connection lost — {e} — retrying in 5s")
            time.sleep(5.0)
            continue
        except RuntimeError as e:
            print(f"[error] {e}")
            print(
                "  → Reload the widget in-game: F11 → AgentBridge → disable then enable."
            )
            time.sleep(10.0)
            continue

        for entry in messages:
            text = entry.get("text", "")
            # Skip internal debug lines that leak through the console capture
            if text.startswith("[AgentBridge"):
                continue
            if _should_route(text):
                user_input = _with_pings(_strip_prefix(text))
                print(f"[→ agent] routing: {user_input!r}")
                _enqueue(user_input, task_id=None, priority=0)
            else:
                print(f"[poll] {text!r}")

        # ── 2. Process pending unit events ──────────────────────────────────
        try:
            events = _get("/events")
        except (ConnectionError, RuntimeError):
            events = []

        if events:
            # Group events by task_id
            by_task: dict[str, list] = {}
            for evt in events:
                tid = evt.get("taskID", "unknown")
                by_task.setdefault(tid, []).append(evt)

            for task_id, task_events in by_task.items():
                print(f"[events] task={task_id!r} events={task_events}")
                # Build a self-contained continuation prompt
                task_desc = activeTasks.get(task_id, {}).get("description", task_id)
                evt_lines = []
                for e in task_events:
                    extra = ""
                    if e.get("newUnitID"):
                        extra = f", newUnitID={e['newUnitID']}"
                    evt_lines.append(
                        f"  • type={e['type']} unitID={e.get('unitID')} "
                        f"frame={e.get('frame')}{extra}"
                    )
                continuation = (
                    f"TASK CONTINUATION — task_id: {task_id!r} ({task_desc})\n"
                    f"The following unit event(s) just fired:\n"
                    + "\n".join(evt_lines)
                    + "\n\nCheck the current game state and proceed to the next step of "
                    "this task. When the task is fully complete, call unreserve_units() "
                    "and unwatch_unit() for each unit to release them."
                )
                _enqueue(continuation, task_id=task_id, priority=1)

        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    agent = build_agent()

    # Start the STT daemon thread (voice → Voxtral → agent queue, no chat hop)
    stt_thread = threading.Thread(target=_run_stt_loop, daemon=True, name="stt-daemon")
    stt_thread.start()
    print("[main] STT daemon thread started.")

    try:
        run_chat_loop(agent)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
