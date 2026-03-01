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
    TTS_SFX_PATH     – background radio crackle mp3 (default: sounds/voice-soundeffects/...)
    TTS_PREFIX_PATH  – comm-open prefix mp3
    TTS_SUFFIX_PATH  – transmission-end suffix mp3
    TTS_SFX_VOLUME_DB    – crackle volume offset in dB (default: 10)
    TTS_PREFIX_VOLUME_DB – prefix volume offset in dB (default: 0)
    TTS_SUFFIX_VOLUME_DB – suffix volume offset in dB (default: 0)
    TTS_PLAYER_CMD_MP3 – ffplay command for mp3 piping (default: ffplay.exe -f mp3 ...)

Requirements
------------
    pip install 'strands-agents[mistral]' strands-agents-tools
    pip install elevenlabs sounddevice numpy   # optional, for TTS
    pip install pydub                          # optional, for radio SFX mixing
"""

import asyncio
import io
import json
import os
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
_SCRIPT_DIR  = Path(__file__).parent
_IPC_DIR     = Path(os.environ.get("STT_IPC_DIR",
                    str(_SCRIPT_DIR.parent.parent.parent / "LuaUI")))
_IPC_DIR.mkdir(parents=True, exist_ok=True)
STT_TRIGGER_FILE = _IPC_DIR / "stt_trigger.flag"
STT_STOP_FILE    = _IPC_DIR / "stt_stop.flag"
STT_CANCEL_FILE  = _IPC_DIR / "stt_cancel.flag"
STT_DONE_FILE    = _IPC_DIR / "stt_done.flag"
STT_RESULT_FILE  = _IPC_DIR / "stt_result.txt"  # written for Lua HUD display

STT_SAMPLE_RATE = 16000
STT_CHANNELS    = 1
STT_BLOCK_SIZE  = 4096
STT_CHUNK_SIZE  = STT_BLOCK_SIZE * 2  # bytes (int16)
STT_MODEL       = "voxtral-mini-transcribe-realtime-2602"

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
EL_VOICE_ID = os.environ.get("EL_VOICE_ID", "pNInz6obpgDQGcFmaJgB")  # Adam
EL_MODEL_ID = os.environ.get("EL_MODEL_ID", "eleven_turbo_v2_5")
EL_AMP_SCALE = float(os.environ.get("EL_AMP_SCALE", "5.0"))  # RMS → 0–1
# Fallback TTS player when sounddevice output is unavailable (e.g. WSL).
# Raw PCM s16le 16 kHz mono is written to its stdin.
TTS_PLAYER_CMD = os.environ.get(
    "TTS_PLAYER_CMD",
    "ffplay.exe -f s16le -ar 16000 -nodisp -autoexit -",
)
# Fallback TTS player when using mp3 output (pydub mixing path).
TTS_PLAYER_CMD_MP3 = os.environ.get(
    "TTS_PLAYER_CMD_MP3",
    "ffplay.exe -f mp3 -nodisp -autoexit -",
)
# TTS sound-effect mixing via pydub (radio / comm effect).
# Paths are resolved relative to the BAR.sdd root.
_BAR_ROOT = _SCRIPT_DIR.parent
TTS_SFX_PATH    = os.environ.get("TTS_SFX_PATH",
    str(_BAR_ROOT / "sounds" / "voice-soundeffects" / "AMBSci-Soft,_crackling_old_-Elevenlabs.mp3"))
TTS_PREFIX_PATH = os.environ.get("TTS_PREFIX_PATH",
    str(_BAR_ROOT / "sounds" / "voice-soundeffects" / "prefix_communication_sound.mp3"))
TTS_SUFFIX_PATH = os.environ.get("TTS_SUFFIX_PATH",
    str(_BAR_ROOT / "sounds" / "voice-soundeffects" / "suffix_transmission.mp3"))
TTS_SFX_VOLUME_DB    = int(os.environ.get("TTS_SFX_VOLUME_DB",    "10"))  # background crackle
TTS_PREFIX_VOLUME_DB = int(os.environ.get("TTS_PREFIX_VOLUME_DB", "0"))   # prefix beep
TTS_SUFFIX_VOLUME_DB = int(os.environ.get("TTS_SUFFIX_VOLUME_DB", "0"))   # suffix sound
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

# pydub SFX mixing (set inside the try block below)
_pydub_available = False
_AudioSegment = None  # pydub.AudioSegment class (None when unavailable)
_tts_sfx    = None   # background radio crackle AudioSegment
_tts_prefix = None   # prefix beep AudioSegment
_tts_suffix = None   # suffix transmission AudioSegment

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
        _out_mode = "sounddevice" if _sd_output_available else f"pipe→ {TTS_PLAYER_CMD.split()[0]}"
        print(f"[tts] ElevenLabs ready  voice={EL_VOICE_ID}  model={EL_MODEL_ID}  output={_out_mode}")
    else:
        print("[tts] ELEVENLABS_API_KEY not set — TTS disabled.")
except ImportError as _tts_import_err:
    print(f"[tts] Optional deps missing ({_tts_import_err}) — TTS disabled.")
    print("      pip install elevenlabs sounddevice numpy")

# pydub for radio SFX mixing (optional — SFX silently skipped when missing)
try:
    from pydub import AudioSegment as _AudioSegment

    _pydub_available = True

    def _load_sfx(path: str, volume_db: int = 0):
        """Load an mp3 SFX file and apply a volume offset. Returns None on failure."""
        try:
            seg = _AudioSegment.from_file(path, format="mp3")
            return seg + volume_db if volume_db else seg
        except Exception as _e:
            print(f"[tts] SFX not found ({path}): {_e}")
            return None

    _tts_sfx    = _load_sfx(TTS_SFX_PATH,    TTS_SFX_VOLUME_DB)
    _tts_prefix = _load_sfx(TTS_PREFIX_PATH, TTS_PREFIX_VOLUME_DB)
    _tts_suffix = _load_sfx(TTS_SUFFIX_PATH, TTS_SUFFIX_VOLUME_DB)
    print(
        f"[tts] pydub SFX mixing enabled  "
        f"sfx={_tts_sfx is not None}  "
        f"prefix={_tts_prefix is not None}  "
        f"suffix={_tts_suffix is not None}"
    )
except ImportError:
    print("[tts] pydub not installed — radio SFX mixing disabled.  pip install pydub")


def _speak(text: str, pose: str = "calm") -> None:
    """
    Generate TTS audio via ElevenLabs and play it, feeding per-chunk RMS
    amplitude to the game UI portrait.
    When pydub is available: requests mp3_44100_128, mixes in radio SFX
    (background crackle overlay + prefix/suffix comm sounds) via pydub.
    Falls back to raw pcm_16000 when pydub is not installed.
    pose: "attack" or "calm" — selects which sprite the game UI displays.
    Blocks until playback is complete.
    """
    if _el_client is None:
        return
    import re as _re

    clean = _re.sub(r"[*_`#]+", "", text).strip()
    if not clean:
        return
    try:
        if _pydub_available:
            # ------------------------------------------------------------------
            # pydub path: mp3_44100_128 + radio SFX mixing
            # ------------------------------------------------------------------
            raw = _el_client.text_to_speech.convert(
                voice_id=EL_VOICE_ID,
                text=clean,
                model_id=EL_MODEL_ID,
                output_format="mp3_44100_128",
            )
            audio_bytes: bytes = (
                raw if isinstance(raw, (bytes, bytearray)) else b"".join(raw)
            )

            speech = _AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")

            # Overlay looped background crackle SFX on the speech
            if _tts_sfx is not None:
                sfx = _tts_sfx
                if len(sfx) < len(speech):
                    sfx = sfx * (len(speech) // len(sfx) + 1)
                sfx = sfx[: len(speech)]
                speech_mixed = speech.overlay(sfx)
            else:
                speech_mixed = speech

            # Assemble: [prefix] + (speech + sfx) + [suffix]
            mixed = speech_mixed
            if _tts_prefix is not None:
                mixed = _tts_prefix + mixed
            if _tts_suffix is not None:
                mixed = mixed + _tts_suffix

            SAMPLE_RATE = mixed.frame_rate
            CHUNK = int(SAMPLE_RATE * 0.05)  # ~50 ms blocks

            # Derive float32 array for amplitude tracking
            mono = mixed.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
            audio_f32 = (
                _np.frombuffer(mono.raw_data, dtype=_np.int16).astype(_np.float32)
                / 32768.0
            )
            duration = len(audio_f32) / SAMPLE_RATE

            # Notify the game: show the portrait with the requested pose
            try:
                _post("/tts/start", {"duration": duration + 0.5, "pose": pose})
            except Exception:
                pass

            if _sd_output_available:
                # Native path — play float32 via sounddevice
                with _sd.OutputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    device=SD_OUTPUT_DEVICE,
                ) as stream:
                    for i in range(0, len(audio_f32), CHUNK):
                        chunk = audio_f32[i : i + CHUNK]
                        if len(chunk) < CHUNK:
                            chunk = _np.pad(chunk, (0, CHUNK - len(chunk)))
                        stream.write(chunk)
                        rms = float(_np.sqrt(_np.mean(chunk**2)))
                        amplitude = min(1.0, rms * EL_AMP_SCALE)
                        try:
                            _post("/tts/amplitude", {"value": amplitude})
                        except Exception:
                            pass
            else:
                # WSL / no-output-device fallback: export mixed audio as mp3
                # and pipe to ffplay; send amplitude updates in parallel.
                buf = io.BytesIO()
                mixed.export(buf, format="mp3")
                mp3_bytes = buf.getvalue()
                cmd = TTS_PLAYER_CMD_MP3.split()
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                try:
                    proc.stdin.write(mp3_bytes)
                    proc.stdin.close()
                    # Amplitude updates (approximate — runs while ffplay decodes)
                    for i in range(0, len(audio_f32), CHUNK):
                        chunk_f = audio_f32[i : i + CHUNK]
                        rms = float(_np.sqrt(_np.mean(chunk_f**2)))
                        amplitude = min(1.0, rms * EL_AMP_SCALE)
                        try:
                            _post("/tts/amplitude", {"value": amplitude})
                        except Exception:
                            pass
                        time.sleep(CHUNK / SAMPLE_RATE)
                except Exception:
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass
                proc.wait()
        else:
            # ------------------------------------------------------------------
            # Legacy path: raw PCM s16le @ 16 kHz — no SFX mixing
            # ------------------------------------------------------------------
            raw = _el_client.text_to_speech.convert(
                voice_id=EL_VOICE_ID,
                text=clean,
                model_id=EL_MODEL_ID,
                output_format="pcm_16000",
            )
            audio_bytes = (
                raw if isinstance(raw, (bytes, bytearray)) else b"".join(raw)
            )
            audio_i16 = _np.frombuffer(audio_bytes, dtype=_np.int16)
            audio_f32 = audio_i16.astype(_np.float32) / 32768.0
            duration = len(audio_f32) / 16000.0

            try:
                _post("/tts/start", {"duration": duration + 0.5, "pose": pose})
            except Exception:
                pass

            SAMPLE_RATE = 16000
            CHUNK = int(SAMPLE_RATE * 0.05)  # 800 samples = 50 ms blocks

            if _sd_output_available:
                with _sd.OutputStream(
                    samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                    device=SD_OUTPUT_DEVICE,
                ) as stream:
                    for i in range(0, len(audio_f32), CHUNK):
                        chunk = audio_f32[i : i + CHUNK]
                        if len(chunk) < CHUNK:
                            chunk = _np.pad(chunk, (0, CHUNK - len(chunk)))
                        stream.write(chunk)
                        rms = float(_np.sqrt(_np.mean(chunk**2)))
                        amplitude = min(1.0, rms * EL_AMP_SCALE)
                        try:
                            _post("/tts/amplitude", {"value": amplitude})
                        except Exception:
                            pass
            else:
                # WSL / no-output-device fallback: pipe raw s16le PCM to ffplay.exe
                cmd = TTS_PLAYER_CMD.split()
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                try:
                    for i in range(0, len(audio_i16), CHUNK):
                        chunk_i16 = audio_i16[i : i + CHUNK]
                        if len(chunk_i16) < CHUNK:
                            chunk_i16 = _np.pad(chunk_i16, (0, CHUNK - len(chunk_i16)))
                        proc.stdin.write(chunk_i16.tobytes())
                        chunk_f = chunk_i16.astype(_np.float32) / 32768.0
                        rms = float(_np.sqrt(_np.mean(chunk_f**2)))
                        amplitude = min(1.0, rms * EL_AMP_SCALE)
                        try:
                            _post("/tts/amplitude", {"value": amplitude})
                        except Exception:
                            pass
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
def get_game_state(mode: str = "summary") -> str:
    """
    Returns current global game state. Choose mode based on what you need:

    "summary" (default) — frame, mapInfo, per-team resources (metal/energy
      income/expense/storage) and unit COUNTS per category. No individual
      units. Use this for quick situational awareness before deciding what
      to do. Cheapest call.

    "units" — frame, mapInfo, per-team resources + slim unit list per team
      (unitID, name, x/y/z, health/maxHealth, boolean flags). No buildOptions.
      Use when you need unit positions or IDs across all teams.

    "full" — same as "units" but also includes each unit's buildOptions name
      list. Use only when you need to check build options for multiple units
      at once; prefer get_unit_details([id]) for a single unit.

    Args:
        mode: One of "summary" (default), "units", "full".
    """
    _UNIT_SLIM = {
        "unitID", "name", "x", "y", "z", "health", "maxHealth",
        "isCommander", "isFactory", "isBuilder",
        "canMove", "canAttack", "isExtractor", "isGenerator",
    }

    def _categorise(unit: dict) -> str:
        is_cmd  = unit.get("isCommander", False)
        is_fact = unit.get("isFactory",   False)
        is_bld  = unit.get("isBuilder",   False)
        can_mv  = unit.get("canMove",     False)
        can_atk = unit.get("canAttack",   False)
        is_ext  = unit.get("isExtractor", False)
        is_gen  = unit.get("isGenerator", False)
        if is_cmd:                                    return "commander"
        if is_fact:                                   return "factory"
        if is_bld and can_mv:                         return "constructor"
        if can_mv and can_atk and not is_bld:         return "combat"
        if not can_mv and can_atk and not is_fact:    return "defense"
        if is_ext:                                    return "extractor"
        if is_gen:                                    return "generator"
        return "other"

    try:
        state = _get("/state")
    except ConnectionError as e:
        return f"ERROR: {e}"

    _RESOURCE_KEYS = (
        "teamID", "isBot", "isMyTeam", "luaAI",
        "metal", "metalIncome", "metalExpense", "metalStorage",
        "energy", "energyIncome", "energyExpense", "energyStorage",
    )

    teams_out = []
    for team in state.get("teams", []):
        entry = {k: team[k] for k in _RESOURCE_KEYS if k in team}
        units = team.get("units", [])

        if mode == "summary":
            counts: dict = {}
            for u in units:
                cat = _categorise(u)
                counts[cat] = counts.get(cat, 0) + 1
            entry["unitCounts"] = counts

        elif mode == "units":
            entry["units"] = [
                {k: v for k, v in u.items() if k in _UNIT_SLIM}
                for u in units
            ]

        else:  # "full"
            def _slim_full(u: dict) -> dict:
                out = {k: v for k, v in u.items() if k in _UNIT_SLIM}
                opts = u.get("buildOptions")
                if opts:
                    out["buildOptions"] = [o["name"] for o in opts]
                return out
            entry["units"] = [_slim_full(u) for u in units]

        teams_out.append(entry)

    out: dict = {
        "frame": state.get("frame"),
        "mapInfo": state.get("mapInfo"),
        "teams": teams_out,
    }
    if mode == "summary":
        out["visibleEnemyCount"] = len(state.get("visibleEnemies", []))
    else:
        out["visibleEnemies"] = state.get("visibleEnemies", [])

    result = json.dumps(out, separators=(",", ":"))
    print(f"[get_game_state] mode={mode!r} \n {result}")
    return result


@tool
def get_game_frame() -> str:
    """
    Returns the current game frame number only. This is an extremely cheap call.

    MANDATORY — call this at the very start of EVERY response, before anything else.
    Use the returned frame to decide whether your cached state is stale:

      • No prior get_game_state() in this conversation
            → call get_game_state() immediately after.
      • Current frame is more than 300 frames ahead of the frame in your last
        get_game_state() response  (300 frames ≈ 10 seconds at 30 fps)
            → state is STALE.  Call get_game_state() before answering or acting.
      • Frame delta ≤ 300
            → state is fresh enough.  You may skip get_game_state() and act on
               cached data.

    BAR runs at 30 frames/second.  Frame 9000 ≈ 5 minutes into the game.
    """
    try:
        state = _get("/state")
        frame = state.get("frame", -1)
        print(f"[get_game_frame] frame={frame}")
        return json.dumps({"frame": frame})
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
    INTERNAL — used by the agent polling loop (run_chat_loop), NOT by the LLM.

    The loop polls /chat automatically and routes each player message as a new
    agent invocation.  The LLM should never call this tool directly — doing so
    would produce an empty list (the loop already drained the queue) and risks
    confusing the control flow.

    Drains and returns all chat messages that have arrived since the last call.
    Returns a JSON list of objects: [{text, frame}, ...].
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
                   IMPORTANT: for build orders where you need completion
                   tracking, use reserve_and_build() instead — it atomically
                   reserves the builder, issues the order, and registers the
                   idle watch in a single call, preventing race conditions.
                   Use this 'build' command only for factory queue builds
                   (set_rally) or when you have already reserved the unit.
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

    Returns a compact JSON list. Each entry contains only:
      unitID, name, x, y, z, health, maxHealth,
      isCommander, isFactory, isBuilder, canMove, canAttack,
      isExtractor, isGenerator, teamID, isBot.

    buildOptions are NOT included. Call get_unit_details([unitID]) if you
    need to inspect a builder's full build options before issuing an order.
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

    _SLIM_FIELDS = {
        "unitID", "name", "humanName",
        "x", "y", "z",
        "health", "maxHealth",
        "isCommander", "isFactory", "isBuilder",
        "canMove", "canAttack", "isExtractor", "isGenerator",
    }

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
            slim = {k: v for k, v in unit.items() if k in _SLIM_FIELDS}
            slim["teamID"] = team["teamID"]
            slim["isBot"] = is_bot
            result.append(slim)
    print(result)
    # One compact object per line: fewer tokens than indent=2, easier to scan than a single blob
    lines = [json.dumps(u, separators=(",", ":")) for u in result]
    return "[\n" + ",\n".join(lines) + "\n]"


@tool
def get_unit_details(unit_ids: list) -> str:
    """
    Returns full unit data (including buildOptions) for a specific list of
    unit IDs. Use this when you need detailed information about particular
    units — for example, to inspect a builder's full buildOptions list before
    deciding what to construct.

    find_allied_units() returns a slim summary intentionally; call this tool
    to get the complete picture for units you have already identified.

    Args:
        unit_ids: List of integer Spring unit IDs.

    Returns a JSON list with all available fields per unit, including
    buildOptions as a flat list of defName strings: ["armmex", "armsolar", ...]
    """
    try:
        state = _get("/state")
    except ConnectionError as e:
        return f"ERROR: {e}"

    id_set = set(unit_ids)
    result = []
    for team in state.get("teams", []):
        for unit in team.get("units", []):
            if unit.get("unitID") in id_set:
                entry = {**unit, "teamID": team["teamID"], "isBot": team.get("isBot", False)}
                # Normalise buildOptions to flat name strings (same as get_game_state("full"))
                opts = entry.get("buildOptions")
                if opts and isinstance(opts[0], dict):
                    entry["buildOptions"] = [o["name"] for o in opts]
                result.append(entry)
    print(f"[get_unit_details] {[u['unitID'] for u in result]}")
    lines = [json.dumps(u, separators=(",", ":")) for u in result]
    return "[\n" + ",\n".join(lines) + "\n]"


@tool
def get_enemy_intel() -> str:
    """
    Returns only the enemy units currently visible to the local player
    (not hidden in fog of war). Includes radar blips (no defName, position
    only) and fully visible units with position and estimated health.

    Use this when you ONLY need enemy data — it returns far fewer tokens
    than get_game_state("units") which also returns all allied unit lists.
    Typical uses: target acquisition, threat assessment, planning a strike.
    """
    try:
        state = _get("/state")
        enemies = state.get("visibleEnemies", [])
        lines = [json.dumps(e, separators=(",", ":")) for e in enemies]
        return "[\n" + ",\n".join(lines) + "\n]"
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
    'buildOptions' list. Call get_unit_details([unit_id]) to inspect a
    builder's options if unsure. This call validates automatically and will
    return an error listing valid names if build_type is wrong.

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


# ---------------------------------------------------------------------------
# Active-task registry  (task_id → context for event-driven continuation)
# ---------------------------------------------------------------------------
activeTasks: dict = {}

# ---------------------------------------------------------------------------
# Module-level work queue  (shared by chat-loop thread AND STT callback)
# ---------------------------------------------------------------------------
_work_queue:   _queue.PriorityQueue = _queue.PriorityQueue()
_seq_counter:  int = 0
_seq_lock:     threading.Lock = threading.Lock()
_queued_tasks: set = set()
_queued_lock:  threading.Lock = threading.Lock()


def _next_seq() -> int:
    global _seq_counter
    with _seq_lock:
        _seq_counter += 1
        return _seq_counter


def _enqueue(inp: str, task_id: "str | None", priority: int) -> None:
    """Thread-safe enqueue for agent work items."""
    if task_id:
        with _queued_lock:
            if task_id in _queued_tasks:
                print(f"[skip] task {task_id!r} already queued")
                return
            _queued_tasks.add(task_id)
    _work_queue.put((priority, _next_seq(), inp, task_id))
    print(f"[queue] enqueued priority={priority} task={task_id!r} len={_work_queue.qsize()}")


# ---------------------------------------------------------------------------
# STT coroutines  (Voxtral real-time transcription, merged from SpeechToTextDaemon)
# ---------------------------------------------------------------------------
_stt_audio_q: asyncio.Queue   # filled by audio_reader, consumed by gated_audio_stream
_stt_start:   asyncio.Event   # key pressed
_stt_stop:    asyncio.Event   # key released
_stt_cancel:  asyncio.Event   # key cancelled


async def _stt_mic_stream() -> AsyncIterator[bytes]:
    """Capture PCM s16le from the default mic via sounddevice."""
    loop = asyncio.get_event_loop()
    q: asyncio.Queue[bytes] = asyncio.Queue()

    def _cb(indata, frames, time_info, status):
        loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

    with _sd.InputStream(
        samplerate=STT_SAMPLE_RATE, channels=STT_CHANNELS,
        dtype="int16", blocksize=STT_BLOCK_SIZE, callback=_cb,
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
        try: _stt_audio_q.get_nowait(); drained += 1
        except asyncio.QueueEmpty: break
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
                try: _stt_audio_q.get_nowait()
                except asyncio.QueueEmpty: break
            return
        if _stt_cancel.is_set():
            _stt_cancel.clear()
            while not _stt_audio_q.empty():
                try: _stt_audio_q.get_nowait()
                except asyncio.QueueEmpty: break
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
                print(f"[stt] Routing to agent: {text!r}")
                # Directly enqueue — no game-chat round-trip
                _enqueue(f"[Voice] {text}", task_id=None, priority=0)
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
    for f in [STT_TRIGGER_FILE, STT_STOP_FILE, STT_CANCEL_FILE,
              STT_DONE_FILE, STT_RESULT_FILE]:
        f.unlink(missing_ok=True)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _stt_audio_q = asyncio.Queue()
    _stt_start   = asyncio.Event()
    _stt_stop    = asyncio.Event()
    _stt_cancel  = asyncio.Event()

    print(f"[stt] STT thread started. IPC dir: {_IPC_DIR}")
    print(f"[stt] Mic mode: {'sounddevice' if _sd_available else 'stdin (ffmpeg pipe)'}")
    try:
        loop.run_until_complete(asyncio.gather(
            _stt_audio_reader(),
            _stt_session_loop(),
            _stt_trigger_loop(),
        ))
    except Exception as exc:
        print(f"[stt] Fatal error in STT loop: {exc}")
    finally:
        loop.close()


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

Unit data schema — fields you will see in tool responses:
- unitID   : integer — the LIVE instance ID. Use this in all commands (command_unit,
             reserve_units, watch_unit, etc.). Unique per game session.
- name     : string — the unit TYPE defName (e.g. 'armcom', 'armbeaver', 'armmex').
             This is also what you pass as unit_type in build orders.
- defID    : integer — internal engine type index. You will never need to pass this
             to any tool; ignore it.
- isCommander / isFactory / isBuilder / canMove / canAttack / isExtractor / isGenerator:
             boolean flags describing the unit's role and capabilities.
- health / maxHealth : current and maximum hit-points.
- x, y, z  : world-space position. x = east-west, z = north-south, y = altitude.
             Build orders use x and z only (y is terrain height, auto-calculated).
- buildOptions : list of defNames this unit is allowed to build. Only present on
             builders and factories. ALWAYS pick a name from this list when issuing
             a build order — using any other name will silently fail or error.
- teamID   : which team owns this unit.
- isBot / isMyTeam : whether the owning team is the AI or the human player.

Your role:
- You watch in-game chat and respond when a player addresses you with
  "@agent" or starts a message with "!".
- You can read the game state, issue unit orders, queue factory production,
  send chat replies, and coordinate multi-unit actions.
MANDATORY state-awareness rules — follow these on EVERY invocation:
1. ALWAYS call get_game_frame() first, before anything else.
2. If no get_game_state() result exists in your conversation history, call
   get_game_state() immediately ("summary" mode is cheapest).
3. If the frame returned by get_game_frame() is more than 300 frames ahead
   of the frame in your last get_game_state() response (~10 s), your cached
   state is STALE — call get_game_state() before answering or issuing orders.
4. If the player is asking for information about the game (resources, units,
   map, enemies, etc.), ALWAYS call get_game_state() regardless of staleness.
5. If you are about to issue a command (build, attack, move…), ALWAYS call
   get_game_state() first if your state is stale (rule 3) OR if you do not
   already have the relevant unit IDs in your context.
6. Use get_game_state("units") only when you need positions/IDs across teams.
   Prefer find_allied_units() to locate specific bot units to command.
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
    # ── State queries ──────────────────────────────────────────────────────
    get_game_frame,       # current frame only — mandatory staleness check
    get_game_state,       # quick summary / slim unit list / full unit list
    get_build_catalog,    # all unit defs grouped by category
    find_allied_units,    # category-filtered allied unit search
    get_unit_details,     # full data (incl. buildOptions) for specific unit IDs
    get_enemy_intel,      # visible enemies only (token-efficient subset of state)
    get_build_queue,      # pending orders for factories / constructors
    # ── Communication ──────────────────────────────────────────────────────
    send_chat_message,    # broadcast in-game + TTS — ONLY way to talk to the player
    map_ping,             # place a visible map marker
    # ── Unit commands ──────────────────────────────────────────────────────
    command_unit,         # single order (move/attack/patrol/fight/stop/reclaim/repair/guard/set_rally)
    command_units_batch,  # convenience wrapper: multiple command_unit calls in one tool call
    # ── Reservation & event watching ───────────────────────────────────────
    reserve_units,        # lock unit(s) so native AI cannot override
    unreserve_units,      # release lock when task is done or failed
    watch_unit,           # register event listener (idle/finished/from_factory/destroyed)
    unwatch_unit,         # cancel a previously registered watch
    reserve_and_build,    # atomic: reserve + build order + watch idle (preferred for builds)
    # ── Unit transfer ──────────────────────────────────────────────────────
    gift_units,           # transfer bot-owned units to the human player
]
# NOTE: get_new_chat_messages is intentionally excluded from BAR_TOOLS.
# The agent loop (run_chat_loop) already routes chat messages to the LLM as
# new invocations — the LLM must never poll it directly.


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
        system_prompt=SYSTEM_PROMPT,
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
                user_input = _strip_prefix(text)
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
    stt_thread = threading.Thread(
        target=_run_stt_loop, daemon=True, name="stt-daemon"
    )
    stt_thread.start()
    print("[main] STT daemon thread started.")

    try:
        run_chat_loop(agent)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
