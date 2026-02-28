#!/usr/bin/env python3
"""
bar_agent.py  –  Beyond All Reason AI Agent
============================================
Connects to the in-game AgentBridge HTTP server, listens to in-game chat,
and responds with real commands via a Strands + Mistral LLM agent.

Usage
-----
    export MISTRAL_API_KEY="your_key_here"
    python tools/bar_agent.py

Optional env vars:
    BAR_HOST  (default: 127.0.0.1)
    BAR_PORT  (default: 7654)
    MISTRAL_MODEL  (default: mistral-large-latest)
    AGENT_PREFIX  – chat prefix that triggers the agent (default: @agent)
    AGENT_PREFIX2 – alternative prefix (default: !)

Requirements
------------
    pip install 'strands-agents[mistral]' strands-agents-tools
"""

import json
import os
import sys
import time
import threading
import urllib.request
import urllib.error
from typing import Optional

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
            return json.loads(resp.read().decode())
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
def send_chat_message(message: str) -> str:
    """
    Sends a chat message visible to all players in-game.
    Use this to acknowledge commands, give status updates, or communicate
    with teammates.

    Args:
        message: The text to broadcast in the in-game 'All' chat channel.
    """
    try:
        resp = _post("/chat/send", {"message": message})
        print(f"[send_chat] sent: {message!r} → {resp}")
        return json.dumps(resp)
    except (ConnectionError, RuntimeError) as e:
        print(f"[send_chat] ERROR: {e}")
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
def find_allied_units(category: str = "", name_filter: str = "", owner: str = "bot") -> str:
    """
    Returns a filtered list of allied units from the current game state.

    Args:
        category:    Filter by unit category. Examples: 'factory', 'constructor',
                     'extractor', 'commander', 'generator'. Empty = all categories.
        name_filter: Filter by defName substring, e.g. 'arm_'. Empty = all.
        owner:       Who owns the units to include:
                     'bot'    – only AI/bot teams (DEFAULT — use this normally)
                     'human'  – only the human player's own units
                     'all'    – all allied units regardless of owner

    IMPORTANT: Default is 'bot'. Always use owner='bot' unless the player
    explicitly asks you to command their own units.

    Returns a JSON list of unit objects: {unitID, name, humanName, isCommander,
    isFactory, canMove, x, y, z, health, maxHealth, buildOptions, teamID, isBot}
    """
    try:
        state = _get("/state")
    except ConnectionError as e:
        return f"ERROR: {e}"

    result = []
    # State returns 'teams' (all ally teams), each with isBot and isMyTeam flags
    for team in state.get("teams", []):
        is_bot = team.get("isBot", False)
        is_my_team = team.get("isMyTeam", False)
        if owner == "bot" and not is_bot:
            continue
        if owner == "human" and not is_my_team:
            continue
        for unit in team.get("units", []):
            if category and not unit.get(category.lower(), False) and \
               unit.get("category", "").lower() != category.lower():
                # Try matching isCommander, isFactory, canMove flags too
                flag_map = {
                    "commander": "isCommander",
                    "factory": "isFactory",
                    "constructor": "isBuilder",
                }
                flag = flag_map.get(category.lower())
                if not (flag and unit.get(flag)):
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
- Be concise in chat replies (≤ 2 sentences). Use send_chat_message to confirm
  what you are doing.
- Never attack or reclaim allied units.
- When building units in a factory, use the exact defName from get_build_catalog.

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
    command it to build the lab, confirm in chat.

  Player: "! we need more metal"
  → find_allied_units(category='constructor', owner='bot'), queue metal extractor builds.

  Player: "@agent attack the east with everything we have"
  → get enemy positions to the east, issue fight-move to BOT combat units.
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
]


def build_agent() -> Agent:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        sys.exit("Set the MISTRAL_API_KEY environment variable before running.")

    model = MistralModel(
        model_id=MODEL_ID,
        api_key=api_key,
    )

    return Agent(
        model=model,
        tools=BAR_TOOLS,
        system_prompt=SYSTEM_PROMPT,
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
    When a player address the agent, forward the message to the LLM.
    Agent responses are sent back as in-game chat via /chat/send.
    """
    print(
        f"AgentBridge chat loop started (polling {BASE_URL}/chat every {POLL_INTERVAL}s)"
    )
    print(f"Trigger prefixes: '{AGENT_PREFIX}'  '{AGENT_PREFIX2}'")
    print("Press Ctrl-C to stop.\n")

    while True:
        try:
            messages = _get("/chat")
        except ConnectionError as e:
            print(f"[warn] connection lost — {e} — retrying in 5s")
            time.sleep(5.0)
            continue
        except RuntimeError as e:
            # HTTP error from the widget (e.g. 404 = wrong widget version loaded)
            print(f"[error] {e}")
            print(
                "  → Reload the widget in-game: F11 → AgentBridge → disable then enable."
            )
            time.sleep(10.0)
            continue

        for entry in messages:
            text = entry.get("text", "")
            print(f"[poll] {text!r}")
            if not _should_route(text):
                continue

            user_input = _strip_prefix(text)
            print(f"[→ agent] routing: {user_input!r}")

            # Run agent in a thread so we keep polling while it thinks
            def _run(inp=user_input):
                print(f"[agent] calling LLM with: {inp!r}")
                try:
                    response = agent(inp)
                    reply = str(response).strip()  # remove trailing \n
                    print(f"[agent] raw response ({len(reply)} chars): {reply[:300]!r}")
                    # Truncate very long replies for in-game chat
                    if len(reply) > 200:
                        reply = reply[:197] + "..."
                    print(f"[← agent] sending to chat: {reply!r}")
                    _post("/chat/send", {"message": f"[AI] {reply}"})
                except Exception as exc:
                    print(f"[agent error] {type(exc).__name__}: {exc}")
                    try:
                        _post("/chat/send", {"message": f"[AI] Error: {exc}"})
                    except Exception as e2:
                        print(f"[agent error] also failed to send error to chat: {e2}")

            threading.Thread(target=_run, daemon=True).start()

        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    agent = build_agent()
    try:
        run_chat_loop(agent)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
