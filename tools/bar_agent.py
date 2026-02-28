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
        # Split messages longer than 250 chars into multiple sends
        results = []
        parts = [message[i : i + 250] for i in range(0, len(message), 250)]
        for part in parts:
            resp = _post("/chat/send", {"message": part})
            results.append(resp)
            print(f"[send_chat] sent: {part!r} → {resp}")
        return json.dumps(results if len(results) > 1 else results[0])
    except (ConnectionError, RuntimeError) as e:
        print(f"[send_chat] ERROR: {e}")
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
            if (
                category
                and not unit.get(category.lower(), False)
                and unit.get("category", "").lower() != category.lower()
            ):
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
# Structure:  activeTasks[task_id] = {
#   "original_request":  str,   # user message that started the task
#   "reserved_units":    list,  # unit IDs currently reserved
#   "steps_done":        list,  # short descriptions of completed steps
# }
activeTasks: dict = {}


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
- Always obey the player's commands as literally as possible. If they say to attack an allied unit or something that seems suboptimal, just do it and don't question it. The player is the commander, you are the co-commander.,
- Address the player like a military subordinate, e.g. "Yes, Commander. Building additional metal extractor at (x, z)."

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
            print(f"[tool_call] {name}({inp})")
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

    A single worker thread consumes a PriorityQueue so Strands is never called
    concurrently.  Priority 0 = human input (highest), 1 = task continuation.
    """
    print(
        f"AgentBridge chat loop started (polling {BASE_URL}/chat every {POLL_INTERVAL}s)"
    )
    print(f"Trigger prefixes: '{AGENT_PREFIX}'  '{AGENT_PREFIX2}'")
    print("Press Ctrl-C to stop.\n")

    import queue as _queue

    # (priority, sequence, inp, task_id)
    # sequence breaks ties to preserve FIFO within the same priority.
    _work_queue: _queue.PriorityQueue = _queue.PriorityQueue()
    _seq = 0
    _seq_lock = threading.Lock()
    # task_ids already in the queue (not yet processed) — avoids duplicates
    _queued_tasks: set = set()
    _queued_lock = threading.Lock()

    def _next_seq() -> int:
        nonlocal _seq
        with _seq_lock:
            _seq += 1
            return _seq

    def _enqueue(inp: str, task_id: str | None, priority: int) -> None:
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
                _work_queue.task_done()

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
            print(f"[poll] {text!r}")
            if not _should_route(text):
                continue
            user_input = _strip_prefix(text)
            print(f"[→ agent] routing: {user_input!r}")
            _enqueue(user_input, task_id=None, priority=0)

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
    try:
        run_chat_loop(agent)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
