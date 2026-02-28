#!/usr/bin/env python3
"""
agent_bridge_client.py — Example LLM/Python agent for Beyond All Reason
========================================================================

This script connects to the AgentBridge widget HTTP server running inside BAR
(default: http://127.0.0.1:7654) and demonstrates how to:

    1. Fetch the current game state (all ally-team units + resources)
    2. Find the allied bot commander
    3. Issue a self-destruct command to it

Architecture
------------
  Python script  ←HTTP→  AgentBridge widget (LuaUI)
                               │
                         Spring.SendLuaRulesMsg
                               │
                    AgentBridgeRelay gadget (LuaRules synced)
                               │
                    Spring.GiveOrderToUnit(commander, CMD.SELFD)

Prerequisites
-------------
  pip install requests

  In-game: enable the AgentBridge widget AND ensure TCPAllowListen=1 in Spring
  config (usually ~/.spring/springsettings.cfg or via /set TCPAllowListen 1
  followed by a widget reload).

Usage
-----
  python tools/agent_bridge_client.py

  Optionally point to a different server:
  python tools/agent_bridge_client.py --host 127.0.0.1 --port 7654
"""

import argparse
import json
import sys
import time
from typing import Any

try:
    import requests
except ImportError:
    sys.exit(
        "The 'requests' library is required. Install it with:\n"
        "  pip install requests"
    )

# ─── API helpers ─────────────────────────────────────────────────────────────

BASE_URL = "http://127.0.0.1:7654"


def get_state(base_url: str) -> dict[str, Any]:
    """Fetch the current game state from the AgentBridge widget."""
    resp = requests.get(f"{base_url}/state", timeout=5)
    resp.raise_for_status()
    return resp.json()


def send_command(base_url: str, **kwargs: Any) -> dict[str, Any]:
    """
    Send a command to the widget.  kwargs become the JSON body.

    Minimum required keys:
        unitID (int)        – target unit
        cmd    (str)        – verb: 'selfd', 'move', 'attack', 'stop', …

    Optional:
        shift (bool)        – append to the unit's queue instead of replacing
        x, y, z (float)    – world-space position for move/patrol/fight/…
        targetID (int)      – enemy/ally unit or feature ID for attack/repair/…
    """
    resp = requests.post(
        f"{base_url}/command",
        json=kwargs,
        timeout=5,
    )
    resp.raise_for_status()
    return resp.json()


# ─── State inspection helpers ─────────────────────────────────────────────────


def find_allied_bot_commanders(state: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Return all commander units that belong to allied BOT teams.

    A bot team has `isBot == True` and `isMyTeam == False`.
    """
    commanders = []
    for team in state.get("teams", []):
        if team.get("isBot") and not team.get("isMyTeam"):
            for unit in team.get("units", []):
                if unit.get("isCommander"):
                    commanders.append({**unit, "teamID": team["teamID"]})
    return commanders


def find_my_commander(state: dict[str, Any]) -> dict[str, Any] | None:
    """Return the player's own commander (the human-controlled team)."""
    for team in state.get("teams", []):
        if team.get("isMyTeam"):
            for unit in team.get("units", []):
                if unit.get("isCommander"):
                    return unit
    return None


def pretty_team_summary(state: dict[str, Any]) -> str:
    """Return a human-readable summary of all ally teams."""
    lines = [f"=== Game frame: {state.get('frame', '?')} ==="]
    for team in state.get("teams", []):
        tag = []
        if team.get("isMyTeam"):
            tag.append("YOU")
        if team.get("isBot"):
            tag.append(f"BOT({team.get('luaAI', '?')})")
        unit_count = len(team.get("units", []))
        metal = team.get("metal", 0)
        energy = team.get("energy", 0)
        lines.append(
            f"  Team {team['teamID']:2d} [{', '.join(tag) or 'ally'}]  "
            f"units={unit_count:3d}  M={metal:5d}  E={energy:5d}"
        )
    return "\n".join(lines)


# ─── Demo actions ─────────────────────────────────────────────────────────────


def demo_selfd_allied_bot_commander(base_url: str) -> None:
    """
    Full demo:
      1. Fetch game state
      2. Print a summary
      3. Find the first allied bot commander
      4. Send it a self-destruct order
    """
    print("Connecting to AgentBridge at", base_url, "…")
    state = get_state(base_url)

    print(pretty_team_summary(state))
    print()

    commanders = find_allied_bot_commanders(state)
    if not commanders:
        print("No allied bot commander found.")
        print("Is there a bot in your ally team?  Is the game running?")
        return

    commander = commanders[0]
    print(
        f"Found allied bot commander: unitID={commander['unitID']}"
        f"  name={commander['name']}"
        f"  pos=({commander['x']}, {commander['y']}, {commander['z']})"
        f"  HP={commander['health']}/{commander['maxHealth']}"
        f"  team={commander['teamID']}"
    )
    print()

    user_input = input("Send SELF-DESTRUCT to this commander? [y/N] ").strip().lower()
    if user_input != "y":
        print("Aborted.")
        return

    result = send_command(base_url, unitID=commander["unitID"], cmd="selfd")
    print("Response:", json.dumps(result, indent=2))
    print("Self-destruct order sent!  Watch the commander explode in-game.")


def demo_move_my_commander(base_url: str, x: float, z: float) -> None:
    """
    Move the player's own commander to a world-space position.
    The y coordinate is automatically set to ground level by the engine.
    """
    state = get_state(base_url)
    com = find_my_commander(state)
    if not com:
        print("Own commander not found (already dead?).")
        return

    result = send_command(base_url, unitID=com["unitID"], cmd="move", x=x, y=0, z=z)
    print(f"Move order sent to own commander (unitID={com['unitID']}):", result)


# ─── Polling loop skeleton (for a real LLM agent) ─────────────────────────────


def run_agent_loop(base_url: str, tick_seconds: float = 1.0) -> None:
    """
    Skeleton for a real LLM-driven agent loop.

    At each tick:
      - Fetch state
      - Pass it to your LLM (or rule-based logic)
      - Execute the returned commands
    """
    print("Starting agent loop (Ctrl-C to stop) …")
    while True:
        try:
            state = get_state(base_url)
            frame = state.get("frame", 0)

            # ── Your LLM / logic goes here ─────────────────────────────────
            # Example: just print the frame number every tick
            print(f"[frame {frame:6d}]  teams={len(state.get('teams', []))}", end="\r")
            # commands = llm.decide(state)         # call your LLM
            # for cmd in commands:                 # issue each command
            #     send_command(base_url, **cmd)
            # ──────────────────────────────────────────────────────────────

        except requests.exceptions.ConnectionError:
            print("\n[agent] Connection lost – waiting for game …", end="\r")
        except Exception as exc:  # noqa: BLE001
            print(f"\n[agent] Error: {exc}")

        time.sleep(tick_seconds)


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AgentBridge client — LLM agent demo for Beyond All Reason",
    )
    p.add_argument(
        "--host", default="127.0.0.1", help="AgentBridge host (default: 127.0.0.1)"
    )
    p.add_argument(
        "--port", default=7654, type=int, help="AgentBridge port (default: 7654)"
    )
    p.add_argument(
        "--action",
        choices=["selfd-bot-com", "move-my-com", "state", "loop"],
        default="selfd-bot-com",
        help=(
            "selfd-bot-com : self-destruct the first allied bot commander (default)\n"
            "move-my-com   : move your own commander (use with --x / --z)\n"
            "state         : just print the current game state\n"
            "loop          : run the agent polling loop skeleton\n"
        ),
    )
    p.add_argument("--x", type=float, default=1000.0, help="World X for move-my-com")
    p.add_argument("--z", type=float, default=1000.0, help="World Z for move-my-com")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"

    if args.action == "selfd-bot-com":
        demo_selfd_allied_bot_commander(base_url)

    elif args.action == "move-my-com":
        demo_move_my_commander(base_url, args.x, args.z)

    elif args.action == "state":
        state = get_state(base_url)
        print(pretty_team_summary(state))
        print()
        print("Full JSON:")
        print(json.dumps(state, indent=2))

    elif args.action == "loop":
        run_agent_loop(base_url)


if __name__ == "__main__":
    main()
