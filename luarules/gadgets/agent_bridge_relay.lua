-- AgentBridge Relay Gadget (synced)
--
-- Receives commands forwarded by the AgentBridge widget (LuaUI) via
-- Spring.SendLuaRulesMsg, validates them, and issues Spring orders.
--
-- Message format (string):
--   "AGENTBRIDGE:<json>"
--
-- JSON command schema:
--   { "unitID": <int>, "cmd": "<verb>", ...verb-specific params... }
--
-- Supported verbs:
--   "selfd"     – self-destruct the unit
--   "stop"      – stop all current orders
--   "move"      – { "x": <n>, "y": <n>, "z": <n> }
--   "attack"    – { "targetID": <unitID> }  OR  { "x", "y", "z" }
--   "patrol"    – { "x": <n>, "y": <n>, "z": <n> }
--   "fight"     – { "x": <n>, "y": <n>, "z": <n> }
--   "reclaim"   – { "targetID": <featureID or unitID> }
--   "repair"    – { "targetID": <unitID> }
--   "guard"     – { "targetID": <unitID> }
--   "build"     – { "unitType": "<defName>" OR "defID": <n>, "x": <n>, "y": <n>, "z": <n>, "facing": 0-3 }
--   "set_rally" – { "x": <n>, "y": <n>, "z": <n> }
--
-- Security: only ally teams are controllable (not enemy teams).

local gadget = gadget ---@type Gadget

function gadget:GetInfo()
	return {
		name    = "AgentBridge Relay",
		desc    = "Relays LLM/Python agent commands from the AgentBridge widget to game units",
		author  = "AgentBridge",
		date    = "2026",
		license = "GNU GPL, v2 or later",
		layer   = 0,
		enabled = true,
	}
end

-- Only the synced part is used
if not gadgetHandler:IsSyncedCode() then return end

--------------------------------------------------------------------------------
-- Dependencies & constants
--------------------------------------------------------------------------------
local Json = VFS.Include("common/luaUtilities/json.lua")

local MSG_PREFIX     = "AGENTBRIDGE:"
local MSG_PREFIX_LEN = #MSG_PREFIX

--------------------------------------------------------------------------------
-- Spring API locals
--------------------------------------------------------------------------------
local spGiveOrderToUnit  = Spring.GiveOrderToUnit
local spGetUnitTeam      = Spring.GetUnitTeam
local spGetTeamInfo      = Spring.GetTeamInfo
local spGetTeamList      = Spring.GetTeamList
local spGetUnitPosition  = Spring.GetUnitPosition
local spGetGroundHeight  = Spring.GetGroundHeight
local spEcho             = Spring.Echo

--------------------------------------------------------------------------------
-- Build a set of ally-team IDs at game start so we can validate targets
--------------------------------------------------------------------------------
local allyTeamSets = {} -- allyTeamSets[allyTeamID] = { [teamID]=true, ... }

local function buildAllyTeamSets()
	-- We enumerate all teams grouped by their allyTeam
	local allTeams = Spring.GetTeamList()
	for _, teamID in ipairs(allTeams or {}) do
		local _, _, _, _, _, allyTeamID = spGetTeamInfo(teamID)
		if allyTeamID then
			if not allyTeamSets[allyTeamID] then
				allyTeamSets[allyTeamID] = {}
			end
			allyTeamSets[allyTeamID][teamID] = true
		end
	end
end

function gadget:GameStart()
	buildAllyTeamSets()
end

function gadget:Initialize()
	buildAllyTeamSets()
end

-- Returns true if teamA and teamB are on the same alliance
local function sameAllyTeam(teamA, teamB)
	for _, teamSet in pairs(allyTeamSets) do
		if teamSet[teamA] and teamSet[teamB] then
			return true
		end
	end
	return false
end

--------------------------------------------------------------------------------
-- Command dispatcher
--------------------------------------------------------------------------------

-- Maps verb → { cmdID, paramBuilder(cmd) → params, options }
local VERB_MAP = {
	selfd   = { cmdID = CMD.SELFD,   params = function(_) return {} end },
	stop    = { cmdID = CMD.STOP,    params = function(_) return {} end },
	move    = { cmdID = CMD.MOVE,    params = function(c) return { c.x or 0, c.y or 0, c.z or 0 } end },
	attack  = { cmdID = CMD.ATTACK,  params = function(c)
		if c.targetID then return { c.targetID }
		else return { c.x or 0, c.y or 0, c.z or 0 } end
	end },
	patrol  = { cmdID = CMD.PATROL,  params = function(c) return { c.x or 0, c.y or 0, c.z or 0 } end },
	fight   = { cmdID = CMD.FIGHT,   params = function(c) return { c.x or 0, c.y or 0, c.z or 0 } end },
	reclaim = { cmdID = CMD.RECLAIM, params = function(c)
		if c.targetID then return { c.targetID } else return { c.x or 0, c.y or 0, c.z or 0, 100 } end
	end },
	repair  = { cmdID = CMD.REPAIR,  params = function(c) return { c.targetID or 0 } end },
	guard   = { cmdID = CMD.GUARD,   params = function(c) return { c.targetID or 0 } end },
}

-- Resolve a unit-type name (defName string) to its UnitDefs key (number)
local function resolveDefID(unitType)
	if type(unitType) == "number" then return unitType end
	for id, def in pairs(UnitDefs) do
		if def.name == unitType then return id end
	end
	return nil
end

local function dispatchCommand(senderTeamID, cmd)
	local unitID  = cmd.unitID
	local verb    = cmd.cmd

	if type(unitID) ~= "number" or type(verb) ~= "string" then
		spEcho("[AgentBridgeRelay] Invalid command shape: unitID=" ..
			tostring(unitID) .. " cmd=" .. tostring(verb))
		return
	end

	local targetTeamID = spGetUnitTeam(unitID)
	if not targetTeamID then
		spEcho("[AgentBridgeRelay] Unknown unitID: " .. unitID)
		return
	end

	-- Security: only allow control of units in the same alliance as the sender
	if senderTeamID and not sameAllyTeam(senderTeamID, targetTeamID) then
		spEcho("[AgentBridgeRelay] Blocked: sender team " .. senderTeamID ..
			" tried to control enemy team " .. targetTeamID)
		return
	end

	local opts = cmd.shift and { "shift" } or 0

	-- ── Special verbs ──────────────────────────────────────────────────────────

	-- build: order a factory (or constructor) to build a unit type
	-- cmdID = -defID  (negative of the target unit's UnitDef ID)
	if verb == "build" then
		local buildDefID = cmd.defID or resolveDefID(cmd.unitType)
		if not buildDefID then
			spEcho("[AgentBridgeRelay] build: unknown unitType '" .. tostring(cmd.unitType) .. "'")
			return
		end
		-- Get position: use provided x/z or fall back to unit's current position
		local x, y, z = cmd.x, cmd.y, cmd.z
		if not x then
			x, y, z = spGetUnitPosition(unitID)
		end
		-- Always use correct terrain height — y=0 from the LLM causes silent failure
		if x and z then
			y = spGetGroundHeight(x, z)
		end
		local facing = cmd.facing or 0
		-- Stop the unit first so the build order executes immediately
		-- (opts=0 replaces the queue anyway, but stop clears movement orders)
		spGiveOrderToUnit(unitID, CMD.STOP, {}, 0)
		spGiveOrderToUnit(unitID, -buildDefID, { x or 0, y or 0, z or 0, facing }, 0)
		spEcho("[AgentBridgeRelay] build " .. tostring(cmd.unitType) .. " @ (" .. tostring(x) .. "," .. tostring(y) .. "," .. tostring(z) .. ") facing=" .. tostring(facing))
		return
	end

	-- set_rally: set factory rally point
	if verb == "set_rally" then
		spGiveOrderToUnit(unitID, CMD.MOVE, { cmd.x or 0, cmd.y or 0, cmd.z or 0 }, opts)
		return
	end

	-- ── Standard verb map ──────────────────────────────────────────────────────

	local entry = VERB_MAP[verb]
	if not entry then
		spEcho("[AgentBridgeRelay] Unknown verb: " .. verb)
		return
	end

	local ok, params = pcall(entry.params, cmd)
	if not ok then
		spEcho("[AgentBridgeRelay] Param error for verb " .. verb .. ": " .. tostring(params))
		return
	end

	spGiveOrderToUnit(unitID, entry.cmdID, params, opts)

	-- Verbose log (comment out in production)
	-- spEcho("[AgentBridgeRelay] " .. verb .. " → unitID=" .. unitID)
end

--------------------------------------------------------------------------------
-- Message receiver
--------------------------------------------------------------------------------

function gadget:RecvLuaMsg(msg, playerID)
	-- Filter messages not intended for us
	if msg:sub(1, MSG_PREFIX_LEN) ~= MSG_PREFIX then return end

	local jsonBody = msg:sub(MSG_PREFIX_LEN + 1)
	local ok, cmd = pcall(Json.decode, jsonBody)
	if not ok or type(cmd) ~= "table" then
		spEcho("[AgentBridgeRelay] Bad JSON: " .. tostring(jsonBody))
		return
	end

	-- Determine the sender's team for security check
	local senderTeamID = nil
	if playerID then
		local _, _, _, teamID = Spring.GetPlayerInfo(playerID, false)
		senderTeamID = teamID
	end

	dispatchCommand(senderTeamID, cmd)
end
