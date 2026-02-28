-- AgentBridge Relay Gadget  (synced + unsynced)
--
-- Receives commands forwarded by the AgentBridge widget (LuaUI) via
-- Spring.SendLuaRulesMsg, validates them, and issues Spring orders.
--
-- Message formats (all prefixed strings):
--   "AGENTBRIDGE:<json>"           – issue a unit order
--   "AGENTBRIDGE_RESERVE:<json>"   – { "unitIDs": [...] }
--   "AGENTBRIDGE_UNRESERVE:<json>" – { "unitIDs": [...] }
--   "AGENTBRIDGE_WATCH:<json>"     – { "unitID": <n>, "event": "idle"|"finished"|"destroyed"|"from_factory"|"any", "taskID": "<str>" }
--                                    add "unwatch": true to stop watching
--
-- Synced callins: AllowCommand, UnitIdle, UnitFinished, UnitDestroyed, UnitFromFactory
-- Events flow: synced gadget → SendToUnsynced (global) → gadget:agentEvent (unsynced)
--              → Spring.SendLuaUIMsg → widget:RecvLuaMsg → eventsBuffer → GET /events
-- Widget reads WG.agentBridgeEvents via GET /events endpoint.
--
-- Supported order verbs:
--   selfd, stop, move, attack, patrol, fight, reclaim, repair, guard, build, set_rally
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

--------------------------------------------------------------------------------
-- JSON needed in BOTH synced and unsynced scopes
--------------------------------------------------------------------------------
local Json = VFS.Include("common/luaUtilities/json.lua")

--------------------------------------------------------------------------------
-- ══════════════════════════════════════════════════════════════════════════════
-- UNSYNCED PART  (runs in LuaRules unsynced context)
-- ══════════════════════════════════════════════════════════════════════════════
if not gadgetHandler:IsSyncedCode() then

	-- SendToUnsynced("agentEvent", jsonStr) triggers the registered SyncAction.
	-- The callback receives an extra leading argument (ignored with _).
	local function handleAgentEvent(_, jsonStr)
		if type(jsonStr) ~= "string" then return end
		Spring.Echo("[AgentBridge-unsynced] forwarding event to widget: " .. jsonStr)
		Spring.SendLuaUIMsg("AGENTBRIDGE_EVT:" .. jsonStr)
	end

	function gadget:Initialize()
		gadgetHandler:AddSyncAction("agentEvent", handleAgentEvent)
	end

	function gadget:Shutdown()
		gadgetHandler:RemoveSyncAction("agentEvent")
	end

	return  -- nothing else needed for unsynced
end

-- ══════════════════════════════════════════════════════════════════════════════
-- SYNCED PART
-- ══════════════════════════════════════════════════════════════════════════════

--------------------------------------------------------------------------------
-- Message prefixes
--------------------------------------------------------------------------------
local MSG_PREFIX          = "AGENTBRIDGE:"
local MSG_PREFIX_LEN      = #MSG_PREFIX
local MSG_RESERVE         = "AGENTBRIDGE_RESERVE:"
local MSG_RESERVE_LEN     = #MSG_RESERVE
local MSG_UNRESERVE       = "AGENTBRIDGE_UNRESERVE:"
local MSG_UNRESERVE_LEN   = #MSG_UNRESERVE
local MSG_WATCH           = "AGENTBRIDGE_WATCH:"
local MSG_WATCH_LEN       = #MSG_WATCH

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
local spGetGameFrame     = Spring.GetGameFrame

--------------------------------------------------------------------------------
-- Runtime state for reservation / watch / event system
--------------------------------------------------------------------------------

-- Reservation: units blocked from native AI orders while agent is using them
local reservedUnits   = {}   -- { [unitID] = true }

-- Watch list: units the agent wants to be notified about
-- watchedUnits[unitID] = { event = "idle"|"finished"|"destroyed"|"from_factory"|"any", taskID = string }
local watchedUnits    = {}

-- Pending events to flush to unsynced in the next Update tick
local pendingEvents   = {}   -- { {type, unitID, taskID, frame, [newUnitID]} }

-- Set true while dispatchCommand is executing so AllowCommand lets agent orders through
-- (Lua is single-threaded; the flag is accurate because AllowCommand fires *during*
-- the spGiveOrderToUnit call on the same call stack)
local _agentOrderInProgress = false

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
		if teamSet[teamA] and teamSet[teamB] then return true end
	end
	return false
end

--------------------------------------------------------------------------------
-- AllowCommand
-- Block native-AI orders on reserved units BUT let agent orders through.
-- fromLua=true + _agentOrderInProgress=true  →  OUR order → allow.
-- fromLua=true + reserved + NOT our order    →  block native AI.
-- fromLua=false (human click)                →  always allow (player override).
--------------------------------------------------------------------------------
function gadget:AllowCommand(unitID, unitDefID, unitTeam,
                              cmdID, cmdParams, cmdOptions,
                              cmdTag, synced, fromLua)
	if reservedUnits[unitID] and fromLua and not _agentOrderInProgress then
		return false  -- native AI trying to order a reserved unit — block it
	end
	return true
end

--------------------------------------------------------------------------------
-- Unit event callins
--------------------------------------------------------------------------------

local function pushEvent(evt)
	evt.frame = spGetGameFrame()
	pendingEvents[#pendingEvents + 1] = evt
end

local function autoUnreserve(unitID)
	reservedUnits[unitID] = nil
	watchedUnits[unitID]  = nil
end

function gadget:UnitFinished(unitID, unitDefID, unitTeam)
	local w = watchedUnits[unitID]
	if w and (w.event == "finished" or w.event == "any") then
		pushEvent({ type = "finished", unitID = unitID, taskID = w.taskID })
	end
end

function gadget:UnitIdle(unitID, unitDefID, unitTeam)
	local w = watchedUnits[unitID]
	if w and (w.event == "idle" or w.event == "any") then
		pushEvent({ type = "idle", unitID = unitID, taskID = w.taskID })
	end
end

function gadget:UnitDestroyed(unitID, unitDefID, unitTeam,
                               attackerID, attackerDefID, attackerTeam, weaponDefID)
	local w = watchedUnits[unitID]
	if w then
		pushEvent({ type = "destroyed", unitID = unitID, taskID = w.taskID })
	end
	autoUnreserve(unitID)  -- always clean up on death
end

function gadget:UnitFromFactory(unitID, unitDefID, unitTeam,
                                  factID, factDefID, userOrders)
	local w = watchedUnits[factID]
	if w and (w.event == "from_factory" or w.event == "any") then
		pushEvent({
			type      = "from_factory",
			unitID    = factID,
			newUnitID = unitID,
			taskID    = w.taskID,
		})
	end
end

--------------------------------------------------------------------------------
-- GameFrame: flush pending events to the unsynced half each tick
--------------------------------------------------------------------------------
function gadget:GameFrame(n)
	if #pendingEvents == 0 then return end
	for _, evt in ipairs(pendingEvents) do
		local ok, jsonStr = pcall(Json.encode, evt)
		if ok then
			spEcho("[AgentBridgeRelay] SendToUnsynced: " .. jsonStr)
			SendToUnsynced("agentEvent", jsonStr)  -- global, available in synced gadget context
		else
			spEcho("[AgentBridgeRelay] Event encode error: " .. tostring(jsonStr))
		end
	end
	pendingEvents = {}
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

	-- Flag all agent-issued orders so AllowCommand lets them through
	_agentOrderInProgress = true

	-- ── Special verbs ──────────────────────────────────────────────────────────

	if verb == "build" then
		local buildDefID = cmd.defID or resolveDefID(cmd.unitType)
		if not buildDefID then
			spEcho("[AgentBridgeRelay] build: unknown unitType '" .. tostring(cmd.unitType) .. "'")
			_agentOrderInProgress = false
			return
		end
		local x, y, z = cmd.x, cmd.y, cmd.z
		if not x then x, y, z = spGetUnitPosition(unitID) end
		if x and z then y = spGetGroundHeight(x, z) end
		local facing = cmd.facing or 0
		-- opts=0 already replaces the full command queue — no need for a prior
		-- CMD.STOP, which would cause a spurious UnitIdle event and wake event
		-- watchers before the build has actually started.
		spGiveOrderToUnit(unitID, -buildDefID, { x or 0, y or 0, z or 0, facing }, 0)
		spEcho("[AgentBridgeRelay] build " .. tostring(cmd.unitType) ..
			" @ (" .. tostring(x) .. "," .. tostring(y) .. "," .. tostring(z) ..
			") facing=" .. tostring(facing))
		_agentOrderInProgress = false
		return
	end

	if verb == "set_rally" then
		spGiveOrderToUnit(unitID, CMD.MOVE, { cmd.x or 0, cmd.y or 0, cmd.z or 0 }, opts)
		_agentOrderInProgress = false
		return
	end

	-- ── Standard verb map ──────────────────────────────────────────────────────

	local entry = VERB_MAP[verb]
	if not entry then
		spEcho("[AgentBridgeRelay] Unknown verb: " .. verb)
		_agentOrderInProgress = false
		return
	end

	local ok, params = pcall(entry.params, cmd)
	if not ok then
		spEcho("[AgentBridgeRelay] Param error for verb " .. verb .. ": " .. tostring(params))
		_agentOrderInProgress = false
		return
	end

	spGiveOrderToUnit(unitID, entry.cmdID, params, opts)
	_agentOrderInProgress = false
end

--------------------------------------------------------------------------------
-- Reservation / watch helpers
--------------------------------------------------------------------------------
local function handleReserve(cmd)
	for _, uid in ipairs(cmd.unitIDs or {}) do
		if type(uid) == "number" then reservedUnits[uid] = true end
	end
end

local function handleUnreserve(cmd)
	for _, uid in ipairs(cmd.unitIDs or {}) do
		if type(uid) == "number" then reservedUnits[uid] = nil end
	end
end

local function handleWatch(cmd)
	local uid    = cmd.unitID
	local event  = cmd.event  or "any"
	local taskID = cmd.taskID or "default"
	if type(uid) ~= "number" then
		spEcho("[AgentBridgeRelay] watch: invalid unitID " .. tostring(uid))
		return
	end
	watchedUnits[uid] = { event = event, taskID = taskID }
end

local function handleUnwatch(cmd)
	local uid = cmd.unitID
	if type(uid) == "number" then watchedUnits[uid] = nil end
end

--------------------------------------------------------------------------------
-- Message receiver
--------------------------------------------------------------------------------

function gadget:RecvLuaMsg(msg, playerID)
	-- Determine sender team (used for security check on order messages)
	local senderTeamID = nil
	if playerID then
		local _, _, _, teamID = Spring.GetPlayerInfo(playerID, false)
		senderTeamID = teamID
	end

	-- ── Unit order ─────────────────────────────────────────────────────────────
	if msg:sub(1, MSG_PREFIX_LEN) == MSG_PREFIX then
		local jsonBody = msg:sub(MSG_PREFIX_LEN + 1)
		local ok, cmd  = pcall(Json.decode, jsonBody)
		if not ok or type(cmd) ~= "table" then
			spEcho("[AgentBridgeRelay] Bad JSON (order): " .. tostring(jsonBody))
			return
		end
		dispatchCommand(senderTeamID, cmd)
		return
	end

	-- ── Reserve ────────────────────────────────────────────────────────────────
	if msg:sub(1, MSG_RESERVE_LEN) == MSG_RESERVE then
		local ok, cmd = pcall(Json.decode, msg:sub(MSG_RESERVE_LEN + 1))
		if ok and type(cmd) == "table" then handleReserve(cmd) end
		return
	end

	-- ── Unreserve ──────────────────────────────────────────────────────────────
	if msg:sub(1, MSG_UNRESERVE_LEN) == MSG_UNRESERVE then
		local ok, cmd = pcall(Json.decode, msg:sub(MSG_UNRESERVE_LEN + 1))
		if ok and type(cmd) == "table" then handleUnreserve(cmd) end
		return
	end

	-- ── Watch / Unwatch ────────────────────────────────────────────────────────
	if msg:sub(1, MSG_WATCH_LEN) == MSG_WATCH then
		local ok, cmd = pcall(Json.decode, msg:sub(MSG_WATCH_LEN + 1))
		if ok and type(cmd) == "table" then
			if cmd.unwatch then handleUnwatch(cmd) else handleWatch(cmd) end
		end
		return
	end
end
