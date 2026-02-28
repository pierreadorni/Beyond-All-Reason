-- AgentBridge Widget
-- Exposes a local HTTP server (port 7654) for external LLM/Python agent control.
--
-- Endpoints:
--   GET  /state        → Full game state (ally teams, units, resources, visible enemies, map info)
--   GET  /defs         → Unit definition catalog (all buildable units with categories)
--   GET  /chat         → Drain the incoming in-game chat buffer (new messages since last poll)
--   POST /chat/send    → Send a message to in-game all-chat
--   POST /command      → Issue a command to a unit (forwarded to synced relay gadget)
--
-- NOTE: JSON is self-contained (no VFS.Include dependency) — works in both
--   the game widget dir (luaui/Widgets/) and the user widget dir (LuaUI/Widgets/).
--
-- REQUIREMENTS:
--   TCPAllowListen=1 in Spring config (or /set TCPAllowListen 1 in game).
--
-- Usage: Enable "AgentBridge" in Widget Manager (F11), then run tools/bar_agent.py

local widget = widget ---@type Widget

function widget:GetInfo()
	return {
		name    = "AgentBridge",
		desc    = "HTTP bridge (localhost:7654) for external LLM/Python agent control",
		author  = "AgentBridge",
		date    = "2026",
		license = "GNU GPL, v2 or later",
		layer   = 0,
		enabled = false,
	}
end

--------------------------------------------------------------------------------
-- Configuration
--------------------------------------------------------------------------------
local SERVER_HOST     = "127.0.0.1"
local SERVER_PORT     = 7654
local MAX_CLIENTS     = 8
local RECV_CHUNK      = 8192
local CHAT_BUFFER_MAX = 100   -- max stored incoming chat lines

--------------------------------------------------------------------------------
-- Minimal JSON encoder / decoder (self-contained, no _G dependency)
-- json.lua from common/ uses `local base = _G` which is nil in the user-widget
-- sandbox (VFS.RAW context), so we embed a lightweight alternative here.
--------------------------------------------------------------------------------

-- Encoder ---------------------------------------------------------------
local function jsonEncode(val, _stack)
	_stack = _stack or {}
	local t = type(val)
	if t == "nil" then
		return "null"
	elseif t == "boolean" then
		return tostring(val)
	elseif t == "number" then
		if val ~= val then return "null" end  -- NaN
		return tostring(val)
	elseif t == "string" then
		return '"' .. val
			:gsub('\\', '\\\\')
			:gsub('"',  '\\"')
			:gsub('\n', '\\n')
			:gsub('\r', '\\r')
			:gsub('\t', '\\t')
			.. '"'
	elseif t == "table" then
		if _stack[val] then return '"<cycle>"' end
		_stack[val] = true
		local parts = {}
		-- Decide array vs object: array if all keys are consecutive integers from 1
		local maxN = 0
		local isArray = true
		for k in pairs(val) do
			local kt = type(k)
			if kt ~= "number" or k ~= math.floor(k) or k < 1 then
				isArray = false; break
			end
			if k > maxN then maxN = k end
		end
		if isArray and maxN ~= #val then isArray = false end
		if isArray then
			for i = 1, maxN do
				parts[i] = jsonEncode(val[i], _stack)
			end
			_stack[val] = nil
			return "[" .. table.concat(parts, ",") .. "]"
		else
			for k, v in pairs(val) do
				if type(k) == "string" or type(k) == "number" then
					parts[#parts + 1] = '"' .. tostring(k) .. '":' .. jsonEncode(v, _stack)
				end
			end
			_stack[val] = nil
			return "{" .. table.concat(parts, ",") .. "}"
		end
	else
		return '"[' .. t .. ']"'
	end
end

-- Decoder ---------------------------------------------------------------
local function jsonDecode(s)
	local pos = 1
	local function skipWS()
		while pos <= #s and s:sub(pos, pos):match("%s") do pos = pos + 1 end
	end
	local parseValue  -- forward declaration
	local function parseString()
		pos = pos + 1  -- skip opening "
		local buf = {}
		while pos <= #s do
			local c = s:sub(pos, pos)
			if c == '"' then pos = pos + 1; return table.concat(buf) end
			if c == '\\' then
				pos = pos + 1
				local e = s:sub(pos, pos)
				if     e == 'n' then buf[#buf+1] = '\n'
				elseif e == 't' then buf[#buf+1] = '\t'
				elseif e == 'r' then buf[#buf+1] = '\r'
				elseif e == '"' then buf[#buf+1] = '"'
				elseif e == '\\' then buf[#buf+1] = '\\'
				elseif e == '/'  then buf[#buf+1] = '/'
				else                  buf[#buf+1] = '\\'; buf[#buf+1] = e
				end
			else
				buf[#buf+1] = c
			end
			pos = pos + 1
		end
		error("Unterminated string at pos " .. pos)
	end
	local function parseObject()
		pos = pos + 1  -- skip {
		local obj = {}
		skipWS()
		if s:sub(pos, pos) == '}' then pos = pos + 1; return obj end
		while true do
			skipWS()
			local key = parseString()
			skipWS()
			pos = pos + 1  -- skip :
			skipWS()
			obj[key] = parseValue()
			skipWS()
			local c = s:sub(pos, pos); pos = pos + 1
			if c == '}' then return obj end
			-- else c == ','
		end
	end
	local function parseArray()
		pos = pos + 1  -- skip [
		local arr = {}
		skipWS()
		if s:sub(pos, pos) == ']' then pos = pos + 1; return arr end
		while true do
			skipWS()
			arr[#arr + 1] = parseValue()
			skipWS()
			local c = s:sub(pos, pos); pos = pos + 1
			if c == ']' then return arr end
			-- else c == ','
		end
	end
	parseValue = function()
		skipWS()
		local c = s:sub(pos, pos)
		if     c == '"' then return parseString()
		elseif c == '{' then return parseObject()
		elseif c == '[' then return parseArray()
		elseif c == 't' then pos = pos + 4; return true
		elseif c == 'f' then pos = pos + 5; return false
		elseif c == 'n' then pos = pos + 4; return nil
		else
			local num = s:match("^-?%d+%.?%d*[eE]?[+-]?%d*", pos)
			if num then pos = pos + #num; return tonumber(num) end
			error("Unexpected character '" .. c .. "' at pos " .. pos)
		end
	end
	return parseValue()
end

-- Unified API (same surface as json.lua: Json.encode / Json.decode)
local Json = { encode = jsonEncode, decode = jsonDecode }

--------------------------------------------------------------------------------
-- Spring API locals
--------------------------------------------------------------------------------
local spEcho             = Spring.Echo
local spGetMyTeamID      = Spring.GetMyTeamID
local spGetMyAllyTeamID  = Spring.GetMyAllyTeamID
local spGetTeamList      = Spring.GetTeamList
local spGetTeamUnits     = Spring.GetTeamUnits
local spGetAllUnits      = Spring.GetAllUnits
local spGetUnitDefID     = Spring.GetUnitDefID
local spGetUnitPosition  = Spring.GetUnitPosition
local spGetUnitHealth    = Spring.GetUnitHealth
local spGetTeamResources = Spring.GetTeamResources
local spGetTeamLuaAI     = Spring.GetTeamLuaAI
local spGetUnitTeam      = Spring.GetUnitTeam
local spGameFrame        = Spring.GetGameFrame
local spSendLuaRulesMsg  = Spring.SendLuaRulesMsg
local spSendCommands     = Spring.SendCommands

local MSG_PREFIX = "AGENTBRIDGE:"

--------------------------------------------------------------------------------
-- Runtime state
--------------------------------------------------------------------------------
local server              = nil
local clients             = {}
local chatBuffer          = {}   -- incoming: { text, frame }
local peekBuffer          = {}   -- last 20 messages ever pushed (never drained)
local outgoingChat        = {}   -- queued messages to send as in-game chat
local defsCatalog         = nil  -- built once on Initialize
local _inAddConsoleLine   = false
local _debugStats         = { addConsoleLineCalls = 0, gotChatMsgCalls = 0, pushChatCalls = 0 }

--------------------------------------------------------------------------------
-- Build unit-def catalog (categories matching ai_simpleai.lua logic)
--------------------------------------------------------------------------------
local function buildDefsCatalog()
	local catalog = {
		commanders   = {},
		factories    = {},
		constructors = {},
		extractors   = {},
		generators   = {},
		converters   = {},
		turrets      = {},
		other        = {},
	}
	local wind = Game.windMax
	for defID, uDef in pairs(UnitDefs) do
		local entry = {
			defID     = defID,
			name      = uDef.name,
			humanName = uDef.humanName or uDef.name,
		}
		if uDef.customParams.iscommander then
			catalog.commanders[#catalog.commanders+1] = entry
		elseif uDef.isFactory and #uDef.buildOptions > 0 then
			local opts = {}
			for _, oid in ipairs(uDef.buildOptions) do
				local od = UnitDefs[oid]
				if od then opts[#opts+1] = { defID = oid, name = od.name } end
			end
			entry.buildOptions = opts
			catalog.factories[#catalog.factories+1] = entry
		elseif uDef.canMove and uDef.isBuilder and #uDef.buildOptions > 0 then
			catalog.constructors[#catalog.constructors+1] = entry
		elseif uDef.extractsMetal > 0 or uDef.customParams.metal_extractor then
			catalog.extractors[#catalog.extractors+1] = entry
		elseif (uDef.energyMake > 19)
		    or (uDef.windGenerator > 0 and wind > 10)
		    or uDef.tidalGenerator > 0
		    or uDef.customParams.solar then
			catalog.generators[#catalog.generators+1] = entry
		elseif uDef.customParams.energyconv_capacity then
			catalog.converters[#catalog.converters+1] = entry
		elseif uDef.isBuilding and #uDef.weapons > 0 then
			catalog.turrets[#catalog.turrets+1] = entry
		else
			catalog.other[#catalog.other+1] = entry
		end
	end
	return catalog
end

--------------------------------------------------------------------------------
-- HTTP helpers
--------------------------------------------------------------------------------

local function buildHttpResponse(statusCode, statusText, body)
	local bodyStr = body or ""
	return table.concat({
		"HTTP/1.1 " .. statusCode .. " " .. statusText,
		"Content-Type: application/json; charset=utf-8",
		"Content-Length: " .. #bodyStr,
		"Access-Control-Allow-Origin: *",
		"Access-Control-Allow-Methods: GET, POST, OPTIONS",
		"Access-Control-Allow-Headers: Content-Type",
		"Connection: close",
		"",
		bodyStr,
	}, "\r\n")
end

local function jsonOk(data)
	return buildHttpResponse(200, "OK", Json.encode(data))
end

local function jsonError(msg, code)
	return buildHttpResponse(code or 400, "Bad Request",
		Json.encode({ error = msg }))
end

--------------------------------------------------------------------------------
-- Game-state builder
--------------------------------------------------------------------------------
local function buildState()
	local myTeamID     = spGetMyTeamID()
	local myAllyTeamID = spGetMyAllyTeamID()
	local frame        = spGameFrame()
	local allyTeamIDs  = spGetTeamList(myAllyTeamID) or {}

	-- Build ally team set for fast lookup
	local allyTeamSet = {}
	for _, tid in ipairs(allyTeamIDs) do allyTeamSet[tid] = true end

	-- ── Ally teams & units ────────────────────────────────────────────────
	local teams = {}
	for _, teamID in ipairs(allyTeamIDs) do
		local luaAI = spGetTeamLuaAI(teamID)
		local isBot = (luaAI ~= nil and luaAI ~= "")

		local metal, metalStorage, _, metalIncome, metalExpense =
			spGetTeamResources(teamID, "metal")
		local energy, energyStorage, _, energyIncome, energyExpense =
			spGetTeamResources(teamID, "energy")

		local units = {}
		for _, unitID in ipairs(spGetTeamUnits(teamID) or {}) do
			local defID = spGetUnitDefID(unitID)
			if defID then
				local uDef = UnitDefs[defID]
				local x, y, z = spGetUnitPosition(unitID)
				local hp, maxHp = spGetUnitHealth(unitID)
				-- Collect build options for factories
				local buildOpts = nil
				if uDef.isFactory and #uDef.buildOptions > 0 then
					buildOpts = {}
					for _, oid in ipairs(uDef.buildOptions) do
						local od = UnitDefs[oid]
						if od then buildOpts[#buildOpts+1] = { defID = oid, name = od.name } end
					end
				end
				units[#units+1] = {
					unitID       = unitID,
					defID        = defID,
					name         = uDef.name,
					humanName    = uDef.humanName or uDef.name,
					isCommander  = (uDef.customParams.iscommander ~= nil),
					isBuilder    = uDef.isBuilder,
					isFactory    = uDef.isFactory,
					canMove      = uDef.canMove,
					x            = x and math.floor(x) or 0,
					y            = y and math.floor(y) or 0,
					z            = z and math.floor(z) or 0,
					health       = hp    and math.floor(hp)    or 0,
					maxHealth    = maxHp and math.floor(maxHp) or 0,
					buildOptions = buildOpts,
				}
			end
		end

		teams[#teams+1] = {
			teamID        = teamID,
			isMyTeam      = (teamID == myTeamID),
			isBot         = isBot,
			luaAI         = luaAI or "",
			metal         = metal        and math.floor(metal)        or 0,
			metalStorage  = metalStorage and math.floor(metalStorage) or 0,
			metalIncome   = metalIncome  and math.floor(metalIncome)  or 0,
			metalExpense  = metalExpense and math.floor(metalExpense) or 0,
			energy        = energy        and math.floor(energy)        or 0,
			energyStorage = energyStorage and math.floor(energyStorage) or 0,
			energyIncome  = energyIncome  and math.floor(energyIncome)  or 0,
			energyExpense = energyExpense and math.floor(energyExpense) or 0,
			units         = units,
		}
	end

	-- ── Visible enemies (units NOT in any ally team) ──────────────────────
	local visibleEnemies = {}
	for _, unitID in ipairs(spGetAllUnits() or {}) do
		local unitTeam = spGetUnitTeam(unitID)
		if unitTeam and not allyTeamSet[unitTeam] then
			local defID = spGetUnitDefID(unitID)
			local x, y, z = spGetUnitPosition(unitID)
			local hp, maxHp = spGetUnitHealth(unitID)
			local entry = {
				unitID = unitID,
				teamID = unitTeam,
				x      = x and math.floor(x) or 0,
				y      = y and math.floor(y) or 0,
				z      = z and math.floor(z) or 0,
			}
			if defID then
				local uDef = UnitDefs[defID]
				entry.defID     = defID
				entry.name      = uDef.name
				entry.humanName = uDef.humanName or uDef.name
				entry.health    = hp    and math.floor(hp)    or 0
				entry.maxHealth = maxHp and math.floor(maxHp) or 0
			else
				-- Radar blip: type unknown, only position
				entry.name      = "unknown(radar)"
				entry.radarOnly = true
			end
			visibleEnemies[#visibleEnemies+1] = entry
		end
	end

	-- ── Map info ─────────────────────────────────────────────────────────
	local mapInfo = {
		name    = Game.mapName or "unknown",
		sizeX   = Game.mapSizeX,
		sizeZ   = Game.mapSizeZ,
		windMin = Game.windMin,
		windMax = Game.windMax,
		tidal   = Game.tidalStrength,
	}

	return {
		frame          = frame,
		myTeamID       = myTeamID,
		myAllyTeamID   = myAllyTeamID,
		teams          = teams,
		visibleEnemies = visibleEnemies,
		mapInfo        = mapInfo,
	}
end

--------------------------------------------------------------------------------
-- Request dispatcher
--------------------------------------------------------------------------------
local function handleRequest(method, path, body)
	if method == "OPTIONS" then
		return buildHttpResponse(204, "No Content", "")
	end

	-- GET /state
	if path == "/state" and method == "GET" then
		local ok, result = pcall(buildState)
		if ok then return jsonOk(result)
		else return jsonError("State error: " .. tostring(result), 500) end

	-- GET /debug  (stats + buffer peek, never drains)
	elseif path == "/debug" and method == "GET" then
		return jsonOk({
			addConsoleLineCalls = _debugStats.addConsoleLineCalls,
			gotChatMsgCalls     = _debugStats.gotChatMsgCalls,
			pushChatCalls       = _debugStats.pushChatCalls,
			bufferSize          = #chatBuffer,
			last20              = peekBuffer,
		})

	-- GET /defs
	elseif path == "/defs" and method == "GET" then
		return jsonOk(defsCatalog or {})

	-- GET /chat  (drain buffer)
	elseif path == "/chat" and method == "GET" then
		local msgs = chatBuffer
		chatBuffer  = {}
		return jsonOk(msgs)

	-- POST /chat/send
	elseif path == "/chat/send" and method == "POST" then
		if not body or body == "" then return jsonError("Empty body") end
		local ok, data = pcall(Json.decode, body)
		if not ok or type(data) ~= "table" then return jsonError("Invalid JSON") end
		local msg = data.message
		if type(msg) ~= "string" or msg == "" then
			return jsonError("Missing 'message' field")
		end
		outgoingChat[#outgoingChat+1] = msg
		return jsonOk({ status = "queued" })

	-- POST /command
	elseif path == "/command" and method == "POST" then
		if not body or body == "" then return jsonError("Empty body") end
		local ok, cmd = pcall(Json.decode, body)
		if not ok or type(cmd) ~= "table" then return jsonError("Invalid JSON body") end
		if not cmd.unitID or not cmd.cmd then
			return jsonError("Missing required fields: unitID, cmd")
		end
		spSendLuaRulesMsg(MSG_PREFIX .. body)
		return jsonOk({ status = "queued", unitID = cmd.unitID, cmd = cmd.cmd })

	else
		return buildHttpResponse(404, "Not Found", Json.encode({ error = "Not found" }))
	end
end

--------------------------------------------------------------------------------
-- Chat capture  (DEBUG MODE)
--
-- AddConsoleLine is the LuaUI callin for all console output incl. chat.
-- GotChatMsg is a separate callin for direct player chat messages.
-- Both are implemented here so we can see in infolog which one fires.
--
-- IMPORTANT: spEcho() must NEVER be called inside AddConsoleLine or pushChat —
-- it writes to the console which re-triggers AddConsoleLine (infinite loop).
-- Debug info is only logged via the /chat drain and GotChatMsg (safe path).
--------------------------------------------------------------------------------

-- Push one entry (ring-buffer, capped at CHAT_BUFFER_MAX)
local function pushChat(text)
	_debugStats.pushChatCalls = _debugStats.pushChatCalls + 1
	if #chatBuffer >= CHAT_BUFFER_MAX then table.remove(chatBuffer, 1) end
	chatBuffer[#chatBuffer+1] = { text = text, frame = spGameFrame() }
	-- Keep a permanent peek buffer (last 20, never drained by /chat)
	if #peekBuffer >= 20 then table.remove(peekBuffer, 1) end
	peekBuffer[#peekBuffer+1] = { text = text, frame = spGameFrame() }
end

-- Primary callin: player chat messages (all/ally/spec channels)
-- spEcho is safe here because GotChatMsg does NOT go through AddConsoleLine
function widget:GotChatMsg(msg, playerID)
	_debugStats.gotChatMsgCalls = _debugStats.gotChatMsgCalls + 1
	if not msg then return end
	local name = playerID and (Spring.GetPlayerInfo(playerID, false) or "?") or "unknown"
	pushChat("[Chat] " .. name .. ": " .. msg)
end

-- Fallback callin: all console/system output (may also include chat)
-- NO spEcho calls allowed here — would cause infinite recursion
function widget:AddConsoleLine(msg, priority)
	if _inAddConsoleLine then return end
	_inAddConsoleLine = true
	_debugStats.addConsoleLineCalls = _debugStats.addConsoleLineCalls + 1
	if msg then
		local clean = msg:match("^%[f=%d+%] (.+)$") or msg
		for line in (clean .. "\n"):gmatch("([^\n]*)\n") do
			if line ~= "" then
				pushChat(line)
			end
		end
	end
	_inAddConsoleLine = false
end

--------------------------------------------------------------------------------
-- Widget lifecycle
--------------------------------------------------------------------------------
function widget:Initialize()
	local listenAllowed = Spring.GetConfigInt("TCPAllowListen", 0)
	if listenAllowed == 0 then
		spEcho("[AgentBridge] WARNING: TCPAllowListen is 0 — run /set TCPAllowListen 1 then reload.")
	end

	-- Build def catalog once
	local ok, result = pcall(buildDefsCatalog)
	if ok then
		defsCatalog = result
		spEcho("[AgentBridge] Def catalog built.")
	else
		spEcho("[AgentBridge] Warning: def catalog failed: " .. tostring(result))
		defsCatalog = {}
	end

	server = socket.tcp()
	server:setoption("reuseaddr", true)
	local bindOk, err = server:bind(SERVER_HOST, SERVER_PORT)
	if not bindOk then
		spEcho("[AgentBridge] Failed to bind " .. SERVER_HOST .. ":" .. SERVER_PORT .. " — " .. tostring(err))
		widgetHandler:RemoveWidget()
		return
	end
	server:listen(MAX_CLIENTS)
	server:settimeout(0)
	spEcho("[AgentBridge] Listening on " .. SERVER_HOST .. ":" .. SERVER_PORT)
end

function widget:Shutdown()
	for _, c in ipairs(clients) do
		pcall(function() c.sock:close() end)
	end
	clients = {}
	if server then
		pcall(function() server:close() end)
		server = nil
	end
	spEcho("[AgentBridge] Server stopped.")
end

function widget:Update(dt)
	if not server then return end

	-- 1. Flush outgoing chat
	for _, msg in ipairs(outgoingChat) do
		-- "say" is the correct Spring command for all-chat (gui_chat.lua uses it)
		-- Strip newlines so the command is not broken
		local clean = msg:gsub("[\r\n]+", " "):match("^%s*(.-)%s*$")
		if clean ~= "" then
			spSendCommands("say " .. clean)
		end
	end
	outgoingChat = {}

	-- 2. Accept new connections
	local newClient, err = server:accept()
	if newClient then
		newClient:settimeout(0)
		clients[#clients + 1] = {
			sock          = newClient,
			buf           = "",
			headersDone   = false,
			method        = "",
			path          = "",
			headers       = {},
			body          = "",
			contentLength = 0,
		}
	end

	-- 3. Process existing connections
	local toRemove = {}
	for i, c in ipairs(clients) do
		local data, recvErr, partial = c.sock:receive(RECV_CHUNK)
		local chunk = data or partial
		if chunk and chunk ~= "" then
			c.buf = c.buf .. chunk
		end

		if recvErr == "closed" then
			toRemove[#toRemove + 1] = i
		else
			-- Parse HTTP headers if not yet done
			if not c.headersDone then
				local headerEnd = c.buf:find("\r\n\r\n", 1, true)
				if headerEnd then
					local headerSection = c.buf:sub(1, headerEnd - 1)
					c.body         = c.buf:sub(headerEnd + 4)
					c.headersDone  = true

					-- Request line
					local requestLine = headerSection:match("^([^\r\n]+)")
					local method, path = (requestLine or ""):match("^(%S+)%s+(%S+)")
					c.method = method or "GET"
					c.path   = path   or "/"

					-- Remaining header lines
					for line in headerSection:gsub("^[^\r\n]+\r\n", ""):gmatch("[^\r\n]+") do
						local k, v = line:match("^([^:]+):%s*(.+)")
						if k then
							c.headers[k:lower()] = v
						end
					end
					c.contentLength = tonumber(c.headers["content-length"]) or 0
				end
			end

			-- Dispatch once we have the full body
			if c.headersDone and #c.body >= c.contentLength then
				local bodyTrimmed = c.body:sub(1, c.contentLength)
				local ok, response = pcall(handleRequest, c.method, c.path, bodyTrimmed)
				if not ok then
					response = jsonError("Internal error: " .. tostring(response), 500)
				end
				c.sock:send(response)
				c.sock:close()
				toRemove[#toRemove + 1] = i
			end
		end
	end

	-- 4. Remove closed/dispatched clients (reverse order to preserve indices)
	for i = #toRemove, 1, -1 do
		table.remove(clients, toRemove[i])
	end
end
