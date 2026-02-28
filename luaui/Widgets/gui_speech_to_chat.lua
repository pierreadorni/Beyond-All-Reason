local widget = widget ---@type Widget

function widget:GetInfo()
	return {
		name    = "Speech to Chat",
		desc    = "Hold , to record voice and send to ally chat (requires SpeechToTextDaemon.py running)",
		author  = "BAR Speech Widget",
		date    = "2026",
		license = "GNU GPL, v2 or later",
		layer   = 0,
		enabled = true,
	}
end

-- ---------------------------------------------------------------------------
-- Config
-- ---------------------------------------------------------------------------
local HOTKEY = string.byte(",")   -- push-to-talk: hold to record, release to send

-- IPC file paths — resolved at init from Spring's writable data dir
local TRIGGER_FILE
local STOP_FILE
local CANCEL_FILE
local DONE_FILE
local RESULT_FILE
local LOG_FILE

local POLL_INTERVAL = 0.3   -- seconds between result-file checks

-- ---------------------------------------------------------------------------
-- State: "idle" → "recording" (key held) → "processing" (key released) → "idle"
-- ---------------------------------------------------------------------------
local state        = "idle"
local pulseTimer   = 0
local lastPollTime = 0

-- ---------------------------------------------------------------------------
-- Helpers
-- ---------------------------------------------------------------------------
local function writeFile(path, content)
	local f = io.open(path, "w")
	if f then f:write(content) f:close() return true end
	return false
end

local function log(msg)
	Spring.Echo("[SpeechToChat] " .. msg)
	if not LOG_FILE then return end
	local f = io.open(LOG_FILE, "a")
	if f then
		f:write(string.format("[%s] %s\n", os.date("%Y-%m-%d %H:%M:%S"), msg))
		f:close()
	end
end

local function cleanStaleFiles()
	os.remove(TRIGGER_FILE)
	os.remove(STOP_FILE)
	os.remove(CANCEL_FILE)
	os.remove(DONE_FILE)
	os.remove(RESULT_FILE)
end

-- ---------------------------------------------------------------------------
-- Widget callins
-- ---------------------------------------------------------------------------
function widget:Initialize()
	-- Discover Spring's writable data directory so io.open paths match
	local writeDir = ""
	if Spring.GetWriteableDataDir then
		writeDir = Spring.GetWriteableDataDir()
		if writeDir ~= "" and not writeDir:match("[/\\]$") then
			writeDir = writeDir .. "/"
		end
	end

	TRIGGER_FILE = writeDir .. "LuaUI/stt_trigger.flag"
	STOP_FILE    = writeDir .. "LuaUI/stt_stop.flag"
	CANCEL_FILE  = writeDir .. "LuaUI/stt_cancel.flag"
	DONE_FILE    = writeDir .. "LuaUI/stt_done.flag"
	RESULT_FILE  = writeDir .. "LuaUI/stt_result.txt"
	LOG_FILE     = writeDir .. "LuaUI/stt_log.txt"

	Spring.Echo("[SpeechToChat] Write dir : " .. writeDir)
	Spring.Echo("[SpeechToChat] IPC path  : " .. TRIGGER_FILE)
	Spring.Echo("[SpeechToChat] Pass this to the daemon: --ipc-dir \"" .. writeDir .. "LuaUI\"")

	cleanStaleFiles()
	log("Widget initialised. Hold , to record, release to send.")
end

function widget:Shutdown()
	if state ~= "idle" then
		writeFile(CANCEL_FILE, "cancel")
	end
	cleanStaleFiles()
	log("Widget shut down.")
end

function widget:KeyPress(key, mods, isRepeat)
	if isRepeat then return false end
	if key == HOTKEY and state == "idle" then
		os.remove(RESULT_FILE)
		os.remove(DONE_FILE)
		os.remove(STOP_FILE)
		if writeFile(TRIGGER_FILE, "record") then
			log("Recording started.")
		else
			log("ERROR: could not write trigger file: " .. tostring(TRIGGER_FILE))
		end
		state      = "recording"
		pulseTimer = 0
		return true
	end
	return false
end

function widget:KeyRelease(key)
	if key == HOTKEY and state == "recording" then
		writeFile(STOP_FILE, "stop")
		state        = "processing"
		lastPollTime = 0
		log("Key released — processing speech...")
		return true
	end
	return false
end

-- ---------------------------------------------------------------------------
-- Drawing
-- ---------------------------------------------------------------------------
function widget:DrawScreen()
	if state == "idle" then return end

	local vsx, vsy = gl.GetViewSizes()
	local boxW = 280
	local boxH = 44
	local boxX = math.floor((vsx - boxW) / 2)
	local boxY = math.floor(vsy * 0.80)

	local pulse = 0.5 + 0.5 * math.abs(math.sin(pulseTimer * math.pi))

	gl.Blending(true)

	-- Background
	gl.Color(0, 0, 0, 0.75)
	gl.Rect(boxX, boxY, boxX + boxW, boxY + boxH)

	-- Border
	if state == "recording" then
		gl.Color(1, 0.1, 0.1, pulse)
	else
		gl.Color(1, 0.6, 0.0, 0.95)
	end
	gl.Rect(boxX,             boxY,             boxX + boxW,    boxY + 2)
	gl.Rect(boxX,             boxY + boxH - 2,  boxX + boxW,    boxY + boxH)
	gl.Rect(boxX,             boxY,             boxX + 2,       boxY + boxH)
	gl.Rect(boxX + boxW - 2, boxY,             boxX + boxW,    boxY + boxH)

	-- Icon + text
	if state == "recording" then
		gl.Color(1, 0.1, 0.1, pulse)
		gl.Text("REC", boxX + 10, boxY + 12, 13, "o")
		gl.Color(1, 1, 1, 0.95)
		gl.Text("RECORDING — release , to send", boxX + 48, boxY + 12, 14, "o")
	else
		gl.Color(1, 0.75, 0.0, 0.95)
		gl.Text("...", boxX + 10, boxY + 12, 16, "o")
		gl.Color(1, 1, 1, 0.95)
		gl.Text("Processing speech...", boxX + 48, boxY + 12, 14, "o")
	end

	gl.Blending(false)
	gl.Color(1, 1, 1, 1)
end

-- ---------------------------------------------------------------------------
-- Update: poll for result while processing
-- ---------------------------------------------------------------------------
function widget:Update(dt)
	pulseTimer = pulseTimer + dt

	if state ~= "processing" then return end

	lastPollTime = lastPollTime + dt
	if lastPollTime < POLL_INTERVAL then return end
	lastPollTime = 0

	local f = io.open(RESULT_FILE, "r")
	if f then
		local text = f:read("*all")
		f:close()
		os.remove(RESULT_FILE)
		os.remove(DONE_FILE)
		text = text:match("^%s*(.-)%s*$")
		if text and text ~= "" then
			Spring.SendCommands("say " .. text)
			log("Sent to chat: " .. text)
		else
			log("Empty transcription — nothing sent.")
		end
		state = "idle"
		return
	end

	local d = io.open(DONE_FILE, "r")
	if d then
		d:close()
		os.remove(DONE_FILE)
		log("Daemon finished with no transcription.")
		state = "idle"
	end
end