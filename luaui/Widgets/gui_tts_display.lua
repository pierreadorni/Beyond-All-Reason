-- gui_tts_display.lua
-- Displays an "Ally Commander" portrait while TTS audio is playing.
--
-- Communication interface (via WG, called by AgentBridge or any other widget):
--   WG.TTSStart(duration, pose)  – reveal portrait
--                                   pose = "attack" (sprite 1) or "calm" (sprite 2, default)
--                                   duration (seconds) sets auto-timeout
--   WG.TTSStop()                 – hide portrait immediately (fade out)
--   WG.TTSAmplitude(0..1)        – feed current audio amplitude for the shake effect
--
-- The external Python TTS script should:
--   1. POST /tts/start  {"duration": <seconds>, "pose": "calm"|"attack"}  before playing
--   2. POST /tts/amplitude  {"value": <0-1>}  periodically while playing (optional)
--   3. POST /tts/stop                           when audio ends (or on error)

local widget = widget ---@type Widget

function widget:GetInfo()
	return {
		name    = "TTS Display",
		desc    = "Animated commander portrait synced to TTS audio (driven by AgentBridge)",
		author  = "AgentBridge",
		date    = "2026",
		license = "GNU GPL, v2 or later",
		layer   = 10,
		enabled = true,
	}
end

--------------------------------------------------------------------------------
-- Config
--------------------------------------------------------------------------------
local PORTRAIT_W      = 320           -- portrait width in pixels
local PORTRAIT_H      = 320           -- portrait height in pixels
local MARGIN_X        = 24            -- pixels from right screen edge
local MARGIN_Y        = 180            -- pixels from bottom (clears typical chat bar)
local FADE_SPEED      = 5.0           -- alpha units per second for fade in / out
local SHAKE_IMPULSE   = 3.5           -- max pixel impulse per frame at amplitude 1.0
local SHAKE_DECAY     = 12.0          -- exponential decay rate (larger = snappier)
local SHAKE_MAX       = 8.0           -- hard clamp on shake offset (pixels)
local AUTO_TIMEOUT    = 45.0          -- safety: auto-hide after this many seconds
local LABEL_TEXT      = "Ally Commander"
local LABEL_FONT_SIZE = 13

-- Texture paths (VFS-relative; note the typo in the second file name is intentional)
local TEX_ATTACK = "bitmaps/ui/ally_commander_1.png"  -- attack / aggressive pose
local TEX_CALM   = "bitmaps/ui/aly_commander_2.png"   -- calm / at-ease pose

--------------------------------------------------------------------------------
-- State
--------------------------------------------------------------------------------
local active    = false    -- is TTS currently declared as playing?
local alpha     = 0.0      -- current display alpha (0 = fully hidden)
local pose      = "calm"   -- "attack" or "calm" — set by WG.TTSStart
local amplitude = 0.0      -- latest amplitude value from external (0–1)
local autoTimer = 0.0      -- countdown to auto-hide
local shakeX    = 0.0      -- current horizontal shake offset
local shakeY    = 0.0      -- current vertical shake offset

--------------------------------------------------------------------------------
-- Widget lifecycle
--------------------------------------------------------------------------------
function widget:Initialize()
	-- Expose the WG interface so AgentBridge (or anything else) can drive us
	-- pose: "attack" = sprite 1 (ally_commander_1), "calm" = sprite 2 (aly_commander_2)
	WG.TTSStart = function(duration, p)
		active    = true
		pose      = (p == "attack") and "attack" or "calm"
		autoTimer = tonumber(duration) or AUTO_TIMEOUT
		Spring.Echo("[TTSDisplay] TTS started  pose=" .. pose .. "  timeout=" .. autoTimer .. "s")
	end

	WG.TTSStop = function()
		if active then
			Spring.Echo("[TTSDisplay] TTS stopped.")
		end
		active    = false
		amplitude = 0.0
	end

	WG.TTSAmplitude = function(val)
		amplitude = math.max(0.0, math.min(1.0, tonumber(val) or 0.0))
	end

	Spring.Echo("[TTSDisplay] Initialized. attack=" .. TEX_ATTACK .. "  calm=" .. TEX_CALM)
end

function widget:Shutdown()
	WG.TTSStart     = nil
	WG.TTSStop      = nil
	WG.TTSAmplitude = nil
	Spring.Echo("[TTSDisplay] Shutdown.")
end

--------------------------------------------------------------------------------
-- Update: animation, shake, fade, auto-timeout
--------------------------------------------------------------------------------
function widget:Update(dt)
	-- ── Fade in / out ──────────────────────────────────────────────────────
	if active then
		alpha = math.min(1.0, alpha + FADE_SPEED * dt)
	else
		alpha = math.max(0.0, alpha - FADE_SPEED * dt)
	end

	-- Nothing more to compute while fully hidden
	if alpha <= 0.0 then
		shakeX, shakeY = 0.0, 0.0
		return
	end

	-- ── Auto-timeout ───────────────────────────────────────────────────────
	if active then
		autoTimer = autoTimer - dt
		if autoTimer <= 0 then
			Spring.Echo("[TTSDisplay] Auto-timeout reached — hiding portrait.")
			active    = false
			amplitude = 0.0
		end
	end

	-- ── Shake ──────────────────────────────────────────────────────────────
	-- Random impulse weighted by amplitude, then exponential decay
	if amplitude > 0.01 then
		shakeX = shakeX + (math.random() * 2.0 - 1.0) * amplitude * SHAKE_IMPULSE
		shakeY = shakeY + (math.random() * 2.0 - 1.0) * amplitude * SHAKE_IMPULSE
	end
	local decay = math.exp(-SHAKE_DECAY * dt)
	shakeX = math.max(-SHAKE_MAX, math.min(SHAKE_MAX, shakeX * decay))
	shakeY = math.max(-SHAKE_MAX, math.min(SHAKE_MAX, shakeY * decay))
end

--------------------------------------------------------------------------------
-- DrawScreen: render the portrait overlay
--------------------------------------------------------------------------------
function widget:DrawScreen()
	if alpha <= 0.01 then return end

	local vsx, vsy = gl.GetViewSizes()

	-- Compute base position (bottom-right, above chat bar)
	local baseX = vsx - PORTRAIT_W - MARGIN_X
	local baseY = MARGIN_Y
	local x1    = math.floor(baseX + shakeX)
	local y1    = math.floor(baseY + shakeY)
	local x2    = x1 + PORTRAIT_W
	local y2    = y1 + PORTRAIT_H

	gl.Blending(GL.SRC_ALPHA, GL.ONE_MINUS_SRC_ALPHA)

	-- ── Portrait sprite (no background — PNG transparency is preserved) ────
	local tex = (pose == "attack") and TEX_ATTACK or TEX_CALM
	gl.Color(1.0, 1.0, 1.0, alpha)
	gl.Texture(tex)
	gl.TexRect(x1, y1, x2, y2)
	gl.Texture(false)

	-- ── Label ──────────────────────────────────────────────────────────────
	if alpha > 0.3 then
		-- Background strip behind the label
		local labelH = LABEL_FONT_SIZE + 6
		gl.Color(0.0, 0.0, 0.0, alpha * 0.70)
		gl.Rect(x1, y1, x2, y1 + labelH)

		gl.Color(0.70, 1.00, 0.75, alpha)
		gl.Text(LABEL_TEXT, x1 + PORTRAIT_W * 0.5, y1 + 4, LABEL_FONT_SIZE, "oc")
	end

	-- Cleanup
	gl.Blending(false)
	gl.Color(1.0, 1.0, 1.0, 1.0)
end
