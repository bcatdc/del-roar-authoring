#!/usr/bin/env python3
"""
puppet.py - drive a layered .psd puppet from an audio track (plus optional cues).

  python puppet.py puppet.psd --list                         # show layer names in the PSD
  python puppet.py puppet.psd --audio line01.wav             # talking clip  -> out.mp4
  python puppet.py puppet.psd --audio line01.wav --cues cues.json -o line01.mp4
  python puppet.py puppet.psd --idle 8 -o idle.mp4           # no audio, idle loop
  python puppet.py puppet.psd --limits-test                  # sweep every channel to its limits
  add --no-fx to skip dust and flicker; add --png frames/ to any render for a lossless PNG sequence (exact loop frames)

Every clip starts and ends on the same REST_POSE frames, so all clips share an
identical first/last frame and chain into loops.

Requires: Python 3, `pip install psd-tools numpy pillow opencv-python`, and ffmpeg on PATH.
(opencv-python is only needed for the mane warp; without it the mane stays still.)
"""
import argparse, json, math, subprocess, sys
import numpy as np
from PIL import Image
from psd_tools import PSDImage

# ---------------------------------------------------------------------------
# RIG - from the Sep 16 walkthrough. Canvas pixels; +x right, +y down.
# Translate limits (px): left/right/up/down.  Rotate limits (deg): ccw/cw.
# "layer" must match a PSD layer or group name (case-insensitive) - check with --list.
# ---------------------------------------------------------------------------
PARTS = {
    "body":      dict(layer="Body", parent=None, pivot=(552, 1577),        # belly button
                      left=40, right=40, up=0, down=40, ccw=4, cw=4),
    "head":      dict(layer="5 Head, no Jaw", parent="body",
                      left=20, right=20, up=30, down=50, turn=12,    # turn: +/- degrees of 3D yaw
                      depth=0.75),
    "upper_jaw": dict(layer="3 Upper Jaw", parent="head", up=17, down=11, depth=1.0),
    "lower_jaw": dict(layer="4 Lower Jaw", parent="head", up=13, down=70, depth=0.5),
    "pupils":    dict(layer="6 Pupils", parent="head", left=6, right=6, up=4, down=4),
    "blink_l":   dict(layer="1 Left Blink", parent="head", toggle=True),
    "blink_r":   dict(layer="2 Right Blink", parent="head", toggle=True),
    "hand_l":    dict(layer="7 Left Hand", parent="body", pivot=(237, 1446),  # elbow
                      ccw=75, cw=50),
    "hand_r":    dict(layer="8 Right Hand", parent="body", pivot=(887, 1407),
                      ccw=30, cw=30),
}
MANE_MASK_LAYER = "Mane Mask"   # hidden grayscale layer: white = mane tips move most, black = still
# Parallax depth on head turns: 1 = nearest (shifts most), 0 = body plane (no shift).
# Parts without a depth sit at their parent's depth (pupils and blinks ride on the head).
# Order: upper jaw > head > lower jaw > mane > body.
MANE_DEPTH = 0.25               # depth of the white areas of the Mane Mask

# ---------------------------------------------------------------------------
# TUNING - how much of each limit the automatic motion uses (cues can use 100%).
# ---------------------------------------------------------------------------
FPS = 12                  # try 24 for smoother mouth motion (renders take twice as long)
REST = 3                  # rest-pose frames forced at start and end
GATE = 0.12               # ignore audio below this (breaths, room noise)

# --- mouth ---
JAW_OPEN = 1.0            # fraction of lower_jaw.down reached on the widest vowels
JAW_CURVE = 1.6           # >1 keeps ordinary speech mid-open; only big vowels open fully
JAW_ATTACK = 0.6          # per-frame catch-up when opening (1 = instant)
JAW_RELEASE = 0.4         # per-frame catch-up when closing (lower = lazier close)
JAW_LEAD = 0.06           # seconds the mouth moves ahead of the sound, as real mouths do
SIBILANCE_DAMP = 0.7      # 0-1: how much s/sh/f/t sounds keep the mouth nearly closed
JAW_STEPS = 0             # 0 = continuous; 2 or 3 = snap to fixed positions (servo look)
UPPER_JAW_OPEN = 0.1      # fraction of upper_jaw.up at full open (0 = upper jaw never moves)

# --- head ---
HEAD_NOD = 0.25           # fraction of head.down, following smoothed volume
HEAD_LEAD = 0.15          # seconds the nod leads the voice, so it lands on the stressed word
HEAD_JAW_LIFT = 3         # px the head rises on a wide open (jaw pushing against the skull)

# --- head turn (keystone + parallax) ---
TURN_FOCAL = 1300         # px; smaller = stronger keystone on turns
TURN_PARALLAX = 90        # px a depth-1.0 part shifts vs. the body (scaled by sin of the turn)
GLANCE_TURN = 8.0         # degrees the head turns to follow a full sideways glance (0 = off)

# --- mane (needs the Mane Mask layer and opencv-python) ---
MANE_PAD = 16             # px of room around the head layer for the mane to move into
MANE_BREEZE = 2.5         # px of slow drift at the mane tips
MANE_FOLLOW = 0.6         # how much the mane trails head/body motion
MANE_STIFF = 0.35         # spring stiffness per frame (lower = lazier, floatier mane)
MANE_DAMP = 0.5           # velocity kept per frame: LOWER = settles faster, higher = more jello
MANE_TURN_PX = 1.5        # px of head motion the mane "feels" per degree of turn
MANE_MAX = 8              # px cap on mane displacement
MANE_BLUR = 21            # px softening of the painted mask so the warp never tears

# --- eyes ---
PUPIL_RANGE = 0.8         # fraction of pupil limits used by random glances
GLANCE_EVERY = (1.5, 4.0) # seconds between glances
PUPIL_DART = 2            # frames to travel to a new glance position (1 = instant snap)
BLINK_EVERY = (2.0, 5.0)  # seconds between blinks
BLINK_LEN = 0.15          # seconds eyes stay shut

# --- hands ---
HAND_IDLE_DEG = 2.5       # degrees of slow idle drift on each hand (0 = dead still)
HAND_IDLE_PERIOD = 5.0    # approx seconds per drift cycle
HAND_TWITCH = 0.8         # degrees of small secondary motion, so it isn't a pure sine

# --- speech gestures: head turns and hand lifts at the start of phrases ---
GESTURES = True
PHRASE_QUIET = 0.35       # seconds of quiet that separates one phrase from the next
GESTURE_GAP = 1.6         # minimum seconds between gestures
GESTURE_TURN = (4.0, 10.0)    # degrees the head turns at a phrase start (range)
GESTURE_HAND = (6.0, 15.0)    # degrees a hand lifts (range)
GESTURE_HAND_CHANCE = 0.55    # how often a phrase also gets a hand
GESTURE_HOLD = (0.7, 1.6)     # seconds a gesture holds before relaxing (range)
HAND_RISE = 0.6               # seconds a hand takes to lift; at 12 fps anything much
                              # shorter is only a few frames and reads as a jump
HAND_LIFT_SIGN = {"hand_l": -1.0, "hand_r": 1.0}   # flip a sign if that hand moves the wrong way

# --- body ---
SWAY_X = 0.2              # fraction of body.left/right for idle sway
SWAY_DEG = 0.0            # degrees of idle body rotation around the belly (0 = off)
SWAY_PERIOD = 6.0         # approx seconds per sway cycle (snapped to fit the clip)

# --- cues ---
DEFAULT_EASE = "inout"    # linear | inout | overshoot | step   (per-cue "ease" overrides)
OVERSHOOT = 1.7           # overshoot strength; 1.7 swings ~10% past the target and settles

# --- atmosphere (fades to nothing on the rest frames, so loops stay exact) ---
FX_FADE = 0.5             # seconds the atmosphere takes to fade in/out at clip ends
FLICKER = 0.04            # 0 = off; gentle brightness wander (0.04 = up to 4% dimmer)
FLICKER_DIPS = (4.0, 10.0)  # seconds between brief deeper dips, like a filament sputter; None = off
FLICKER_DIP = 0.12        # depth of those dips
FLICKER_WARM = True       # dimming drops blue more than red, like a dimming bulb
DUST = 40                 # motes visible at once; 0 = off
DUST_REGION = None        # (x0, y0, x1, y1) to confine dust to a light shaft; None = whole frame
DUST_FALL = (12, 35)      # px per second
DUST_LIFE = (3.0, 8.0)    # seconds each mote drifts before fading out of the light
DUST_SIZE = (1.5, 4.5)    # radius px
DUST_ALPHA = (0.25, 0.7)
DUST_BEHIND = 0.4         # fraction of motes drawn behind the puppet (in front of the backdrop)
DUST_COLOR = (255, 240, 215)

# Resting pose used for the first/last frames and for silence. Channels not listed rest at 0
# (the PSD as drawn). Remove the jaw entry to rest on the PSD's slightly open mouth.
REST_POSE = {"lower_jaw.y": 0}   # mouth closed; set to the value you checked on the tower


def clamp(part, axis, v):
    p = PARTS[part]
    if axis == "x": return np.clip(v, -p.get("left", 0), p.get("right", 0))
    if axis == "y": return np.clip(v, -p.get("up", 0), p.get("down", 0))
    if axis == "r": return np.clip(v, -p.get("ccw", 0), p.get("cw", 0))   # +r = clockwise
    if axis == "turn": return np.clip(v, -p.get("turn", 0), p.get("turn", 0))  # + = toward screen right
    return np.clip(v, 0, 1)                                                # "on" toggles


def channels():
    for k, p in PARTS.items():
        if p.get("toggle"): yield k, "on"; continue
        if p.get("left") or p.get("right"): yield k, "x"
        if p.get("up") or p.get("down"): yield k, "y"
        if p.get("ccw") or p.get("cw"): yield k, "r"
        if p.get("turn"): yield k, "turn"


def ease(u, kind):
    u = np.asarray(u, dtype=float)
    if kind == "linear": return u
    if kind == "step": return np.ones_like(u)
    if kind == "overshoot":
        s = OVERSHOOT
        return 1 + (s + 1) * (u - 1) ** 3 + s * (u - 1) ** 2
    return 0.5 - 0.5 * np.cos(math.pi * u)                                  # inout


def lead(a, seconds):
    """Shift a track earlier in time."""
    k = int(round(seconds * FPS))
    return a if k <= 0 else np.concatenate([a[k:], np.zeros(min(k, len(a)))])


def soften_steps(tr, frames):
    """Turn instant jumps into short eased moves."""
    if frames < 2: return tr
    out = tr.copy()
    u = ease(np.arange(1, frames + 1) / frames, "inout")
    for i in np.nonzero(np.diff(tr))[0] + 1:
        seg = tr[i - 1] + (tr[i] - tr[i - 1]) * u
        out[i:i + frames] = seg[:len(out[i:i + frames])]
    return out


# ----------------------------- inputs --------------------------------------
def audio_features(path):
    """Per-frame volume (for the head) and mouth-open target (for the jaw).
    The jaw follows voiced energy (150-1500 Hz, where vowels live) and is held
    nearly closed on hissy consonants, instead of flapping on raw loudness."""
    sr = 16000
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
    hop = sr // FPS
    win = 2 * hop
    xp = np.pad(x, (hop // 2, win))
    freqs = np.fft.rfftfreq(win, 1 / sr)
    voiced = (freqs >= 150) & (freqs <= 1500)
    hiss = (freqs >= 3500) & (freqs <= 7500)
    w = np.hanning(win)
    rms, low, high = [], [], []
    for i in range(0, len(x), hop):
        rms.append(np.sqrt(np.mean(x[i:i + hop] ** 2)))
        spec = np.abs(np.fft.rfft(xp[i:i + win] * w)) ** 2
        low.append(spec[voiced].sum()); high.append(spec[hiss].sum())
    rms, low, high = map(np.array, (rms, low, high))
    norm = lambda a: np.clip(a / (np.percentile(a, 95) + 1e-12), 0, 1)
    vol = norm(rms)
    sib = high / (low + high + 1e-12)
    target = norm(np.sqrt(low)) ** JAW_CURVE * (1 - SIBILANCE_DAMP * sib)
    target[vol < GATE] = 0
    vol[vol < GATE] = 0
    return vol, np.clip(target, 0, 1)


def jaw_follow(target):
    """Lead, then fast-open / slow-close smoothing, optional snapping."""
    target = lead(target, JAW_LEAD)
    out = np.zeros_like(target)
    cur = 0.0
    for i, v in enumerate(target):
        cur += (v - cur) * (JAW_ATTACK if v > cur else JAW_RELEASE)
        out[i] = cur
    if JAW_STEPS:
        out = np.round(out * JAW_STEPS) / JAW_STEPS
    return out


def cue_track(cues, name, n):
    """Cues: {"t": sec, "ch": "hand_l.r", "v": -40, "dur": 0.3, "ease": "overshoot"}
    Movement channels move from the previous cue value to v over dur, then hold.
    Toggle channels ("blink_l.on") are switched on for dur seconds."""
    tr = np.zeros(n)
    mine = sorted((c for c in cues if c["ch"] == name), key=lambda c: c["t"])
    cur = 0.0
    for c in mine:
        s = REST + int(round(c["t"] * FPS))
        d = max(1, int(round(c.get("dur", 0) * FPS)))
        if s >= n: continue
        if name.endswith(".on"):
            tr[s:s + d] = c["v"]
        else:
            e = min(n, s + d)
            u = ease(np.arange(1, d + 1) / d, c.get("ease", DEFAULT_EASE))
            tr[s:e] = (cur + (c["v"] - cur) * u)[: e - s]
            tr[e:] = c["v"]
            cur = c["v"]
    return tr


def _phrase_starts(vol, n_active):
    """Frames where speech resumes after a pause, plus strong peaks inside long
    phrases, spaced at least GESTURE_GAP apart."""
    k = max(2, FPS // 4)
    sm = np.convolve(vol, np.ones(k) / k, mode="same")
    speaking = sm > 0.15
    quiet = int(PHRASE_QUIET * FPS)
    gap = int(GESTURE_GAP * FPS)
    starts, last = [], -gap
    for i in range(n_active):
        if not speaking[i] or i - last < gap:
            continue
        fresh = i >= quiet and not speaking[max(0, i - quiet):i].any()
        peak = sm[i] > 0.8 and sm[i] >= sm[max(0, i - 3):i + 4].max()
        if fresh or (i == 0) or (peak and i - last >= gap * 1.5):
            starts.append(i)
            last = i
    return starts


def _add_gestures(t, vol, n_active, rng):
    """Head turns and hand lifts timed to the phrasing of the voice. Every move
    eases in, holds, and relaxes, and all of it fades out before the clip ends so
    the rest frames still match."""
    turn = np.zeros(n_active)
    hands = {h: np.zeros(n_active) for h in ("hand_l", "hand_r")}

    def move(track, at, amount, rise, hold, fall, kind="inout"):
        r, h, f = max(1, int(rise * FPS)), int(hold * FPS), max(1, int(fall * FPS))
        up = ease(np.arange(1, r + 1) / r, kind)
        down = 1 - ease(np.arange(1, f + 1) / f, "inout")
        seg = np.concatenate([up, np.ones(h), down]) * amount
        end = min(n_active, at + len(seg))
        track[at:end] += seg[:end - at]

    side = 1.0 if rng.random() < 0.5 else -1.0
    for i in _phrase_starts(vol, n_active):
        hold = rng.uniform(*GESTURE_HOLD)
        side = -side if rng.random() < 0.7 else side          # mostly alternate
        move(turn, i, side * rng.uniform(*GESTURE_TURN), 0.35, hold, 0.55)
        if rng.random() < GESTURE_HAND_CHANCE:
            h = "hand_l" if rng.random() < 0.5 else "hand_r"
            move(hands[h], i + int(0.1 * FPS), HAND_LIFT_SIGN[h] * rng.uniform(*GESTURE_HAND),
                 HAND_RISE, hold * 0.8, 0.7, "inout")

    # relax everything over the last half second of speech
    tail = min(n_active, int(0.5 * FPS))
    fade = np.ones(n_active)
    fade[n_active - tail:] = 1 - ease(np.arange(1, tail + 1) / tail, "inout")
    act = slice(REST, REST + n_active)
    t["head.turn"][act] += turn * fade
    for h, tr in hands.items():
        if f"{h}.r" in t:
            t[f"{h}.r"][act] += tr * fade


def build_tracks(n_active, vol, open_target, cues, rng):
    n = n_active + 2 * REST
    act = slice(REST, REST + n_active)
    t = {f"{k}.{a}": np.full(n, float(REST_POSE.get(f"{k}.{a}", 0))) for k, a in channels()}

    if vol is not None:
        o = jaw_follow(open_target[:n_active])
        closed = REST_POSE.get("lower_jaw.y", 0)
        t["lower_jaw.y"][act] = closed + o * (PARTS["lower_jaw"]["down"] * JAW_OPEN - closed)
        t["upper_jaw.y"][act] = -o * PARTS["upper_jaw"]["up"] * UPPER_JAW_OPEN
        sm = np.convolve(lead(vol[:n_active], HEAD_LEAD), np.ones(6) / 6, mode="same")
        t["head.y"][act] = sm * PARTS["head"]["down"] * HEAD_NOD - o * HEAD_JAW_LIFT

    if vol is not None and GESTURES:
        _add_gestures(t, vol[:n_active], n_active, rng)

    # glances: holds with a short eased dart between positions
    i = 0
    while i < n_active:
        hold = int(rng.uniform(*GLANCE_EVERY) * FPS)
        p = PARTS["pupils"]
        gx = rng.uniform(-p["left"], p["right"]) * PUPIL_RANGE
        gy = rng.uniform(-p["up"], p["down"]) * PUPIL_RANGE
        if rng.random() < 0.35: gx = gy = 0          # look back to center often
        t["pupils.x"][REST + i:REST + i + hold] = round(gx)
        t["pupils.y"][REST + i:REST + i + hold] = round(gy)
        i += hold

    # blinks: both eyes together, kept clear of the rest frames
    blen = max(2, int(round(BLINK_LEN * FPS)))
    i = int(rng.uniform(*BLINK_EVERY) * FPS)
    while i + blen < n_active - FPS // 2:
        t["blink_l.on"][REST + i:REST + i + blen] = 1
        t["blink_r.on"][REST + i:REST + i + blen] = 1
        i += blen + int(rng.uniform(*BLINK_EVERY) * FPS)

    # idle sway: whole cycles over the active span, so it returns to zero
    cycles = max(1, round(n_active / FPS / SWAY_PERIOD))
    ph = np.linspace(0, 2 * math.pi * cycles, n_active)
    # hands drift on their own slow cycle, out of phase with each other and the
    # body, easing to nothing at the ends so the rest frames still match
    ease_ends = np.sin(np.linspace(0, math.pi, n_active))
    hand_cycles = max(1, round(n_active / FPS / HAND_IDLE_PERIOD))
    hph = np.linspace(0, 2 * math.pi * hand_cycles, n_active)
    for hand, offset, sign in (("hand_l", 0.0, 1.0), ("hand_r", 2.1, -1.0)):
        if f"{hand}.r" in t and HAND_IDLE_DEG:
            drift = np.sin(hph + offset) * HAND_IDLE_DEG
            twitch = np.sin(hph * 3 + offset * 1.7) * HAND_TWITCH
            t[f"{hand}.r"][act] += sign * (drift + twitch) * ease_ends   # adds to any gestures

    t["body.x"][act] = np.sin(ph) * PARTS["body"]["right"] * SWAY_X
    t["body.r"][act] = np.sin(ph + 0.6) * SWAY_DEG * np.sin(np.linspace(0, math.pi, n_active))

    known = set(t)
    for c in cues:
        if c["ch"] not in known:
            sys.exit(f"Cue channel '{c['ch']}' unknown. Valid: {', '.join(sorted(known))}")

    # let automatic motion settle: pupils recenter in the last half second, nod eases out
    t["pupils.x"][REST + n_active - FPS // 2:] = 0
    t["pupils.y"][REST + n_active - FPS // 2:] = 0
    t["pupils.x"] = soften_steps(t["pupils.x"], PUPIL_DART)
    t["pupils.y"] = soften_steps(t["pupils.y"], PUPIL_DART)
    if GLANCE_TURN:
        k = max(2, FPS // 2)                       # head follows the eyes a beat late
        look = np.convolve(t["pupils.x"] / PARTS["pupils"]["right"], np.ones(k) / k, mode="full")[:n]
        t["head.turn"] = t["head.turn"] + look * GLANCE_TURN
    fade = min(4, n_active)
    t["head.y"][REST + n_active - fade:REST + n_active] *= np.linspace(1, 0, fade)

    for name in t:
        part, axis = name.split(".")
        ct = cue_track(cues, name, n)
        if axis != "on" and abs(ct[REST + n_active - 1]) > 1e-6:
            print(f"  warning: cue leaves {name} at {ct[REST + n_active - 1]:g} when the clip "
                  f"ends - it will snap to neutral. Add a cue back to 0.")
        tr = clamp(part, axis, t[name] + ct)
        rv = clamp(part, axis, REST_POSE.get(name, 0))
        tr[:REST] = rv; tr[-REST:] = rv
        t[name] = tr
    return t, n



def limits_test_tracks():
    """Each channel eases 0 -> min -> 0 -> max -> 0, one at a time."""
    seg = []
    for part, axis in channels():
        p = PARTS[part]
        lo, hi = {"x": (-p.get("left", 0), p.get("right", 0)),
                  "y": (-p.get("up", 0), p.get("down", 0)),
                  "r": (-p.get("ccw", 0), p.get("cw", 0)),
                  "turn": (-p.get("turn", 0), p.get("turn", 0)),
                  "on": (0, 1)}[axis]
        ease = lambda a, b: a + (b - a) * (0.5 - 0.5 * np.cos(np.linspace(0, math.pi, FPS)))
        hold = lambda v: np.full(FPS // 2, v)
        seg.append((f"{part}.{axis}", np.concatenate(
            [ease(0, lo), hold(lo), ease(lo, 0), ease(0, hi), hold(hi), ease(hi, 0)])))
    n = REST * 2 + sum(len(s) for _, s in seg)
    t = {f"{k}.{a}": np.zeros(n) for k, a in channels()}
    i = REST
    for name, s in seg:
        print(f"  {i / FPS:6.1f}s  {name}")
        t[name][i:i + len(s)] = s
        i += len(s)
    return t, n


# ----------------------------- atmosphere ----------------------------------
def fade_env(n):
    """1 during the clip, easing to 0 on the rest frames."""
    env = np.zeros(n)
    env[REST:n - REST] = 1
    f = min(max(1, int(FX_FADE * FPS)), max(1, (n - 2 * REST) // 2))
    ramp = ease(np.arange(1, f + 1) / f, "inout")
    env[REST:REST + f] = ramp
    env[n - REST - f:n - REST] = ramp[::-1]
    return env


def make_mane(tracks, n):
    """Spring-lagged follow-through: how far the mane trails the head each frame."""
    env = fade_env(n)
    g = lambda k: tracks.get(k, np.zeros(n))
    targets = (g("head.x") + g("body.x") + g("head.turn") * MANE_TURN_PX,
               g("head.y") + g("body.y"))
    lags = []
    for tgt in targets:
        p = v = 0.0
        lag = np.zeros(n)
        for i in range(n):
            v = (v + (tgt[i] - p) * MANE_STIFF) * MANE_DAMP
            p += v
            lag[i] = p - tgt[i]
        lags.append(np.clip(lag * MANE_FOLLOW, -MANE_MAX, MANE_MAX) * env)
    return dict(env=env, fx=lags[0], fy=lags[1])


def make_fx(n, size, rng):
    W, H = size
    env = fade_env(n)

    dim = np.zeros(n)
    if FLICKER:
        k = max(2, FPS // 3)
        noise = np.convolve(rng.normal(0, 1, n), np.ones(k) / k, mode="same")
        noise = (noise - noise.min()) / (np.ptp(noise) + 1e-9)          # 0..1
        dim += FLICKER * noise
        if FLICKER_DIPS:
            i = REST + int(rng.uniform(*FLICKER_DIPS) * FPS)
            while i < n - REST:
                L = int(rng.integers(2, 5))
                dim[i:i + L] += (FLICKER_DIP * np.sin(np.linspace(0, math.pi, L + 2)[1:-1]))[:len(dim[i:i + L])]
                i += int(rng.uniform(*FLICKER_DIPS) * FPS)
    bright = 1 - np.clip(dim, 0, 0.9) * env          # only ever dims, so rest frames are untouched

    x0, y0, x1, y1 = DUST_REGION or (0, 0, W, H)
    motes = []
    if DUST:
        mean_life = sum(DUST_LIFE) / 2
        count = int(DUST * (n / FPS + DUST_LIFE[1]) / mean_life)
        for _ in range(count):
            motes.append(dict(
                f0=rng.uniform(-DUST_LIFE[1] * FPS, n), life=rng.uniform(*DUST_LIFE) * FPS,
                x=rng.uniform(x0, x1), y=rng.uniform(y0, y1), vy=rng.uniform(*DUST_FALL) / FPS,
                amp=rng.uniform(2, 8), freq=rng.uniform(0.1, 0.4) / FPS, ph=rng.uniform(0, 6.28),
                r=rng.uniform(*DUST_SIZE), a=rng.uniform(*DUST_ALPHA),
                behind=rng.random() < DUST_BEHIND))
    return dict(bright=bright, env=env, motes=motes, region=(x0, y0, x1, y1))


def draw_dust(frame, fx, i, behind):
    if not fx["motes"] or fx["env"][i] <= 0: return
    from PIL import ImageDraw, ImageFilter
    x0, y0, x1, y1 = fx["region"]
    layer = Image.new("RGBA", (x1 - x0, y1 - y0))
    d = ImageDraw.Draw(layer)
    drew = False
    for m in fx["motes"]:
        if m["behind"] != behind: continue
        age = i - m["f0"]
        if not 0 <= age <= m["life"]: continue
        a = m["a"] * math.sin(math.pi * age / m["life"]) * fx["env"][i]
        px = m["x"] + m["amp"] * math.sin(2 * math.pi * m["freq"] * age + m["ph"]) - x0
        py = m["y"] + m["vy"] * age - y0
        if a < 0.01 or not (0 <= px < x1 - x0 and 0 <= py < y1 - y0): continue
        r = m["r"]
        d.ellipse([px - r, py - r, px + r, py + r], fill=DUST_COLOR + (int(255 * a),))
        drew = True
    if drew:
        frame.alpha_composite(layer.filter(ImageFilter.GaussianBlur(1.2 if behind else 0.8)), dest=(x0, y0))


def apply_flicker(frame, b):
    if b >= 1 - 1e-6: return frame
    arr = np.asarray(frame).astype(np.float32)
    loss = 1 - b
    gains = (1 - loss, 1 - loss * 1.25, 1 - loss * 1.6) if FLICKER_WARM else (b, b, b)
    for c, g in enumerate(gains):
        arr[..., c] *= max(g, 0)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


# ----------------------------- rendering ------------------------------------
def T(x, y): return np.array([[1, 0, x], [0, 1, y], [0, 0, 1.0]])


def R(deg, p):  # clockwise on screen (y down), about pivot p
    a = math.radians(deg); c, s = math.cos(a), math.sin(a); px, py = p
    return np.array([[c, -s, px - c * px + s * py], [s, c, py - s * px - c * py], [0, 0, 1.0]])


def turn_matrix(deg, center):
    """Perspective yaw of a flat card about its vertical center line (+ = toward screen right)."""
    if abs(deg) < 1e-6: return np.eye(3)
    a = math.radians(deg); cx, cy = center; f = TURN_FOCAL
    K = np.array([[math.cos(a), 0, 0], [0, 1, 0], [math.sin(a) / f, 0, 1.0]])
    return T(cx, cy) @ K @ T(-cx, -cy)


class Mane:
    """Warps the head layer: mane areas drift, trail motion, and sit at MANE_DEPTH for parallax."""
    def __init__(self, head_img, l, t, mask_lyr, scale=1.0):
        import cv2
        self.cv2 = cv2
        pad = MANE_PAD
        w, h = head_img.size
        big = Image.new("RGBa", (w + 2 * pad, h + 2 * pad))
        big.paste(head_img, (pad, pad))
        self.img, self.l, self.t = big, l - pad, t - pad
        self.arr = np.asarray(big).copy()
        m = mask_lyr.topil().convert("RGBA")
        if scale != 1.0:
            m = m.resize((max(1, int(m.width * scale)), max(1, int(m.height * scale))),
                         Image.LANCZOS)
        gray = np.asarray(m.convert("L"), np.float32) / 255.0 * (np.asarray(m)[..., 3] / 255.0)
        mask = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
        ox, oy = int(mask_lyr.left * scale) - self.l, int(mask_lyr.top * scale) - self.t
        x0, y0 = max(ox, 0), max(oy, 0)
        x1, y1 = min(ox + gray.shape[1], mask.shape[1]), min(oy + gray.shape[0], mask.shape[0])
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = gray[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        k = MANE_BLUR | 1
        self.mask = cv2.GaussianBlur(mask, (k, k), 0)
        self.gx, self.gy = np.meshgrid(np.arange(mask.shape[1], dtype=np.float32),
                                       np.arange(mask.shape[0], dtype=np.float32))
        self.ys = np.arange(mask.shape[0], dtype=np.float32)
        self.xs = np.arange(mask.shape[1], dtype=np.float32)

    def frame(self, i, mane, shift):
        e = mane["env"][i]
        if e <= 0 and abs(shift) < 1e-3 and mane["fx"][i] == 0 and mane["fy"][i] == 0:
            return self.img
        tt = i / FPS
        bx = MANE_BREEZE * np.sin(2 * math.pi * (tt / 4.1 + self.ys / 300.0))[:, None]
        by = 0.5 * MANE_BREEZE * np.sin(2 * math.pi * (tt / 5.3 + self.xs / 350.0) + 1.0)[None, :]
        dx = (self.mask * (e * bx + mane["fx"][i] + shift)).astype(np.float32)
        dy = (self.mask * (e * by + mane["fy"][i])).astype(np.float32)
        out = self.cv2.remap(self.arr, self.gx - dx, self.gy - dy, self.cv2.INTER_LINEAR,
                             borderMode=self.cv2.BORDER_CONSTANT, borderValue=0)
        return Image.frombytes("RGBa", self.img.size, out.tobytes())


def scale_rig(s):
    """Shrink every pixel-valued setting so a small render behaves like a big one."""
    global TURN_PARALLAX, TURN_FOCAL, HEAD_JAW_LIFT
    global MANE_PAD, MANE_BREEZE, MANE_MAX, MANE_BLUR, MANE_TURN_PX
    for part in PARTS.values():
        for key in ("left", "right", "up", "down"):
            if key in part:
                part[key] = part[key] * s
        if part.get("pivot"):
            part["pivot"] = tuple(v * s for v in part["pivot"])
    for key in list(REST_POSE):
        REST_POSE[key] = REST_POSE[key] * s
    TURN_PARALLAX *= s
    TURN_FOCAL *= s
    HEAD_JAW_LIFT *= s
    MANE_PAD = max(4, int(MANE_PAD * s))
    MANE_BREEZE *= s
    MANE_MAX *= s
    MANE_TURN_PX *= s
    MANE_BLUR = max(3, int(MANE_BLUR * s) | 1)


def report(target):
    """Where the mouth opens and closes, as text, so audio timing can be checked
    before committing to a full render."""
    o = jaw_follow(target)
    total = len(o) / FPS
    print(f"{total:.1f}s of audio at {FPS} fps\n")
    print("    start      end   length  peak")
    start, spans = None, []
    for i, is_open in enumerate(list(o > 0.08) + [False]):
        if is_open and start is None:
            start = i
        elif not is_open and start is not None:
            spans.append((start / FPS, i / FPS, float(o[start:i].max())))
            start = None
    prev_end = 0.0
    for s, e, peak in spans:
        if s - prev_end >= 0.35:
            print(f"{prev_end:9.2f} {s:8.2f} {s - prev_end:8.2f}   -- silence --")
        print(f"{s:9.2f} {e:8.2f} {e - s:8.2f}  {peak:4.2f}")
        prev_end = e
    tail = total - prev_end
    if tail >= 0.35:
        print(f"{prev_end:9.2f} {total:8.2f} {tail:8.2f}   -- silence --")
    lead = spans[0][0] if spans else 0.0
    print(f"\nmouth opens {len(spans)} times | {lead:.2f}s of quiet at the start, "
          f"{tail:.2f}s at the end")
    if spans and (lead < 0.2 or tail < 0.2):
        print("WARNING: speech runs right up to an edge - the cut will clip the jaw")


def load_psd(path, scale=1.0):
    psd = PSDImage.open(path)
    by_name = {}
    for lyr in psd.descendants():
        by_name.setdefault(lyr.name.strip().lower(), lyr)
    mapped, used = {}, set()
    for k, p in PARTS.items():
        lyr = by_name.get(p["layer"].lower())
        if lyr is None:
            sys.exit(f"No layer named '{p['layer']}' for part '{k}'. Run with --list and fix PARTS.")
        mapped[id(lyr)] = k
        used.add(id(lyr))
        if lyr.is_group():
            used.update(id(d) for d in lyr.descendants())

    # draw list in PSD stack order (bottom to top); unmapped visible layers are static
    draw = []
    mask_lyr = by_name.get(MANE_MASK_LAYER.lower())
    for lyr in psd.descendants():
        if lyr is mask_lyr: continue
        inside_mapped_group = id(lyr) in used and id(lyr) not in mapped
        if inside_mapped_group or lyr.is_group():
            if id(lyr) not in mapped: continue
        part = mapped.get(id(lyr))
        if part is None and not lyr.is_visible(): continue
        if lyr.is_group():
            img = lyr.composite(force=True, layer_filter=lambda l: True)
        else:
            img = lyr.topil()   # raw pixels, ignores hidden flag, blend modes and effects
        if img is None: continue
        img = img.convert("RGBA")
        if scale != 1.0:
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                             Image.LANCZOS)
        img = img.convert("RGBa")                      # premultiplied: no dark edge fringes
        draw.append((part, img, int(lyr.left * scale), int(lyr.top * scale)))
        if part is None: print(f"  static layer: {lyr.name}")
        if part == "head":
            PARTS["head"]["center"] = ((lyr.left + lyr.width / 2) * scale,
                                       (lyr.top + lyr.height / 2) * scale)

    mane = None
    if mask_lyr is not None:
        try:
            idx = next(j for j, d in enumerate(draw) if d[0] == "head")
            _, img, l, t = draw[idx]
            mane = Mane(img, l, t, mask_lyr, scale)
            draw[idx] = ("head", mane.img, mane.l, mane.t)
            print("  mane mask found: mane warp on, mane sits behind the face on turns")
        except ImportError:
            print("  mane mask found, but opencv-python isn't installed: pip install opencv-python")
    else:
        print(f"  no '{MANE_MASK_LAYER}' layer: mane stays still and moves with the face on turns")
    canvas = (max(1, int(psd.width * scale)), max(1, int(psd.height * scale)))
    return canvas, draw, mane


def place(frame, img, l, t, M):
    W, H = frame.size
    lin = M[:2, :2]
    if np.allclose(lin, np.eye(2)) and np.allclose(M[:2, 2], np.round(M[:2, 2])):
        src, ox, oy = img, l + int(round(M[0, 2])), t + int(round(M[1, 2]))
    else:
        w, h = img.size
        c = M @ np.array([[l, l + w, l, l + w], [t, t, t + h, t + h], [1, 1, 1, 1.0]])
        c = c[:2] / c[2]
        x0, y0 = np.floor(c.min(axis=1)).astype(int)
        x1, y1 = np.ceil(c.max(axis=1)).astype(int)
        A = T(-l, -t) @ np.linalg.inv(M) @ T(x0, y0)
        A = A / A[2, 2]
        if np.allclose(A[2, :2], 0):
            src = img.transform((x1 - x0, y1 - y0), Image.AFFINE, tuple(A[:2].flatten()),
                                Image.BICUBIC)
        else:
            src = img.transform((x1 - x0, y1 - y0), Image.PERSPECTIVE, tuple(A.flatten()[:8]),
                                Image.BICUBIC)
        ox, oy = x0, y0
    cx0, cy0 = max(ox, 0), max(oy, 0)
    cx1, cy1 = min(ox + src.width, W), min(oy + src.height, H)
    if cx1 <= cx0 or cy1 <= cy0: return
    frame.alpha_composite(src.convert("RGBA"), dest=(cx0, cy0),
                          source=(cx0 - ox, cy0 - oy, cx1 - ox, cy1 - oy))


def render(size, draw, tracks, n, audio, out, png_dir=None, fx=None, mane_rig=None):
    W, H = size
    if png_dir:
        import os; os.makedirs(png_dir, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if audio: cmd += ["-itsoffset", str(REST / FPS), "-i", audio, "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def val(k, a, i): return tracks.get(f"{k}.{a}", np.zeros(n))[i]

    mane = make_mane(tracks, n) if mane_rig else None

    def depth_of(k):
        p = PARTS[k]
        if "depth" in p: return p["depth"]
        return depth_of(p["parent"]) if p["parent"] else 0.0
    for i in range(n):
        world = {}
        turn_shift = math.sin(math.radians(val("head", "turn", i))) * TURN_PARALLAX
        def M_of(k):
            if k in world: return world[k]
            p = PARTS[k]
            M = M_of(p["parent"]) if p["parent"] else np.eye(3)
            rel = depth_of(k) - (depth_of(p["parent"]) if p["parent"] else 0.0)
            L = T(val(k, "x", i) + turn_shift * rel, val(k, "y", i))
            if p.get("pivot") is not None:
                L = L @ R(val(k, "r", i), p["pivot"])
            if p.get("turn"):
                L = L @ turn_matrix(val(k, "turn", i), p["center"])
            world[k] = M @ L
            return world[k]

        frame = Image.new("RGBA", (W, H))
        behind_done = fx is None
        for part, img, l, t in draw:
            if part is None:
                place(frame, img, l, t, np.eye(3)); continue
            if not behind_done:
                draw_dust(frame, fx, i, behind=True); behind_done = True
            if PARTS[part].get("toggle") and val(part, "on", i) < 0.5:
                continue
            if part == "head" and mane_rig:
                img = mane_rig.frame(i, mane, turn_shift * (MANE_DEPTH - depth_of("head")))
            place(frame, img, l, t, M_of(part))
        if fx is not None:
            draw_dust(frame, fx, i, behind=False)
            frame = apply_flicker(frame, fx["bright"][i])
        ff.stdin.write(frame.tobytes())
        if i == 0: frame.save(out.rsplit(".", 1)[0] + "_rest.png")
        if png_dir: frame.save(f"{png_dir}/frame_{i:05d}.png")
        if i % (FPS * 5) == 0: print(f"  frame {i}/{n}", end="\r")
    ff.stdin.close(); ff.wait()
    print(f"  wrote {out} ({n} frames, {n / FPS:.1f}s) and rest-frame PNG")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("psd")
    ap.add_argument("--audio")
    ap.add_argument("--cues", help="JSON list of cues")
    ap.add_argument("--idle", type=float, metavar="SECONDS", help="render an idle clip with no audio")
    ap.add_argument("--limits-test", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--seed", type=int, default=1, help="same seed = same blinks/glances")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="render smaller and faster for checking timing, e.g. 0.3")
    ap.add_argument("--report", action="store_true",
                    help="print where the mouth opens and closes, and render nothing")
    ap.add_argument("-o", "--out", default="out.mp4")
    ap.add_argument("--png", metavar="DIR", help="also write a lossless PNG frame sequence")
    ap.add_argument("--no-fx", action="store_true", help="skip dust and flicker")
    a = ap.parse_args()

    if a.list:
        for lyr in PSDImage.open(a.psd).descendants():
            print(f"{'[group] ' if lyr.is_group() else ''}{lyr.name!r}  bbox={lyr.bbox}  visible={lyr.visible}")
        return

    if a.report:
        if not a.audio:
            sys.exit("--report needs --audio")
        vol, target = audio_features(a.audio)
        report(target)
        return

    if a.scale != 1.0:
        scale_rig(a.scale)
    size, draw, mane_rig = load_psd(a.psd, a.scale)
    rng = np.random.default_rng(a.seed)
    cues = json.load(open(a.cues)) if a.cues else []

    if a.limits_test:
        tracks, n = limits_test_tracks()
        a.audio = None
        if a.out == "out.mp4": a.out = "limits_test.mp4"
    elif a.audio:
        vol, target = audio_features(a.audio)
        tracks, n = build_tracks(len(vol), vol, target, cues, rng)
    elif a.idle:
        tracks, n = build_tracks(int(a.idle * FPS), None, None, cues, rng)
    else:
        sys.exit("Give --audio, --idle SECONDS, --limits-test, or --list.")
    fx = None if (a.limits_test or a.no_fx) else make_fx(n, size, np.random.default_rng(a.seed + 1000))
    render(size, draw, tracks, n, a.audio, a.out, a.png, fx, mane_rig)


if __name__ == "__main__":
    main()
