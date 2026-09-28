#!/usr/bin/env python3
"""
audiotest.py - audition part1 -> looping vamp -> part2, live, with no rendering.

Plays part 1, loops the vamp until you press Enter, then dissolves it and plays
part 2 - the same shape as the piece. Everything is mixed here in one continuous
audio stream, so every seam is a sample-accurate crossfade: no player starting or
stopping, no stepped volume changes, nothing to click.

  python3 audiotest.py part1.wav vamp.wav part2.wav
  python3 audiotest.py part1.wav vamp.wav part2.wav --xfade 0.4 --fade 0.8 --gap 1.5
  python3 audiotest.py part1.wav vamp.wav part2.wav --fade 0.8 --overlap 0.8
      part 2 starts the moment you answer, while the vamp dies away under it
  python3 audiotest.py --list-devices
  python3 audiotest.py part1.wav vamp.wav part2.wav --device 7

  Enter  answer now: dissolve the vamp and go on to part 2
  s      skip ahead, during part 1
  q      quit

Needs ffmpeg on PATH, and: python3 -m pip install sounddevice numpy
(sounddevice bundles PortAudio on Windows; on Linux: sudo apt install libportaudio2)
"""
import argparse
import subprocess
import sys
import threading
import time

import numpy as np

SR = 48000
CH = 2


def load(path):
    """Decode anything ffmpeg can read to float32 stereo at SR."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", str(CH), "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True)
    if raw.returncode != 0 or not raw.stdout:
        sys.exit(f"can't decode {path}: {raw.stderr.decode(errors='replace').strip()}")
    buf = np.frombuffer(raw.stdout, np.float32).reshape(-1, CH).copy()
    # a 5 ms micro-fade at each edge: a cut that doesn't land on a zero crossing
    # otherwise starts or stops mid-waveform, which is heard as a click
    edge = min(int(0.005 * SR), len(buf) // 2)
    if edge:
        ramp = np.linspace(0.0, 1.0, edge, dtype=np.float32)[:, None]
        buf[:edge] *= ramp
        buf[-edge:] *= ramp[::-1]
    return buf


class Voice:
    """One buffer playing, with a gain that can ramp smoothly to a target."""

    def __init__(self, buf, fade_in=0):
        self.buf, self.pos = buf, 0
        self.gain = 0.0 if fade_in else 1.0
        self.target, self.step = 1.0, (1.0 / fade_in) if fade_in else 0.0
        self.handed_over = False

    def fade_to(self, target, samples):
        self.target = target
        self.step = abs(target - self.gain) / max(samples, 1)

    @property
    def remaining(self):
        return len(self.buf) - self.pos

    @property
    def done(self):
        return self.remaining <= 0 or (self.target == 0.0 and self.gain <= 0.0)

    def render(self, frames):
        chunk = self.buf[self.pos:self.pos + frames]
        k = len(chunk)
        self.pos += k
        if k == 0:
            return np.zeros((0, CH), np.float32)
        # per-sample gain ramp: smooth, so there is no zipper noise
        direction = 1.0 if self.target > self.gain else -1.0
        ramp = self.gain + direction * self.step * np.arange(1, k + 1, dtype=np.float32)
        ramp = np.minimum(ramp, self.target) if direction > 0 else np.maximum(ramp, self.target)
        if self.step == 0:
            ramp = np.full(k, self.gain, np.float32)
        self.gain = float(ramp[-1])
        return chunk * ramp[:, None]


class Mixer:
    """Voices handle crossfades between passes; a separate master envelope handles
    the exit, so the vamp keeps cycling underneath while it dissolves."""

    def __init__(self, part1, vamp, part2, xfade, fade, gap, overlap=0.0):
        self.part1, self.vamp, self.part2 = part1, vamp, part2
        self.xf = int(xfade * SR)
        # a "0" fade is really 5 ms: still heard as a cut, but it can't click
        self.fade = max(int(0.005 * SR), int(fade * SR))
        self.gap = int(gap * SR)
        self.overlap = min(int(overlap * SR), self.fade)
        if self.xf and len(vamp) <= self.xf * 2:
            sys.exit("the vamp is too short for that crossfade")
        self.voices = [Voice(part1)]
        self.main = self.voices[0]
        self.p2voice = None
        self.phase = "part1"
        self.passes = 0
        self.master = 1.0            # exit envelope on everything except part 2
        self.fading = False
        self.fade_elapsed = 0
        self.gap_left = 0
        self.answer = threading.Event()
        self.skip = threading.Event()
        self.finished = threading.Event()
        self.log = []

    def _hand_over(self):
        old = self.main
        old.handed_over = True
        old.fade_to(0.0, min(self.xf, max(old.remaining, 1)) if self.xf else 1)
        new = Voice(self.vamp, fade_in=self.xf)
        self.voices.append(new)
        self.main = new
        if self.phase == "part1":
            self.phase = "vamp"
            self.log.append("vamp")
        else:
            self.passes += 1
            if not self.fading:
                self.log.append(f"pass {self.passes + 1}")

    def _start_part2(self):
        self.p2voice = Voice(self.part2)
        self.voices.append(self.p2voice)
        self.phase = "part2"
        self.log.append("part2")

    def _schedule(self):
        looping = self.phase in ("part1", "vamp") or (self.fading and self.p2voice is None)
        if self.phase == "vamp" and self.answer.is_set() and not self.fading:
            self.fading, self.fade_elapsed = True, 0
            self.log.append("fadeout")
        if looping and self.phase != "gap" and not self.main.handed_over and (
                self.main.remaining <= self.xf + 1 or
                (self.phase == "part1" and self.skip.is_set())):
            self.skip.clear()
            self._hand_over()
        if self.fading and self.p2voice is None and self.phase != "gap":
            if self.overlap and self.fade_elapsed >= self.fade - self.overlap:
                self._start_part2()
            elif self.fade_elapsed >= self.fade:
                self.voices = []
                self.phase, self.gap_left = "gap", self.gap
        elif self.phase == "gap" and self.gap_left <= 0:
            self._start_part2()
        elif self.phase == "part2" and self.p2voice.done and \
                (self.master <= 0.0 or not self.fading):
            self.phase = "done"
            self.finished.set()

    def _next_event(self):
        """Samples until something needs deciding, so every change is sample-exact."""
        limits = []
        if self.phase in ("part1", "vamp") or (self.fading and self.p2voice is None):
            if not self.main.handed_over and self.phase != "gap":
                limits.append(self.main.remaining - self.xf)
        if self.fading and self.p2voice is None and self.phase != "gap":
            due = self.fade - self.overlap if self.overlap else self.fade
            limits.append(due - self.fade_elapsed)
        if self.phase == "gap":
            limits.append(self.gap_left)
        return max(1, min(limits)) if limits else None

    def callback(self, outdata, frames, _time, status):
        out = np.zeros((frames, CH), np.float32)
        filled = 0
        while filled < frames:
            self._schedule()
            if self.phase == "done":
                break
            n = frames - filled
            nxt = self._next_event()
            if nxt is not None:
                n = min(n, nxt)
            if self.phase == "gap":
                self.gap_left -= n
                filled += n
                continue
            under = np.zeros((n, CH), np.float32)
            for v in list(self.voices):
                part = v.render(n)
                target = out[filled:filled + len(part)] if v is self.p2voice else under[:len(part)]
                target += part
                if v.done and v is not self.p2voice:
                    self.voices.remove(v)
            if self.fading and self.master > 0.0:
                ramp = self.master - np.arange(1, n + 1, dtype=np.float32) / self.fade
                np.maximum(ramp, 0.0, out=ramp)
                under *= ramp[:, None]
                self.master = float(ramp[-1])
                self.fade_elapsed += n
            elif self.fading:
                under[:] = 0.0
                self.fade_elapsed += n
            out[filled:filled + n] += under
            filled += n
        np.clip(out, -1.0, 1.0, out=out)
        outdata[:] = out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part1", nargs="?")
    ap.add_argument("vamp", nargs="?")
    ap.add_argument("part2", nargs="?")
    ap.add_argument("--xfade", type=float, default=0.25, help="overlap at each seam, seconds")
    ap.add_argument("--fade", type=float, default=1.4, help="vamp dissolve on answering")
    ap.add_argument("--gap", type=float, default=0.3, help="silence before part 2")
    ap.add_argument("--overlap", type=float, default=0.0,
                    help="seconds of the vamp's fade that part 2 plays under; "
                         "set it equal to --fade to start part 2 the instant you answer")
    ap.add_argument("--device", default=None, help="output device index or name")
    ap.add_argument("--list-devices", action="store_true")
    a = ap.parse_args()

    import sounddevice as sd
    if a.list_devices:
        print(sd.query_devices())
        return
    if not (a.part1 and a.vamp and a.part2):
        ap.error("give part1, vamp and part2")

    print("decoding ...")
    p1, vp, p2 = load(a.part1), load(a.vamp), load(a.part2)
    print(f"part1 {len(p1) / SR:.1f}s | vamp {len(vp) / SR:.1f}s | part2 {len(p2) / SR:.1f}s")
    mix = Mixer(p1, vp, p2, a.xfade, a.fade, a.gap, a.overlap)

    def keys():
        while not mix.finished.is_set():
            line = sys.stdin.readline().strip().lower()
            if line == "q":
                mix.finished.set()
            elif line == "s" and mix.phase == "part1":
                mix.skip.set()
            elif mix.phase == "vamp":
                mix.answer.set()
    threading.Thread(target=keys, daemon=True).start()

    device = int(a.device) if a.device and a.device.isdigit() else a.device
    print(f"\n[1] {a.part1}   (s to skip ahead, q to quit)")
    vamp_at = None
    with sd.OutputStream(samplerate=SR, channels=CH, dtype="float32", device=device,
                         blocksize=512, callback=mix.callback):
        shown = 0
        while not mix.finished.wait(0.05):
            while shown < len(mix.log):
                event = mix.log[shown]
                shown += 1
                if event == "vamp":
                    vamp_at = time.monotonic()
                    print(f"[2] {a.vamp} looping   -- press Enter when you would answer")
                elif event.startswith("pass"):
                    print(f"    {event}")
                elif event == "fadeout" and vamp_at:
                    print(f"    answered after {time.monotonic() - vamp_at:.1f}s; "
                          f"dissolving over {a.fade}s")
                elif event == "part2":
                    print(f"[3] {a.part2}")
    print("done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
