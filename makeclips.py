#!/usr/bin/env python3
"""
makeclips.py - placeholder clips for testing the renderer without real puppet clips.

Builds idle.mp4 plus a few line clips. Every clip opens and closes on the SAME
rest frame, exactly as puppet.py's output does, so the seamless switching is
being tested for real and not faked. Each clip is labelled and carries a frame
counter, so a visible jump at a transition is obvious.

  python3 makeclips.py                 -> clips/ with idle + 3 lines + sidecars
  python3 makeclips.py --dir clips --secs 4

Needs ffmpeg on PATH (free, ffmpeg.org). Nothing else.
"""
import argparse
import json
import os
import subprocess

W, H, FPS = 1080, 1920, 12
REST_HOLD = 0.5          # seconds of rest frame at each end
FONT = "sans"            # ffmpeg's drawtext font lookup; see notes at the bottom


def ffmpeg(args):
    subprocess.run(["ffmpeg", "-y", "-v", "error"] + args, check=True)


def rest_frame(path):
    """The frame every clip starts and ends on."""
    ffmpeg(["-f", "lavfi", "-i", f"color=c=0x101418:s={W}x{H}",
            "-vf", f"drawtext=font={FONT}:text='REST FRAME':fontcolor=0x8899aa:fontsize=70:"
                   f"x=(w-text_w)/2:y=(h-text_h)/2,"
                   f"drawbox=x=40:y=40:w={W-80}:h={H-80}:color=0x2a3340:t=6",
            "-frames:v", "1", path])


def clip(path, label, secs, rest_png, hue, silent=False):
    """rest frame -> moving labelled content -> rest frame, with a tone for audio."""
    body = secs
    ffmpeg([
        "-loop", "1", "-t", str(REST_HOLD), "-i", rest_png,
        "-f", "lavfi", "-t", str(body), "-i",
        f"color=c=0x{hue}:s={W}x{H},"
        f"drawtext=font={FONT}:text='{label}':fontcolor=white:fontsize=110:"
        f"x=(w-text_w)/2:y=h/2-260,"
        f"drawtext=font={FONT}:text='FRAME %{{frame_num}}':fontcolor=white:fontsize=64:"
        f"x=(w-text_w)/2:y=h/2-80,"
        f"drawtext=font={FONT}:text='%{{pts\\:hms}}':fontcolor=white:fontsize=64:"
        f"x=(w-text_w)/2:y=h/2+40,"
        f"drawbox=x=40:y=40:w={W-80}:h={H-80}:color=white@0.35:t=6",
        "-loop", "1", "-t", str(REST_HOLD), "-i", rest_png,
        "-f", "lavfi", "-t", str(REST_HOLD * 2 + body), "-i",
        f"sine=frequency=220:beep_factor=4",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
        "-map", "[v]"] + ([] if silent else ["-map", "3:a"]) + ["-r", str(FPS),
        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p"] +
        ([] if silent else ["-c:a", "aac"]) + ["-shortest", path])


def sidecar(path, label, secs):
    """Board cues spread across the clip, so timeline firing can be checked."""
    cues = [{"t": 0.2, "id": "open", "lines": [label, "CUE AT 0.2S"]},
            {"t": round(secs * 0.4, 2), "id": "mid", "lines": [label, "CUE AT " +
             str(round(secs * 0.4, 2)) + "S", "SECOND LINE"]},
            {"t": round(secs * 0.8, 2), "id": "clear", "lines": ["", "", ""]}]
    with open(path, "w") as f:
        json.dump({"cues": cues}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="clips")
    ap.add_argument("--secs", type=float, default=4.0, help="length of each clip's body")
    ap.add_argument("--silent", action="store_true",
                    help="no audio track, for loops whose sound comes from the bed")
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)

    rest = os.path.join(a.dir, "_rest.png")
    rest_frame(rest)

    plan = [("idle", "IDLE LOOP", "141c24"), ("line01", "LINE 01", "23303f"),
            ("line02", "LINE 02", "2d2438"), ("line03", "LINE 03", "1f3330")]
    for name, label, hue in plan:
        mp4 = os.path.join(a.dir, name + ".mp4")
        clip(mp4, label, a.secs, rest, hue, a.silent)
        sidecar(os.path.join(a.dir, name + ".board.json"), label, a.secs)
        print(f"  {mp4}")

    tone = os.path.join(a.dir, "..", "audio")
    os.makedirs(tone, exist_ok=True)
    for name, freq in (("vamp_1", 110), ("vamp_2", 146)):
        ffmpeg(["-f", "lavfi", "-t", "8", "-i",
                f"sine=frequency={freq}:beep_factor=3", os.path.join(tone, name + ".wav")])
        print(f"  {os.path.join(tone, name + '.wav')}")

    print(f"\nDone. Try:\n"
          f"  python3 renderer.py --clips {a.dir}\n"
          f"  python3 showclient.py            then: p line01 idle\n\n"
          f"Watch the transitions: the rest frame should hold steady with no flash,\n"
          f"and the board should fire its cues while each clip plays.")


if __name__ == "__main__":
    main()

# If drawtext fails with a font error, replace FONT above with a full path, e.g.
#   Windows: C\\:/Windows/Fonts/arial.ttf      (the escaped colon is required)
#   Linux:   /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
# and change "font=" to "fontfile=" in the three drawtext filters.
