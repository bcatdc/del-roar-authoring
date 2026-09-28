#!/usr/bin/env python3
"""
splitshow.py - cut one long voice track into per-beat clips and audio beds,
then render each one with puppet.py.

Three steps, so you stay in control of where the cuts land:

  0. (better, if the vamps have music under them) label the beats in Audacity:
       select a beat, Ctrl-B, type its name, then File > Export > Export Labels.
       python3 splitshow.py labels labels.txt
       A label whose name ends in " bed" becomes an audio bed; the rest are clips.

  1. python3 splitshow.py detect full.wav
       Finds the silences, prints the segments it would make, and writes
       show_plan.json. Edit that file: name the beats, mark the vamps, merge
       or drop anything wrong.

  2. python3 splitshow.py cut full.wav
       Writes one .wav per segment into cuts/, cutting at the middle of each
       silence so nothing is clipped and each piece starts and ends on quiet.

  3. python3 splitshow.py render del-roar.psd
       Runs puppet.py over each voice segment into clips/, copies the beds into
       audio/, and renders a silent idle loop. Long: one render per beat.

The plan file looks like this, and hand-editing it is the point:

  {"segments": [
    {"name": "intro",   "start": 0.0,  "end": 12.4, "kind": "clip"},
    {"name": "vamp_1",  "start": 12.4, "end": 28.9, "kind": "bed"},
    {"name": "prompt1", "start": 28.9, "end": 41.2, "kind": "clip"}
  ]}

kind "clip" becomes a puppet clip with its own audio; "bed" becomes a looping
track for the audio bed; "skip" is ignored.

Needs ffmpeg on PATH. puppet.py is called for rendering; point at it with --puppet
if it isn't in this folder.
"""
import argparse
import json
import os
import re
import subprocess
import sys

PLAN = "show_plan.json"
SILENCE_DB = "-34dB"          # quieter than this counts as silence
SILENCE_MIN = 0.45            # seconds of quiet before it's a gap
PAD = 0.12                    # seconds of quiet kept at each end of a cut


def run(args, capture=False):
    return subprocess.run(args, capture_output=capture, text=True, check=True)


def duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "csv=p=0", path], capture=True).stdout.strip()
    return float(out)


def silences(path, db, minlen):
    """[(start, end)] of every silent stretch, via ffmpeg's silencedetect."""
    p = subprocess.run(["ffmpeg", "-v", "info", "-i", path, "-af",
                        f"silencedetect=noise={db}:d={minlen}", "-f", "null", "-"],
                       capture_output=True, text=True)
    starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", p.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", p.stderr)]
    return list(zip(starts, ends + [duration(path)]))[:len(starts)]


def detect(a):
    total = duration(a.audio)
    gaps = silences(a.audio, a.db, a.min_silence)
    print(f"{a.audio}: {total:.1f}s, {len(gaps)} silences over {a.min_silence}s\n")

    # cut in the middle of each silence; long silences are probably vamps
    bounds = [0.0] + [round((s + e) / 2, 2) for s, e in gaps] + [round(total, 2)]
    segs = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        if end - start < 0.4:
            continue
        quiet = sum(max(0.0, min(e, end) - max(s, start)) for s, e in gaps)
        kind = "bed" if quiet > (end - start) * 0.6 else "clip"
        segs.append({"name": f"{'vamp' if kind == 'bed' else 'beat'}{len(segs):02d}",
                     "start": start, "end": end, "kind": kind})

    for s in segs:
        print(f"  {s['start']:7.2f} - {s['end']:7.2f}  {s['end']-s['start']:6.2f}s  "
              f"{s['kind']:5s}  {s['name']}")
    if not a.dry_run:
        with open(a.plan, "w") as f:
            json.dump({"segments": segs}, f, indent=2)
        print(f"\nwrote {a.plan} - edit the names and kinds, then: splitshow.py cut {a.audio}")


def labels(a):
    """Audacity label export: start<TAB>end<TAB>name, one per line."""
    segs = []
    for line in open(a.labels):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        start, end, name = float(parts[0]), float(parts[1]), parts[2].strip()
        kind = "clip"
        for suffix, k in ((" bed", "bed"), (" skip", "skip")):
            if name.lower().endswith(suffix):
                name, kind = name[: -len(suffix)].strip(), k
        segs.append({"name": name.replace(" ", "_") or f"beat{len(segs):02d}",
                     "start": round(start, 2), "end": round(end, 2), "kind": kind})
    segs.sort(key=lambda s: s["start"])
    for s in segs:
        print(f"  {s['start']:7.2f} - {s['end']:7.2f}  {s['end']-s['start']:6.2f}s  "
              f"{s['kind']:5s}  {s['name']}")
    with open(a.plan, "w") as f:
        json.dump({"segments": segs}, f, indent=2)
    print(f"\nwrote {a.plan} from {len(segs)} labels")


def cut(a):
    plan = json.load(open(a.plan))["segments"]
    os.makedirs(a.cuts, exist_ok=True)
    total = duration(a.audio)
    made = []
    for s in plan:
        if s["kind"] == "skip":
            continue
        start = max(0.0, s["start"] - PAD)
        end = min(total, s["end"] + PAD)
        out = os.path.join(a.cuts, s["name"] + ".wav")
        run(["ffmpeg", "-y", "-v", "error", "-i", a.audio, "-ss", f"{start:.3f}",
             "-to", f"{end:.3f}", out])
        made.append((out, s["kind"], end - start))
        print(f"  {out}  {end-start:.2f}s  {s['kind']}")
    print(f"\n{len(made)} files in {a.cuts}/. Listen through them before rendering.")


def render(a):
    plan = json.load(open(a.plan))["segments"]
    os.makedirs(a.clips, exist_ok=True)
    os.makedirs(a.audio_dir, exist_ok=True)
    clips = [s for s in plan if s["kind"] == "clip"]
    beds = [s for s in plan if s["kind"] == "bed"]

    for i, s in enumerate(beds, 1):
        src = os.path.join(a.cuts, s["name"] + ".wav")
        dst = os.path.join(a.audio_dir, s["name"] + ".wav")
        if os.path.isfile(src):
            run(["ffmpeg", "-y", "-v", "error", "-i", src, dst])
            print(f"  bed {i}/{len(beds)}: {dst}")

    print(f"\nrendering {len(clips)} clips - this is the slow part")
    for i, s in enumerate(clips, 1):
        wav = os.path.join(a.cuts, s["name"] + ".wav")
        out = os.path.join(a.clips, s["name"] + ".mp4")
        if not os.path.isfile(wav):
            print(f"  skipping {s['name']}: no cut found")
            continue
        if os.path.isfile(out) and not a.force:
            print(f"  {i}/{len(clips)} {s['name']}: already rendered")
            continue
        print(f"  {i}/{len(clips)} {s['name']} ...", flush=True)
        run([sys.executable, a.puppet, a.psd, "--audio", wav, "-o", out])

    idle = os.path.join(a.clips, "idle.mp4")
    if not os.path.isfile(idle) or a.force:
        print(f"  idle loop ({a.idle_secs}s, silent) ...")
        run([sys.executable, a.puppet, a.psd, "--idle", str(a.idle_secs), "-o", idle])
    print(f"\nDone. Try:  python3 renderer.py --clips {a.clips} --audio {a.audio_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect")
    d.add_argument("audio")
    d.add_argument("--db", default=SILENCE_DB)
    d.add_argument("--min-silence", type=float, default=SILENCE_MIN)
    d.add_argument("--plan", default=PLAN)
    d.add_argument("--dry-run", action="store_true")
    d.set_defaults(func=detect)

    l = sub.add_parser("labels")
    l.add_argument("labels", help="Audacity label export (.txt)")
    l.add_argument("--plan", default=PLAN)
    l.set_defaults(func=labels)

    c = sub.add_parser("cut")
    c.add_argument("audio")
    c.add_argument("--plan", default=PLAN)
    c.add_argument("--cuts", default="cuts")
    c.set_defaults(func=cut)

    r = sub.add_parser("render")
    r.add_argument("psd")
    r.add_argument("--plan", default=PLAN)
    r.add_argument("--cuts", default="cuts")
    r.add_argument("--clips", default="clips")
    r.add_argument("--audio-dir", default="audio")
    r.add_argument("--puppet", default="puppet.py")
    r.add_argument("--idle-secs", type=float, default=8.0)
    r.add_argument("--force", action="store_true", help="re-render clips that exist")
    r.set_defaults(func=render)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
