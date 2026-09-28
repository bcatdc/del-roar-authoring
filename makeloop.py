#!/usr/bin/env python3
"""
makeloop.py - turn a rough region of audio into a seamless loop.

Finding a perfect loop point by ear is slow. Instead this overlaps the end of the
region back onto its beginning with a crossfade, so the join is hidden rather
than avoided. The result is shorter than the input by the crossfade length.

  python3 makeloop.py vamp_raw.wav                       # 1.5s crossfade
  python3 makeloop.py vamp_raw.wav --xfade 3 --out vamp_1.wav
  python3 makeloop.py vamp_raw.wav --start 4 --end 20    # use part of a longer file
  python3 makeloop.py vamp_raw.wav --check               # writes a 4x preview too

What the crossfade length does:
  short (0.3-0.8s)  keeps rhythmic detail, but a strong beat may flam
  long  (2-4s)      hides almost anything; best for drones, hum, texture

Needs ffmpeg on PATH.
"""
import argparse
import os
import subprocess
import sys


def dur(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    if not out:
        sys.exit(f"can't read {path}")
    return float(out)


def run(args):
    subprocess.run(["ffmpeg", "-y", "-v", "error"] + args, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--out", default=None)
    ap.add_argument("--xfade", type=float, default=1.5, help="seconds of overlap")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--check", action="store_true", help="also write a 4x repeat to listen to")
    a = ap.parse_args()

    out = a.out or os.path.splitext(a.audio)[0] + "_loop.wav"
    total = dur(a.audio)
    end = a.end if a.end is not None else total
    region = end - a.start
    if region <= a.xfade * 2:
        sys.exit(f"region is {region:.1f}s; needs to be more than twice the "
                 f"{a.xfade}s crossfade")

    # cut the region, then crossfade its tail onto its head
    tmp = out + ".region.wav"
    run(["-i", a.audio, "-ss", f"{a.start:.3f}", "-to", f"{end:.3f}", tmp])
    body = region - a.xfade
    run(["-i", tmp, "-i", tmp, "-filter_complex",
         f"[0:a]atrim=0:{body:.3f},asetpts=PTS-STARTPTS[head];"
         f"[1:a]atrim={body:.3f}:{region:.3f},asetpts=PTS-STARTPTS[tail];"
         f"[tail][head]acrossfade=d={a.xfade}:c1=tri:c2=tri[out]",
         "-map", "[out]", out])
    os.remove(tmp)
    print(f"{a.audio}: {region:.2f}s region -> {dur(out):.2f}s loop "
          f"({a.xfade:.1f}s crossfade)  ->  {out}")

    if a.check:
        chk = os.path.splitext(out)[0] + "_x4.wav"
        run(["-stream_loop", "3", "-i", out, chk])
        print(f"listen to {chk}: four passes back to back, the join should not stand out")


if __name__ == "__main__":
    main()
