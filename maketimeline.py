#!/usr/bin/env python3
"""
maketimeline.py - turn a voice track into a board timeline sidecar.

Runs Whisper over the recording, wraps each spoken line to the board, and writes
<clip>.board.json so the flapboard captions Del Roar as he talks. Meant for a
rough cut: get something watchable, then hand-edit the JSON, which is the point
of it being a plain file.

  python3 maketimeline.py full.wav --out clips/full.board.json
  python3 maketimeline.py full.wav --model small.en --lead 0.6
  python3 maketimeline.py --segments segments.json --out clips/full.board.json
  python3 maketimeline.py full.wav --dry-run

--segments takes a JSON list of {"start":s,"end":s,"text":"..."} instead of
running Whisper, so you can hand-time a track or reuse an earlier pass.

Install: python3 -m pip install faster-whisper     (free, github.com/SYSTRAN/faster-whisper)
"""
import argparse
import json
import sys

# the board flaps for roughly this long on a full change; text shown later than
# the voice reads as lagging, so cues are pulled earlier by this much
DEFAULT_LEAD = 1.2
MIN_GAP = 0.35            # merge segments closer together than this
CLEAR_AFTER = 2.5         # blank the board if nothing is said for this long


def wrap(text, cols, rows, keep="split"):
    """Fit words into lines of `cols`. keep= how to handle more lines than fit:
    split  -> every page of `rows` lines, paced across the segment (Del Roar talking)
    tail   -> just the last page (a visitor's answer, where the end matters)
    head   -> just the first page"""
    words = "".join(c for c in text.upper() if c.isalnum() or c in " .,:'?!-/&$#@*+%").split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + (1 if cur else 0) <= cols:
            cur = f"{cur} {w}".strip()
        else:
            lines.append(cur)
            cur = w[:cols]
    if cur:
        lines.append(cur)
    pages = [lines[i:i + rows] for i in range(0, len(lines), rows)] or [[]]
    if keep == "tail":
        return [pages[-1]]
    if keep == "head":
        return [pages[0]]
    return pages


def segments_from_whisper(path, model_name, compute):
    from faster_whisper import WhisperModel
    print(f"[timeline] loading {model_name} ...", file=sys.stderr)
    model = WhisperModel(model_name, device="cpu", compute_type=compute)
    segs, _info = model.transcribe(path, language="en", beam_size=5, vad_filter=True)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segs]


def merge(segments, min_gap):
    out = []
    for s in segments:
        if out and s["start"] - out[-1]["end"] < min_gap:
            out[-1]["end"] = s["end"]
            out[-1]["text"] = (out[-1]["text"] + " " + s["text"]).strip()
        else:
            out.append(dict(s))
    return out


def build(segments, cols, rows, lead, clear_after, keep="split"):
    cues = []
    for i, s in enumerate(segments):
        pages = [p for p in wrap(s["text"], cols, rows, keep) if any(p)]
        if not pages:
            continue
        span = max(0.6, s["end"] - s["start"])
        for j, page in enumerate(pages):
            # pages land evenly across the line's delivery, each shown early
            t = max(0.0, s["start"] + span * j / len(pages) - lead)
            if cues and t <= cues[-1]["t"] + 0.1:
                t = cues[-1]["t"] + 0.2
            cues.append({"t": round(t, 2), "id": f"line{i:02d}_{j}" if len(pages) > 1
                         else f"line{i:02d}", "lines": page + [""] * (rows - len(page))})
        nxt = segments[i + 1]["start"] if i + 1 < len(segments) else None
        if nxt is None or nxt - s["end"] >= clear_after:
            cues.append({"t": round(s["end"] + 0.6, 2), "id": f"clear{i:02d}",
                         "lines": [""] * rows})
    return cues


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="?", help="the voice track")
    ap.add_argument("--segments", help="JSON list of {start,end,text} instead of Whisper")
    ap.add_argument("--out", default="timeline.board.json")
    ap.add_argument("--model", default="base.en")
    ap.add_argument("--compute", default="int8")
    ap.add_argument("--cols", type=int, default=20)
    ap.add_argument("--rows", type=int, default=2, help="board rows the captions use")
    ap.add_argument("--lead", type=float, default=DEFAULT_LEAD,
                    help="seconds to show text early, covering the flap time")
    ap.add_argument("--gap", type=float, default=MIN_GAP)
    ap.add_argument("--clear-after", type=float, default=CLEAR_AFTER)
    ap.add_argument("--keep", choices=["split", "tail", "head"], default="split",
                    help="long lines: page through them, or keep only the end/start")
    ap.add_argument("--dry-run", action="store_true", help="print, don't write")
    a = ap.parse_args()

    if a.segments:
        segments = json.load(open(a.segments))
    elif a.audio:
        segments = segments_from_whisper(a.audio, a.model, a.compute)
    else:
        sys.exit("give an audio file or --segments")

    segments = merge(segments, a.gap)
    cues = build(segments, a.cols, a.rows, a.lead, a.clear_after, a.keep)

    for c in cues:
        shown = " | ".join(x for x in c["lines"] if x) or "(clear)"
        print(f"{c['t']:8.2f}  {shown}")
    print(f"\n{len(cues)} cues from {len(segments)} spoken segments", file=sys.stderr)

    if not a.dry_run:
        with open(a.out, "w") as f:
            json.dump({"cues": cues}, f, indent=2)
        print(f"wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
