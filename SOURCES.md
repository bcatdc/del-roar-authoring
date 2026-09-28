# Sources (not in this repo)

The tools here are useless without these. They are large and binary, so they are
backed up rather than versioned.

## Artwork
`Del Roar.psd`, 1080x1920, last known location
`C:\Users\benjc\OneDrive\Desktop\Friday Night - del Roar\`.

puppet.py matches layers by name, so these must not be renamed:

| Layer | Used as |
|---|---|
| `Backdrop` | static background |
| `Body` | parent of everything; pivot (552, 1577) at the belly |
| `5 Head, no Jaw` | head, pivots at the neck |
| `3 Upper Jaw` / `4 Lower Jaw` | mouth |
| `6 Pupils` | eyes |
| `1 Left Blink` / `2 Right Blink` | eyelids, hidden by default |
| `7 Left Hand` | elbow pivot (237, 1446) |
| `8 Right Hand` | elbow pivot (887, 1407) |
| `Mane Mask` | hidden greyscale: white = mane moves, black = still |

## Audio
`C:\Users\benjc\OneDrive\Desktop\Sunday Night\`:
`Segment 1-7 - voice.wav` (drives the jaw), `Segment 1-7.wav` (the mix that plays),
`Loop 1-6.wav` (the vamps under the listening states).

## Outputs
Rendered clips live only on the NUC: `~/delroar/clips3` and `~/delroar/audio`.
They are derived from the above and are not backed up; re-render if lost.

## Backup
OneDrive is sync, not backup: a deletion or corruption propagates. Keep a dated
copy of the PSD and the Sunday Night audio on another drive.
