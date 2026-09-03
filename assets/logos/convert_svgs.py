#!/usr/bin/env python3
import os
from pathlib import Path
try:
    import cairosvg
except Exception:
    print('cairosvg not installed')
    raise

ASSETS_DIR = Path(__file__).parent
SVG_DIR = ASSETS_DIR
OUT_DIR = ASSETS_DIR
SIZES = [128, 256, 512]

for svg in SVG_DIR.glob('*.svg'):
    name = svg.stem
    for s in SIZES:
        out = OUT_DIR / f"{name}-{s}.png"
        print(f"Rendering {svg} -> {out} ({s}x{s})")
        cairosvg.svg2png(url=str(svg), write_to=str(out), output_width=s, output_height=s)
print('Done')
