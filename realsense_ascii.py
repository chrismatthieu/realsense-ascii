#!/usr/bin/env python3
"""Live ASCII art from an Intel RealSense depth camera.

Closer surfaces draw denser, brighter characters. Far surfaces fade out.
A relief style swaps in slope glyphs so round objects read as 3D.

    python3 realsense_ascii.py
    python3 realsense_ascii.py --demo
"""

from __future__ import annotations

import argparse
import select
import shutil
import sys
import time
import termios
import tty
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - depends on the local librealsense build
    rs = None


RAMPS = {
    "classic": " .:-=+*#%@",
    "soft": " .·:+*#",
    "blocks": " ░▒▓█",
    "heavy": " .oO#@",
}

# Theme stops run from far (0) to near (1).
THEMES: dict[str, list[tuple[float, tuple[int, int, int]]]] = {
    "thermal": [
        (0.0, (12, 0, 28)),
        (0.22, (140, 0, 70)),
        (0.48, (230, 40, 16)),
        (0.72, (255, 160, 0)),
        (1.0, (255, 248, 220)),
    ],
    "ocean": [
        (0.0, (2, 10, 28)),
        (0.35, (0, 70, 140)),
        (0.68, (0, 190, 200)),
        (1.0, (230, 255, 255)),
    ],
    "matrix": [
        (0.0, (0, 24, 0)),
        (0.45, (0, 140, 40)),
        (1.0, (190, 255, 190)),
    ],
    "amber": [
        (0.0, (36, 14, 0)),
        (0.5, (190, 90, 0)),
        (1.0, (255, 214, 130)),
    ],
    "candy": [
        (0.0, (36, 0, 48)),
        (0.4, (190, 0, 150)),
        (0.72, (255, 90, 190)),
        (1.0, (190, 255, 255)),
    ],
    "mono": [
        (0.0, (55, 55, 60)),
        (1.0, (235, 235, 240)),
    ],
}

THEME_ORDER = list(THEMES)
RAMP_ORDER = list(RAMPS)
LUT_N = 24
CHAR_ASPECT = 0.55  # cell width / cell height

HELP_LINES = (
    "depth mirror",
    "",
    "  c        cycle color",
    "  i        invert near and far",
    "  m        mirror the camera",
    "  r        density or relief",
    "  1-4      character set",
    "  a        auto contrast",
    "  [ ]      shorter or longer depth window",
    "  space    pause",
    "  h        hide this help",
    "  q        quit",
)


@dataclass
class View:
    theme: str = "thermal"
    ramp: str = "classic"
    style: str = "density"
    invert: bool = False
    mirror: bool = True
    auto_contrast: bool = True
    near: float = 0.4
    span: float = 1.8
    show_help: bool = False
    paused: bool = False


@dataclass
class Stats:
    lo: float = 0.0
    hi: float = 0.0
    fps: float = 0.0
    demo: bool = False
    valid: float = 0.0


def build_lut(stops: list[tuple[float, tuple[int, int, int]]], n: int = LUT_N) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, n)
    cols = np.zeros((n, 3), np.float32)
    for i, x in enumerate(xs):
        for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
            if x <= p1 or i == n - 1:
                span = p1 - p0
                t = 0.0 if span <= 0 else (x - p0) / span
                t = float(np.clip(t, 0.0, 1.0))
                cols[i] = (1.0 - t) * np.array(c0) + t * np.array(c1)
                break
    return cols.astype(np.uint8)


LUTS = {name: build_lut(stops) for name, stops in THEMES.items()}
PREFIXES = {
    name: [f"\033[38;2;{r};{g};{b}m" for r, g, b in lut]
    for name, lut in LUTS.items()
}


def fit_grid(term_w: int, term_h: int, src_w: int, src_h: int, aspect: float = CHAR_ASPECT) -> tuple[int, int]:
    """Pick a character grid that fills the terminal without stretching the depth image."""
    usable_w = max(20, term_w)
    usable_h = max(8, term_h)
    h = int(round(usable_w * aspect * src_h / src_w))
    w = usable_w
    if h > usable_h:
        h = usable_h
        w = int(round(usable_h * src_w / (src_h * aspect)))
        w = max(20, min(w, usable_w))
    return w, max(8, h)


def sample_depth(depth: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Downsample by taking the nearest surface in a small neighborhood."""
    h, w = depth.shape
    y = np.linspace(0, h - 1, out_h).astype(np.int32)
    x = np.linspace(0, w - 1, out_w).astype(np.int32)
    step_y = max(int(h / out_h / 2), 1)
    step_x = max(int(w / out_w / 2), 1)
    y2 = np.clip(y + step_y, 0, h - 1)
    x2 = np.clip(x + step_x, 0, w - 1)
    samples = np.stack(
        [
            depth[y[:, None], x[None, :]],
            depth[y[:, None], x2[None, :]],
            depth[y2[:, None], x[None, :]],
            depth[y2[:, None], x2[None, :]],
        ]
    )
    samples = samples.astype(np.float32, copy=False)
    invalid = samples <= 0
    samples[invalid] = np.inf
    out = samples.min(axis=0)
    out[np.isinf(out)] = 0
    return out


def _sphere(depth: np.ndarray, nx: np.ndarray, ny: np.ndarray, cx: float, cy: float, z: float, r: float, sy: float = 1.0) -> None:
    d2 = (nx - cx) ** 2 + ((ny - cy) / sy) ** 2
    mask = d2 <= r * r
    if not np.any(mask):
        return
    bulge = np.sqrt(np.clip(r * r - d2, 0, None)) / r
    surface = z - bulge * r * 2.2
    closer = mask & (surface < depth)
    depth[closer] = surface[closer]


def synth_depth(t: float, h: int = 240, w: int = 320) -> np.ndarray:
    """A waving visitor, a floor, and a wandering orb. Units are meters."""
    _yy, xx = np.mgrid[0:h, 0:w]
    nx = ((xx - w / 2) / (w / 2)).astype(np.float32)
    ny = ((_yy - h / 2) / (h / 2)).astype(np.float32)

    depth = np.full((h, w), 3.6, np.float32)
    floor = ny > 0.48
    along = np.clip((ny - 0.48) / 0.52, 0.0, 1.0)
    depth[floor] = (2.45 - along * 0.45)[floor]

    bob = 0.03 * np.sin(t * 2.2)
    _sphere(depth, nx, ny, 0.0, 0.10 + bob, 1.28, 0.28, 1.5)
    _sphere(depth, nx, ny, 0.0, -0.50 + bob, 1.16, 0.155, 1.15)
    _sphere(depth, nx, ny, -0.38, 0.20 + bob, 1.36, 0.065, 1.8)
    wave = float(np.sin(t * 3.2))
    _sphere(depth, nx, ny, 0.34 + 0.08 * wave, -0.22 + bob - 0.22 * wave, 1.12, 0.07, 1.35)
    _sphere(depth, nx, ny, 0.55 * np.cos(t * 0.8), -0.15 + 0.2 * np.sin(t * 1.3), 1.7, 0.1)
    return depth


class DemoSource:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def read(self) -> np.ndarray:
        return synth_depth(time.perf_counter() - self._t0)

    def close(self) -> None:
        return None


class CameraSource:
    def __init__(self, width: int, height: int, fps: int) -> None:
        if rs is None:
            raise RuntimeError("pyrealsense2 is not importable")
        ctx = rs.context()
        if ctx.query_devices().size() == 0:
            raise RuntimeError("no RealSense device")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(config)
        sensor = profile.get_device().first_depth_sensor()
        self.scale = float(sensor.get_depth_scale())
        if sensor.supports(rs.option.visual_preset):
            sensor.set_option(rs.option.visual_preset, int(rs.rs400_visual_preset.high_density))
        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, 2)
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        self.filters = (decimation, spatial, temporal)

    def read(self) -> np.ndarray | None:
        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        depth = frames.get_depth_frame()
        if not depth:
            return None
        for filt in self.filters:
            depth = filt.process(depth)
        raw = np.asanyarray(depth.get_data())
        return raw.astype(np.float32) * self.scale

    def close(self) -> None:
        self.pipeline.stop()


def contrast_range(depth: np.ndarray, view: View, memory: list[float]) -> tuple[float, float, np.ndarray]:
    """Map a short window in front of the nearest surface across the full ramp.

    A fixed 0–5 m scale spends almost every character on empty room, so a person
    collapses to one glyph. Tracking the near depth keeps the subject modeled.
    """
    valid = (depth > 0.15) & (depth < 8.0)
    span = max(view.span, 0.25)
    if view.auto_contrast and int(valid.sum()) > 32:
        vals = depth[valid]
        lo = float(np.percentile(vals, 3))
        hi = min(lo + span, float(np.percentile(vals, 98)))
        if hi - lo < 0.25:
            hi = lo + 0.25
        if memory:
            lo = memory[0] * 0.82 + lo * 0.18
            hi = memory[1] * 0.82 + hi * 0.18
        memory[:] = [lo, hi]
    else:
        lo, hi = view.near, view.near + span
    return lo, hi, valid


def shade_grid(depth: np.ndarray, lo: float, hi: float, invert: bool, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    span = max(hi - lo, 1e-3)
    nearness = 1.0 - np.clip((depth - lo) / span, 0.0, 1.0)
    if invert:
        nearness = 1.0 - nearness
    # Spend more of the ramp on the closest surface, where the shape is.
    nearness = np.power(np.clip(nearness, 0.0, 1.0), 1.25)
    filled = depth.copy()
    if np.any(valid):
        filled[~valid] = float(np.median(depth[valid]))
    gy, gx = np.gradient(filled)
    mag = np.hypot(gx, gy)
    ref_src = mag[valid] if np.any(valid) else mag
    ref = float(np.percentile(ref_src, 90)) + 1e-6
    edge = np.clip(mag / ref, 0.0, 1.0)
    edge[~valid] = 0
    edge[[0, -1], :] = 0
    edge[:, [0, -1]] = 0
    shade = np.clip(nearness + edge * 0.22, 0.0, 1.0)
    shade[~valid] = 0
    return shade.astype(np.float32), edge


def density_chars(shade: np.ndarray, valid: np.ndarray, ramp: str) -> np.ndarray:
    glyphs = np.array(list(ramp))
    idx = np.clip((shade * (len(glyphs) - 1)).astype(np.int32), 0, len(glyphs) - 1)
    chars = glyphs[idx]
    chars = chars.copy()
    chars[~valid] = " "
    return chars


def relief_chars(depth: np.ndarray, shade: np.ndarray, valid: np.ndarray) -> np.ndarray:
    filled = depth.copy()
    if np.any(valid):
        filled[~valid] = float(np.median(depth[valid]))
    gy, gx = np.gradient(filled)
    mag = np.hypot(gx, gy)
    ref_src = mag[valid] if np.any(valid) else mag
    norm = mag / (float(np.percentile(ref_src, 90)) + 1e-6)
    faint = np.array(list(" .:-=+"))
    flat = faint[np.clip((shade * (len(faint) - 1)).astype(np.int32), 0, len(faint) - 1)]
    slope = np.where(
        np.abs(gx) >= np.abs(gy),
        np.where(gx >= 0, "/", "\\"),
        np.where(gy >= 0, "_", "-"),
    )
    steep = np.where(shade > 0.55, "#", "*")
    chars = np.where(~valid, " ", np.where(norm < 0.28, flat, np.where(norm < 1.05, slope, steep)))
    return chars.astype("<U1")


def paint(chars: np.ndarray, shade: np.ndarray, theme: str) -> str:
    prefixes = PREFIXES[theme]
    codes = np.clip((shade * (LUT_N - 1)).astype(np.int32), 0, LUT_N - 1)
    lines: list[str] = []
    h, w = chars.shape
    for y in range(h):
        row = chars[y]
        code = codes[y]
        changes = np.flatnonzero(np.diff(code, prepend=-1))
        ends = list(changes[1:]) + [w]
        parts: list[str] = []
        for start, end in zip(changes, ends):
            parts.append(prefixes[int(code[start])])
            parts.append("".join(row[start:end].tolist()))
        parts.append("\033[0m")
        lines.append("".join(parts))
    return "\n".join(lines)


def render(depth: np.ndarray, view: View, out_w: int, out_h: int, memory: list[float]) -> tuple[str, Stats]:
    grid = sample_depth(depth, out_h, out_w)
    if view.mirror:
        grid = grid[:, ::-1]
    lo, hi, valid = contrast_range(grid, view, memory)
    shade, _edge = shade_grid(grid, lo, hi, view.invert, valid)
    if view.style == "relief":
        chars = relief_chars(grid, shade, valid)
    else:
        chars = density_chars(shade, valid, RAMPS[view.ramp])
    stats = Stats(lo=lo, hi=hi, valid=float(valid.mean()) if valid.size else 0.0)
    return paint(chars, shade, view.theme), stats


def hud(view: View, stats: Stats, width: int) -> str:
    mode = "demo" if stats.demo else "live"
    title = (
        f" depth mirror  {stats.lo:.2f}-{stats.hi:.2f}m  {view.theme}  "
        f"{view.style}  {view.ramp}  {stats.fps:4.0f} fps  {mode}"
    )
    if view.paused:
        title += "  paused"
    hint = " c color  i invert  m mirror  r style  h help  q quit "
    gap = width - len(title) - len(hint)
    if gap < 1:
        line = (title + "  " + hint)[:width]
    else:
        line = title + (" " * gap) + hint
    return f"\033[48;2;14;14;24m\033[38;2;214;214;230m{line[:width]}\033[0m"


def center_line(text: str, width: int) -> str:
    text = text[:width]
    pad = max(width - len(text), 0)
    left = pad // 2
    return " " * left + text + " " * (pad - left)


def compose(picture: str, view: View, stats: Stats, term_w: int, art_h: int) -> str:
    rows = picture.split("\n") if picture else []
    blank = " " * term_w
    top = max((art_h - len(rows)) // 2, 0)
    padded = [blank] * top + rows
    if len(padded) < art_h:
        padded.extend([blank] * (art_h - len(padded)))
    parts = ["\n".join(padded[:art_h])]
    if view.show_help:
        parts.append(
            "\n".join(
                f"\033[38;2;186;186;210m{center_line(line, term_w)}\033[0m" for line in HELP_LINES
            )
        )
    parts.append(hud(view, stats, term_w))
    return "\n".join(parts)


def apply_key(view: View, key: str) -> bool:
    if key in {"q", "Q", "\x03"}:
        return True
    if key in {"c", "C"}:
        view.theme = THEME_ORDER[(THEME_ORDER.index(view.theme) + 1) % len(THEME_ORDER)]
    elif key in {"i", "I"}:
        view.invert = not view.invert
    elif key in {"m", "M"}:
        view.mirror = not view.mirror
    elif key in {"r", "R"}:
        view.style = "relief" if view.style == "density" else "density"
    elif key in {"1", "2", "3", "4"}:
        view.ramp = RAMP_ORDER[int(key) - 1]
        view.style = "density"
    elif key in {"a", "A"}:
        view.auto_contrast = not view.auto_contrast
    elif key == "[":
        view.span = max(0.4, view.span - 0.2)
    elif key == "]":
        view.span = min(6.0, view.span + 0.2)
    elif key == " ":
        view.paused = not view.paused
    elif key in {"h", "H", "?"}:
        view.show_help = not view.show_help
    return False


def read_key() -> str:
    if not sys.stdin.isatty():
        return ""
    if not select.select([sys.stdin], [], [], 0)[0]:
        return ""
    key = sys.stdin.read(1)
    if key == "\x1b":
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.read(8)
        return ""
    return key


@contextmanager
def terminal_session(enabled: bool):
    if not enabled:
        yield
        return
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        sys.stdout.write("\033[?1049h\033[?25l\033[H")
        sys.stdout.flush()
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()


def open_source(args: argparse.Namespace) -> tuple[DemoSource | CameraSource, bool]:
    if args.demo:
        return DemoSource(), True
    try:
        return CameraSource(args.width, args.height, args.fps), False
    except Exception as exc:
        if args.require_camera:
            raise SystemExit(
                "No RealSense depth camera is available "
                f"({exc}). Plug it into a USB 3 port and check `rs-enumerate-devices`."
            ) from exc
        print(
            "No RealSense depth camera found. Showing a demo scene.\n"
            "macOS can list the RGB sensor as a normal webcam while librealsense\n"
            "still cannot open the depth endpoint. Use a USB 3 port and run\n"
            "`rs-enumerate-devices` from Terminal. Pass --demo to skip this check.",
            file=sys.stderr,
        )
        return DemoSource(), True


def run(source: DemoSource | CameraSource, demo: bool, args: argparse.Namespace) -> None:
    view = View(theme=args.theme, ramp=args.ramp)
    if args.relief:
        view.style = "relief"
    memory: list[float] = []
    last_depth: np.ndarray | None = None
    stats = Stats(demo=demo)
    prev_t = time.perf_counter()
    frames = 0
    interactive = sys.stdout.isatty() and args.frames <= 0

    with terminal_session(interactive):
        while True:
            if interactive and apply_key(view, read_key()):
                break
            if not view.paused:
                try:
                    depth = source.read()
                except RuntimeError:
                    if last_depth is None:
                        raise
                    depth = None
                if depth is not None:
                    last_depth = depth
            if last_depth is None:
                time.sleep(0.02)
                continue

            term = shutil.get_terminal_size(fallback=(80, 24))
            term_w = max(40, term.columns)
            term_h = max(12, term.lines)
            extra = len(HELP_LINES) if view.show_help else 0
            art_h = max(8, term_h - 1 - extra)
            src_h, src_w = last_depth.shape
            out_w, out_h = fit_grid(term_w, art_h, src_w, src_h, args.aspect)
            picture, frame_stats = render(last_depth, view, out_w, out_h, memory)
            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                stats.fps = frame_stats.fps = (stats.fps * 0.85 + (1.0 / dt) * 0.15) if stats.fps else (1.0 / dt)
            frame_stats.demo = demo
            frame_stats.fps = stats.fps

            pad = max((term_w - out_w) // 2, 0)
            right = max(term_w - out_w - pad, 0)
            centered = "\n".join(f"{' ' * pad}{line}{' ' * right}" for line in picture.split("\n"))
            screen = compose(centered, view, frame_stats, term_w, art_h)

            if interactive or args.frames > 0:
                sys.stdout.write("\033[H" + screen)
                sys.stdout.flush()

            frames += 1
            if args.frames > 0 and frames >= args.frames:
                break
            if demo and not interactive:
                time.sleep(0.01)


def self_test() -> None:
    depth = synth_depth(0.8)
    assert depth.shape == (240, 320)
    center = float(depth[120, 160])
    corner = float(depth[8, 8])
    assert center < corner, (center, corner)

    view = View()
    memory: list[float] = []
    picture, stats = render(depth, view, 80, 36, memory)
    assert picture.count("\n") == 35
    assert "\033[38;2;" in picture
    assert stats.hi > stats.lo

    view.style = "relief"
    relief, _ = render(depth, view, 72, 28, [])
    assert any(ch in relief for ch in "/\\#")

    # Nearer shade must outrank farther shade.
    grid = np.array([[0.4, 3.0], [0.0, 1.2]], np.float32)
    shade, _ = shade_grid(grid, 0.4, 3.0, False, grid > 0)
    assert shade[0, 0] > shade[1, 1] > shade[0, 1]
    assert shade[1, 0] == 0

    t0 = time.perf_counter()
    for i in range(20):
        render(synth_depth(i * 0.05), View(), 100, 40, memory)
    elapsed = time.perf_counter() - t0
    fps = 20 / elapsed
    print(f"self-test ok  ({fps:.0f} render fps at 100x40)")
    if fps < 20:
        raise SystemExit(f"renderer is too slow: {fps:.1f} fps")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealSense depth as live ASCII art")
    parser.add_argument("--demo", action="store_true", help="animated scene, no camera")
    parser.add_argument("--require-camera", action="store_true", help="exit if no RealSense is found")
    parser.add_argument("--width", type=int, default=640, help="depth stream width")
    parser.add_argument("--height", type=int, default=480, help="depth stream height")
    parser.add_argument("--fps", type=int, default=30, help="depth stream rate")
    parser.add_argument("--theme", choices=THEME_ORDER, default="thermal")
    parser.add_argument("--ramp", choices=RAMP_ORDER, default="classic")
    parser.add_argument("--relief", action="store_true", help="start in slope-glyph style")
    parser.add_argument("--aspect", type=float, default=CHAR_ASPECT, help="terminal cell width/height")
    parser.add_argument("--frames", type=int, default=0, help="exit after N frames (0 runs forever)")
    parser.add_argument("--self-test", action="store_true", help="render the demo scene and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        self_test()
        return
    source, demo = open_source(args)
    try:
        run(source, demo, args)
    except KeyboardInterrupt:
        pass
    finally:
        source.close()


if __name__ == "__main__":
    main()
