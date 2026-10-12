# RealSense ASCII

Live ASCII art from an Intel RealSense depth camera. Closer surfaces draw denser, brighter characters. Far surfaces fade out.

## Mac

The Mac build has to run as root. librealsense cannot open the depth sensor otherwise.

```bash
sudo -E ./run
```

`-E` keeps your current Python environment, including the `pyrealsense2` module. `./run --demo` does not need sudo. It plays a built-in scene with no camera.

## Setup

Python 3.9+ with `numpy` and the librealsense Python bindings:

```bash
python3 -m pip install -r requirements.txt
```

On this machine the bindings come from a local librealsense build. `./run` uses that interpreter when `python3` cannot import `pyrealsense2`.

Plug the camera into a USB 3 port.

## Run

```bash
sudo -E ./run                  # Mac, live camera
./run --demo                   # animated scene, no camera
python3 realsense_ascii.py --theme ocean --relief
```

## Keys

| Key | Action |
| --- | --- |
| `c` | Cycle color (thermal, ocean, matrix, amber, candy, mono) |
| `r` | Density characters or relief shading |
| `i` | Invert near and far |
| `m` | Mirror the camera |
| `1`–`4` | Character set |
| `a` | Toggle auto contrast |
| `[` `]` | Shorten or lengthen the depth window |
| space | Pause |
| `h` | Help |
| `q` | Quit |

## Options

| Flag | Meaning |
| --- | --- |
| `--demo` | Skip the camera and play the built-in scene |
| `--require-camera` | Exit if no RealSense is found |
| `--theme` | `thermal`, `ocean`, `matrix`, `amber`, `candy`, or `mono` |
| `--ramp` | `classic`, `soft`, `blocks`, or `heavy` |
| `--relief` | Start in slope-glyph style |
| `--width` `--height` `--fps` | Depth stream size (default 640×480 @ 30) |
