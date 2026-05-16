"""Quantize PylaAI's ONNX detection models to INT8 for faster CPU inference.

By default this performs *static* INT8 quantization on every ``.onnx`` file
under ``models/``, writing the result alongside the original as
``<name>_int8.onnx``. The bot will pick up the quantized variants at runtime
when ``use_int8_models = "yes"`` is set in ``cfg/general_config.toml``.

On a typical CPU INT8 inference is ~1.4-2x faster than FP32 with the trade-off
of slightly noisier detections. The accuracy drop depends a lot on how
representative the **calibration data** is of real Brawl Stars frames:

* ``--frames-dir <path>``: point at a directory containing real game frames
  (e.g. ``debug_frames/`` once you've run the bot for a minute with
  ``super_debug = "yes"`` in the general config). This is by far the most
  accurate option.
* default: synthesize calibration scenes from the bundled brawler icons and
  state textures. This is what we ship by default since it works without
  the user having to capture frames first.

Run from the repo root::

    python tools/quantize_models.py                       # synthetic calibration
    python tools/quantize_models.py --frames-dir debug_frames/   # real frames
    python tools/quantize_models.py --models models/mainInGameModel.onnx
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


REPO_ROOT = Path(__file__).resolve().parent.parent


def _letterbox(frame: np.ndarray, in_h: int, in_w: int) -> np.ndarray:
    """Letterbox-resize an HWC RGB frame into a (1, 3, in_h, in_w) float32
    tensor, matching ``detect.preprocess_image`` so the calibrator sees inputs
    in the same distribution the deployed model receives.
    """
    h, w = frame.shape[:2]
    scale = min(in_h / h, in_w / w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out = np.full((in_h, in_w, 3), 128, dtype=np.uint8)
    out[:new_h, :new_w] = resized
    chw = out.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw[np.newaxis]


def _load_real_frames(frames_dir: Path, max_frames: int) -> list[np.ndarray]:
    paths: list[Path] = []
    for ext in ("png", "PNG", "jpg", "jpeg", "JPG", "JPEG"):
        paths.extend(sorted(frames_dir.rglob(f"*.{ext}")))
    if not paths:
        raise FileNotFoundError(f"No image files found under {frames_dir}")
    if len(paths) > max_frames:
        # Even spacing across the directory keeps the calibration set
        # representative of the whole run instead of just the first frames.
        idx = np.linspace(0, len(paths) - 1, max_frames).astype(int)
        paths = [paths[i] for i in idx]
    frames = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        # Bot frames come from scrcpy in RGB but cv2.imwrite (used by
        # super_debug) writes them as BGR. Convert back to RGB before
        # calibration so the activation distribution matches inference.
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not frames:
        raise FileNotFoundError(f"No readable images under {frames_dir}")
    return frames


def _synth_frames(num_samples: int, seed: int = 42) -> list[np.ndarray]:
    """Compose 'Brawl Stars-like' scenes out of bundled icons.

    Not as good as real frames, but far closer to in-game activation
    distributions than uniform noise, which is what fully random calibration
    would give us.
    """
    icons: list[np.ndarray] = []
    icon_roots = ["images", "api/assets/brawler_icons"]
    for root in icon_roots:
        root_p = REPO_ROOT / root
        if not root_p.is_dir():
            continue
        for ext in ("png", "PNG", "jpg", "jpeg"):
            for p in root_p.rglob(f"*.{ext}"):
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                icons.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not icons:
        raise FileNotFoundError(
            "No icon textures found under images/ or api/assets/brawler_icons/."
        )

    rng = np.random.RandomState(seed)
    frames = []
    for _ in range(num_samples):
        dst_h, dst_w = 1080, 1920
        bg_top = rng.randint(40, 90, size=3, dtype=np.uint8)
        bg_bot = rng.randint(80, 160, size=3, dtype=np.uint8)
        ramp = np.linspace(0, 1, dst_h, dtype=np.float32).reshape(-1, 1, 1)
        frame = (bg_top * (1 - ramp) + bg_bot * ramp).astype(np.uint8)
        frame = np.broadcast_to(frame, (dst_h, dst_w, 3)).copy()
        for _i in range(rng.randint(6, 15)):
            icon = icons[rng.randint(0, len(icons))]
            ih, iw = icon.shape[:2]
            scale = rng.uniform(0.4, 1.4)
            new_w = max(8, int(iw * scale))
            new_h = max(8, int(ih * scale))
            # Clamp icons that ended up larger than the canvas so the paste
            # never overflows the frame array.
            new_w = min(new_w, dst_w)
            new_h = min(new_h, dst_h)
            icon_r = cv2.resize(icon, (new_w, new_h))
            x = rng.randint(0, max(1, dst_w - new_w))
            y = rng.randint(0, max(1, dst_h - new_h))
            frame[y:y + new_h, x:x + new_w] = icon_r
        frames.append(frame)
    return frames


class _CalibrationReader(CalibrationDataReader):
    def __init__(self, input_name: str, in_h: int, in_w: int, frames: list[np.ndarray]):
        self._iter = iter(
            {input_name: _letterbox(f, in_h, in_w)} for f in frames
        )

    def get_next(self):
        return next(self._iter, None)


def _quantize_one(
    src: Path,
    dst: Path,
    frames: list[np.ndarray],
    per_channel: bool,
) -> None:
    preproc = src.with_suffix(".preproc.onnx")
    quant_pre_process(str(src), str(preproc), skip_symbolic_shape=False)
    try:
        sess = ort.InferenceSession(str(preproc), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        shape = [1 if not isinstance(d, int) else d for d in inp.shape]
        in_h, in_w = int(shape[2]), int(shape[3])
        del sess

        reader = _CalibrationReader(inp.name, in_h, in_w, frames)
        quantize_static(
            model_input=str(preproc),
            model_output=str(dst),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QOperator,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            reduce_range=False,
            calibrate_method=CalibrationMethod.Percentile,
            extra_options={
                "CalibPercentile": 99.99,
                "CalibMovingAverage": True,
            },
        )
    finally:
        if preproc.exists():
            preproc.unlink()


def _bench(src: Path, dst: Path, frames: list[np.ndarray]) -> tuple[float, float]:
    """Return (fp32_ms, int8_ms) using one of the calibration frames."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_f = ort.InferenceSession(str(src), sess_options=so, providers=["CPUExecutionProvider"])
    sess_q = ort.InferenceSession(str(dst), sess_options=so, providers=["CPUExecutionProvider"])
    inp = sess_f.get_inputs()[0]
    shape = [1 if not isinstance(d, int) else d for d in inp.shape]
    x = _letterbox(frames[0], int(shape[2]), int(shape[3])).astype(np.float32)
    feeds = {inp.name: x}
    for _ in range(2):
        sess_f.run(None, feeds)
        sess_q.run(None, feeds)
    n = 10
    t = time.perf_counter()
    for _ in range(n):
        sess_f.run(None, feeds)
    t_f = (time.perf_counter() - t) / n * 1000
    t = time.perf_counter()
    for _ in range(n):
        sess_q.run(None, feeds)
    t_q = (time.perf_counter() - t) / n * 1000
    return t_f, t_q


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Quantize PylaAI ONNX models to INT8 with static calibration."
    )
    parser.add_argument(
        "--models-dir",
        default=str(REPO_ROOT / "models"),
        help="Directory containing the *.onnx models (default: ./models).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Specific model files to quantize. Default: every *.onnx in --models-dir "
        "that doesn't already end in '_int8.onnx'.",
    )
    parser.add_argument(
        "--frames-dir",
        default=None,
        help="Optional directory of real game frames (PNG/JPG) for calibration. "
        "When omitted, synthetic scenes built from bundled icons are used.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
        help="Number of calibration samples to use (default: 32).",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Use per-channel weight quantization (sometimes better accuracy, "
        "but onnxruntime has shape bugs on some YOLOv8 exports).",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Also benchmark FP32 vs INT8 latency after each quantization.",
    )
    args = parser.parse_args(argv)

    models_dir = Path(args.models_dir).resolve()
    if args.models:
        sources = [Path(p).resolve() for p in args.models]
    else:
        sources = sorted(
            p for p in models_dir.glob("*.onnx") if not p.name.endswith("_int8.onnx")
        )
    if not sources:
        print(f"No .onnx files found in {models_dir}", file=sys.stderr)
        return 1

    if args.frames_dir:
        frames_dir = Path(args.frames_dir).resolve()
        print(f"Loading real frames from {frames_dir} ...")
        frames = _load_real_frames(frames_dir, args.num_samples)
        print(f"  loaded {len(frames)} frame(s)")
    else:
        print(f"Synthesizing {args.num_samples} calibration scene(s) from bundled icons ...")
        frames = _synth_frames(args.num_samples)
        print(
            "  NOTE: synthetic calibration is a stopgap. For best accuracy run the bot "
            "for ~1 minute with super_debug=\"yes\", then re-run with --frames-dir debug_frames/."
        )

    for src in sources:
        dst = src.with_name(src.stem + "_int8.onnx")
        print(f"== {src.name} -> {dst.name} ==")
        _quantize_one(src, dst, frames, per_channel=args.per_channel)
        fp_mb = src.stat().st_size / 1e6
        q_mb = dst.stat().st_size / 1e6
        print(f"  size: {fp_mb:.1f} MB -> {q_mb:.1f} MB ({q_mb / fp_mb:.2f}x)")
        if args.bench:
            t_f, t_q = _bench(src, dst, frames)
            speedup = t_f / t_q if t_q > 0 else 0
            print(f"  speed: fp32 ~{t_f:.1f} ms vs int8 ~{t_q:.1f} ms ({speedup:.2f}x)")

    print("Done. Set use_int8_models = \"yes\" in cfg/general_config.toml to use them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
