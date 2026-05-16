import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import onnxruntime as ort

from utils import load_toml_as_dict

# Enable hardware-optimised OpenCV paths (SSE4, AVX2, NEON, etc.).
cv2.setUseOptimized(True)

warnings.filterwarnings(
    "ignore",
    message=".*'pin_memory' argument is set as true but no accelerator is found.*",
    category=UserWarning,
)

debug = load_toml_as_dict("cfg/general_config.toml")['super_debug'] == "yes"


def resolve_model_path(model_path):
    """If ``use_int8_models = "yes"`` in the config and an INT8 sibling exists
    next to ``model_path`` (e.g. ``mainInGameModel_int8.onnx`` next to
    ``mainInGameModel.onnx``), return that instead. Otherwise return the
    original path unchanged so existing setups keep working.

    INT8 models are produced by ``tools/quantize_models.py`` and are
    typically 1.4-2x faster on CPU than the FP32 originals.
    """
    cfg = load_toml_as_dict("cfg/general_config.toml")
    if cfg.get('use_int8_models', 'no') != 'yes':
        return model_path
    base, ext = os.path.splitext(model_path)
    if base.endswith('_int8'):
        return model_path
    int8_path = base + '_int8' + ext
    if os.path.isfile(int8_path):
        print(f"Using INT8 model: {int8_path}")
        return int8_path
    print(
        f"use_int8_models=yes but no {int8_path} found; falling back to FP32. "
        f"Run `python tools/quantize_models.py` to generate it."
    )
    return model_path


def get_optimal_threads():
    """Use all physical cores for inference.

    The previous implementation capped at 4 threads which left most of the
    CPU idle.  Now we default to ``cpu_count`` (logical cores) so that ONNX
    Runtime and OpenCV can saturate the machine.  Users who want to reserve
    cores for other processes can set ``onnx_threads`` in
    ``cfg/general_config.toml``.
    """
    cfg = load_toml_as_dict("cfg/general_config.toml")
    configured = cfg.get('onnx_threads', 'auto')
    total = os.cpu_count() or 2
    if configured != 'auto':
        try:
            threads_amount = max(1, int(configured))
        except (ValueError, TypeError):
            threads_amount = total
    else:
        threads_amount = total
    print(f"Detected {total} CPU threads, using {threads_amount} for inference.")
    # Let OpenCV also use all available cores for parallel primitives
    # (resize, cvtColor, matchTemplate, etc.).
    cv2.setNumThreads(threads_amount)
    return threads_amount


optimal_threads_amount = get_optimal_threads()


def _try_set_torch_threads(threads):
    """Set torch CPU thread count if torch is importable, otherwise ignore.

    Torch is only used by some downstream components; importing it lazily
    keeps the hot inference path free of torch overhead while still honoring
    the original thread-count behavior when torch happens to be installed.
    """
    try:
        import torch  # noqa: WPS433
        torch.set_num_threads(threads)
    except Exception:
        pass


_try_set_torch_threads(optimal_threads_amount)


def _numpy_nms(preds, conf_thresh, iou_thresh=0.6):
    """Fast NMS over a YOLOv8-style ONNX output (shape (1, 4+nc, n_anchors)).

    Returns a list with a single ``np.ndarray`` of shape (N, 6) where each row
    is ``[x1, y1, x2, y2, score, class_id]`` in letterboxed-input coordinate
    space, matching what the original ultralytics-based path produced.
    Returns an empty list when nothing survives filtering.
    """
    if preds is None:
        return []
    arr = preds[0]
    if arr.ndim != 2:
        return []
    arr = arr.T
    if arr.shape[1] < 5:
        return []

    boxes_xywh = arr[:, :4]
    scores_all = arr[:, 4:]

    cls_ids = scores_all.argmax(axis=1)
    cls_conf = scores_all[np.arange(scores_all.shape[0]), cls_ids]

    mask = cls_conf > conf_thresh
    if not mask.any():
        return []

    boxes_xywh = boxes_xywh[mask]
    cls_conf = cls_conf[mask].astype(np.float32, copy=False)
    cls_ids = cls_ids[mask]

    cx = boxes_xywh[:, 0]
    cy = boxes_xywh[:, 1]
    w = boxes_xywh[:, 2]
    h = boxes_xywh[:, 3]
    half_w = w * 0.5
    half_h = h * 0.5
    x1 = cx - half_w
    y1 = cy - half_h
    x2 = cx + half_w
    y2 = cy + half_h

    boxes_for_nms = np.stack([x1, y1, w, h], axis=1).astype(np.float32, copy=False)

    keep = None
    nms_batched = getattr(cv2.dnn, "NMSBoxesBatched", None)
    if nms_batched is not None:
        try:
            keep = nms_batched(
                boxes_for_nms,
                cls_conf,
                cls_ids.astype(np.int32, copy=False),
                float(conf_thresh),
                float(iou_thresh),
            )
        except cv2.error:
            keep = None

    if keep is None:
        # Older OpenCV (<4.7) lacks NMSBoxesBatched. Per-class NMS keeps the
        # class-aware behavior of agnostic=False.
        keep_list = []
        for cls in np.unique(cls_ids):
            cmask = cls_ids == cls
            cls_boxes = boxes_for_nms[cmask]
            cls_scores = cls_conf[cmask]
            cls_indices = np.where(cmask)[0]
            cls_keep = cv2.dnn.NMSBoxes(
                cls_boxes.tolist(),
                cls_scores.tolist(),
                float(conf_thresh),
                float(iou_thresh),
            )
            if len(cls_keep) == 0:
                continue
            if hasattr(cls_keep, "flatten"):
                cls_keep = cls_keep.flatten()
            keep_list.extend(int(cls_indices[i]) for i in cls_keep)
        keep = keep_list

    if keep is None or len(keep) == 0:
        return []

    if hasattr(keep, "flatten"):
        keep = keep.flatten()
    if not isinstance(keep, np.ndarray):
        keep = np.asarray(keep, dtype=np.intp)

    result = np.empty((keep.shape[0], 6), dtype=np.float32)
    result[:, 0] = x1[keep]
    result[:, 1] = y1[keep]
    result[:, 2] = x2[keep]
    result[:, 3] = y2[keep]
    result[:, 4] = cls_conf[keep]
    result[:, 5] = cls_ids[keep]
    return [result]


class Detect:
    def __init__(self, model_path, ignore_classes=None, classes=None, input_size=(640, 640)):
        self.preferred_device = load_toml_as_dict("cfg/general_config.toml")['cpu_or_gpu']
        self.model_path = resolve_model_path(model_path)
        self.classes = classes
        self.ignore_classes = ignore_classes if ignore_classes else []
        # Resolve ignored class names to ids once so we can do fast int
        # comparisons inside the inner detection loop instead of repeated
        # string lookups against ``self.ignore_classes``.
        self._ignore_class_ids = set()
        if self.classes is not None:
            for entry in self.ignore_classes:
                if isinstance(entry, int):
                    self._ignore_class_ids.add(entry)
                elif entry in self.classes:
                    self._ignore_class_ids.add(self.classes.index(entry))
        self.input_size = input_size
        self.model, self.device = self.load_model()
        # Pre-allocated CHW float32 buffer used as the ONNX input tensor. The
        # padded background is set once to a mid-grey (128/255) and only
        # repainted when the previous frame wrote a larger region than this
        # one, which keeps the per-frame cost down.
        self._padded_img_buffer = np.full(
            (1, 3, self.input_size[0], self.input_size[1]),
            128.0 / 255.0,
            dtype=np.float32,
        )
        self._last_resized_h = 0
        self._last_resized_w = 0
        # Shared thread pool for running multiple model inferences in parallel.
        # Created once per Detect instance to avoid pool-creation overhead.
        self._thread_pool = None

    @staticmethod
    def _pick_provider():
        """Select the fastest available execution provider.

        Priority: TensorRT > CUDA > OpenVINO > DirectML > CPU.
        TensorRT gives the best throughput on NVIDIA GPUs; OpenVINO
        accelerates Intel iGPUs & CPUs.
        """
        available = ort.get_available_providers()
        preference = [
            ("TensorrtExecutionProvider", "TensorRT GPU"),
            ("CUDAExecutionProvider", "CUDA GPU"),
            ("OpenVINOExecutionProvider", "OpenVINO (Intel)"),
            ("DmlExecutionProvider", "DirectML GPU"),
            ("AzureExecutionProvider", "Azure"),
        ]
        for ep, label in preference:
            if ep in available:
                print(f"Using {label} for inference")
                return ep
        print("Using CPU for inference")
        return "CPUExecutionProvider"

    def load_model(self):
        if self.preferred_device in ("gpu", "auto"):
            onnx_provider = self._pick_provider()
        else:
            onnx_provider = "CPUExecutionProvider"
            print("Using CPU for inference (forced by config)")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = optimal_threads_amount
        so.inter_op_num_threads = optimal_threads_amount
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Reduce memory allocation overhead by letting ORT reuse buffers.
        so.enable_mem_pattern = True
        so.enable_mem_reuse = True
        # If available, save the optimized graph so subsequent launches
        # skip the optimisation pass (noticeable when loading 4 models).
        cache_dir = os.path.join(os.path.dirname(self.model_path), ".ort_cache")
        os.makedirs(cache_dir, exist_ok=True)
        so.optimized_model_filepath = os.path.join(
            cache_dir,
            os.path.basename(self.model_path).replace(".onnx", "_opt.onnx"),
        )
        model = ort.InferenceSession(self.model_path, sess_options=so, providers=[onnx_provider])
        self._input_name = model.get_inputs()[0].name
        return model, onnx_provider

    def preprocess_image(self, img):
        """Letterbox-resize ``img`` into the pre-allocated CHW buffer.

        Uses a single ``np.multiply`` + ``np.transpose`` which NumPy can
        dispatch to BLAS / SIMD, then writes the result straight into the
        contiguous ONNX input tensor.
        """
        h, w = img.shape[:2]
        in_h, in_w = self.input_size
        scale = min(in_h / h, in_w / w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        buf = self._padded_img_buffer
        if new_h < self._last_resized_h or new_w < self._last_resized_w:
            buf.fill(128.0 / 255.0)

        # Fused uint8->float32 conversion + channel transpose in one pass.
        # np.multiply with casting="unsafe" does the type promotion in-place
        # and is faster than a separate astype + divide.
        buf[0, :, :new_h, :new_w] = np.multiply(
            resized_img.transpose(2, 0, 1),
            np.float32(1.0 / 255.0),
            dtype=np.float32,
            casting="unsafe",
        )

        self._last_resized_h = new_h
        self._last_resized_w = new_w
        return buf, new_w, new_h

    @staticmethod
    def postprocess(preds, orig_img_shape, resized_shape, conf_tresh=0.6):
        results = _numpy_nms(preds, conf_thresh=conf_tresh, iou_thresh=0.6)
        if not results:
            return []

        orig_h, orig_w = orig_img_shape
        resized_w, resized_h = resized_shape
        scale_w = orig_w / resized_w
        scale_h = orig_h / resized_h

        scaled = []
        for pred in results:
            if not len(pred):
                continue
            pred[:, 0] *= scale_w
            pred[:, 1] *= scale_h
            pred[:, 2] *= scale_w
            pred[:, 3] *= scale_h
            scaled.append(pred)
        return scaled

    def detect_objects(self, img, conf_tresh=0.6):
        orig_h, orig_w = img.shape[:2]
        orig_img_shape = (orig_h, orig_w)

        preprocessed_img, resized_w, resized_h = self.preprocess_image(img)
        resized_shape = (resized_w, resized_h)

        outputs = self.model.run(None, {self._input_name: preprocessed_img})

        detections = self.postprocess(outputs[0], orig_img_shape, resized_shape, conf_tresh)

        if not detections:
            return {}

        results = {}
        classes = self.classes
        ignore_ids = self._ignore_class_ids
        num_classes = len(classes) if classes else 0
        for detection in detections:
            if not len(detection):
                continue
            cls_ids = detection[:, 5].astype(np.intp)
            # Vectorized filtering: remove ignored and out-of-range classes
            if num_classes:
                valid = (cls_ids < num_classes)
                if ignore_ids:
                    for iid in ignore_ids:
                        valid &= (cls_ids != iid)
                if not valid.any():
                    continue
                detection = detection[valid]
                cls_ids = cls_ids[valid]
            else:
                continue
            xyxy_int = detection[:, :4].astype(np.int32)
            # Group by class using numpy unique
            for cid in np.unique(cls_ids):
                class_name = classes[cid]
                if class_name in self.ignore_classes:
                    continue
                mask = cls_ids == cid
                boxes = xyxy_int[mask].tolist()
                bucket = results.get(class_name)
                if bucket is None:
                    results[class_name] = boxes
                else:
                    bucket.extend(boxes)

        return results
