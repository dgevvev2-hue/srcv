import os
import warnings

import cv2
import numpy as np
import onnxruntime as ort

from utils import load_toml_as_dict

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


def get_optimal_threads(max_limit=4):
    threads = os.cpu_count() or 2
    threads_amount = min(max(2, threads // 2), max_limit)
    print(f"Detected {threads} CPU threads, using {threads_amount} threads.")
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
    keep = np.asarray(list(keep), dtype=np.int64)

    result = np.empty((keep.shape[0], 6), dtype=np.float32)
    result[:, 0] = x1[keep]
    result[:, 1] = y1[keep]
    result[:, 2] = x2[keep]
    result[:, 3] = y2[keep]
    result[:, 4] = cls_conf[keep]
    result[:, 5] = cls_ids[keep].astype(np.float32, copy=False)
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
        # Reusable float32 staging buffer for the resized image (HWC).
        self._resize_buffer = None

    def load_model(self):
        available_providers = ort.get_available_providers()
        onnx_provider = "CPUExecutionProvider"
        if self.preferred_device == "gpu" or self.preferred_device == "auto":
            if "CUDAExecutionProvider" in available_providers:
                onnx_provider = "CUDAExecutionProvider"
                print("Using CUDA GPU")
            elif "DmlExecutionProvider" in available_providers:
                onnx_provider = "DmlExecutionProvider"
                print("Using GPU")
            elif "AzureExecutionProvider" in available_providers:
                onnx_provider = "AzureExecutionProvider"
            else:
                print("Using CPU as no GPU provider found")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = optimal_threads_amount
        so.inter_op_num_threads = optimal_threads_amount
        # Sequential execution is generally faster for single-image inference;
        # parallel execution adds inter-op pool overhead with no benefit here.
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        model = ort.InferenceSession(self.model_path, sess_options=so, providers=[onnx_provider])
        self._input_name = model.get_inputs()[0].name
        return model, onnx_provider

    def preprocess_image(self, img):
        """Letterbox-resize ``img`` into the pre-allocated CHW buffer.

        Avoids creating intermediate full-frame copies via ``np.transpose``
        and a separate ``astype``/divide by writing each channel directly
        into the contiguous output buffer with a single fused multiply.
        """
        h, w = img.shape[:2]
        in_h, in_w = self.input_size
        scale = min(in_h / h, in_w / w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if (
            self._resize_buffer is None
            or self._resize_buffer.shape[0] < new_h
            or self._resize_buffer.shape[1] < new_w
        ):
            self._resize_buffer = np.empty((max(new_h, in_h), max(new_w, in_w), 3), dtype=np.float32)
        np.multiply(
            resized_img,
            np.float32(1.0 / 255.0),
            out=self._resize_buffer[:new_h, :new_w],
            dtype=np.float32,
            casting="unsafe",
        )

        buf = self._padded_img_buffer
        if new_h < self._last_resized_h or new_w < self._last_resized_w:
            buf.fill(128.0 / 255.0)

        src = self._resize_buffer[:new_h, :new_w]
        buf[0, 0, :new_h, :new_w] = src[:, :, 0]
        buf[0, 1, :new_h, :new_w] = src[:, :, 1]
        buf[0, 2, :new_h, :new_w] = src[:, :, 2]

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

        results = {}
        classes = self.classes
        ignore_ids = self._ignore_class_ids
        for detection in detections:
            xyxy_int = detection[:, :4].astype(np.int32, copy=False)
            cls_ids = detection[:, 5].astype(np.int32, copy=False)
            for i in range(detection.shape[0]):
                class_id = int(cls_ids[i])
                if class_id in ignore_ids:
                    continue
                if classes is None or class_id >= len(classes):
                    continue
                class_name = classes[class_id]
                if class_name in self.ignore_classes:
                    continue
                bucket = results.get(class_name)
                if bucket is None:
                    bucket = []
                    results[class_name] = bucket
                bucket.append(
                    [int(xyxy_int[i, 0]), int(xyxy_int[i, 1]), int(xyxy_int[i, 2]), int(xyxy_int[i, 3])]
                )

        return results
