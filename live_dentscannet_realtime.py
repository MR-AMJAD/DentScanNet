# live_dentscannet_realtime.py
# Real-time periodontal ultrasound annotation with DentScanNet.
# Requires trained checkpoint and Elgato HD60 X (or compatible capture card).
# Controls: q=quit  s=screenshot

import os

# ── GPU / threading env vars (set before importing TF) ──────────────────────
os.environ['TF_GPU_ALLOCATOR']       = 'cuda_malloc_async'
os.environ['TF_ENABLE_ONEDNN_OPTS']  = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL']   = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_CACHE_DISABLE']     = '0'
os.environ['OMP_NUM_THREADS']        = '2'

import cv2
import numpy as np
import tensorflow as tf
import argparse
import time
import threading
import queue
import gc
from collections import deque
from datetime import datetime

cv2.setNumThreads(2)

from model_dentscannet import (
    CUSTOM_OBJECTS,
    ALL_FEATURES,
    POINT_FEATURES,
    REGION_FEATURES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CLASSES,
    PIXELS_PER_MM,
)

FEATURE_COLORS = {
    'GM'      : (0,   255, 255),  # yellow  BGR (paper Fig. 4)
    'CEJ'     : (255,   0,   0),  # blue    BGR (paper Fig. 4)
    'ABC'     : (255,   0, 255),  # magenta BGR (paper Fig. 4)
    'GINGIVA' : (100, 100, 255),  # red-ish
    'TOOTH'   : (255, 200, 200),  # light blue
    'BONE'    : (100, 255, 100),  # green
}

CLINICAL_COLORS = {
    'iGR' : (0,   255, 255),  # yellow  — CEJ to GM
    'iGH' : (0,   165, 255),  # orange  — GM to ABC
    'iABL': (0,     0, 255),  # red     — CEJ to ABC
}

# Per-feature inference thresholds (tuned on validation set)
THRESHOLDS = {
    'GM'      : 0.18,
    'CEJ'     : 0.28,
    'ABC'     : 0.20,
    'GINGIVA' : 0.24,
    'TOOTH'   : 0.30,
    'BONE'    : 0.08,
}

class HD60XCapture:
    """Low-latency frame capture for Elgato HD60 X."""

    def __init__(self, device_id=0,
                 crop_x=800, crop_y=220,
                 crop_width=610, crop_height=640):
        self.device_id   = device_id
        self.crop_x      = crop_x
        self.crop_y      = crop_y
        self.crop_width  = crop_width
        self.crop_height = crop_height
        self.cap              = None
        self.is_running       = False
        self.frame_queue      = queue.Queue(maxsize=1)
        self.capture_thread   = None
        self.last_frame       = None

    def start(self):
        print(f"Initializing HD60 X (device {self.device_id})...")

        # Try MSMF first (best for HD60 X on Windows), fall back to DSHOW
        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            print("  MSMF unavailable, trying DSHOW...")
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open capture device {self.device_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1924)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  980)
        self.cap.set(cv2.CAP_PROP_FPS,            60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,      1)

        # Prefer uncompressed YUY2 for lowest latency; fall back to MJPG
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc(*'YUY2'))
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("  YUY2 unavailable, trying MJPG...")
            self.cap.set(cv2.CAP_PROP_FOURCC,
                         cv2.VideoWriter_fourcc(*'MJPG'))
            ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("Cannot read frames from capture device")

        self.last_frame = frame

        w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        # Clamp crop to actual frame size
        self.crop_width  = min(self.crop_width,  w - self.crop_x)
        self.crop_height = min(self.crop_height, h - self.crop_y)

        print(f"  Capture: {w}x{h} @ {fps} FPS")
        print(f"  Crop:    {self.crop_width}x{self.crop_height} "
              f"at ({self.crop_x}, {self.crop_y})")

        self.is_running = True
        self.capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        """Background thread: grab → crop → push to queue."""
        failures = 0
        while self.is_running and failures < 30:
            if not self.cap.grab():
                failures += 1
                continue
            ret, frame = self.cap.retrieve()
            if not ret or frame is None:
                failures += 1
                continue
            failures = 0
            cropped = frame[self.crop_y : self.crop_y + self.crop_height,
                            self.crop_x : self.crop_x + self.crop_width]
            self.last_frame = cropped
            # Keep queue at size-1: discard stale frame, push fresh one
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            try:
                self.frame_queue.put_nowait(cropped)
            except queue.Full:
                pass
        if failures >= 30:
            print("HD60 X connection lost")
            self.is_running = False

    def get_frame(self):
        try:
            return self.frame_queue.get(timeout=0.05)
        except queue.Empty:
            return self.last_frame

    def stop(self):
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

class DentScanNetAnnotator:
    """Per-frame inference and annotation for real-time periodontal ultrasound."""

    def __init__(self, model_path,
                 features=None,
                 show_clinical=True,
                 pixels_per_mm=PIXELS_PER_MM):
        self.model_path    = model_path
        self.features      = features if features else ALL_FEATURES
        self.show_clinical = show_clinical
        self.pixels_per_mm = pixels_per_mm

        self.model      = None
        self.predict_fn = None

        # Caches to reduce per-frame computation
        self.point_coords_cache       = {}
        self.measurements_cache       = {}
        self.measurements_counter     = 0
        self.measurements_interval    = 3   # recalculate every N frames
        self.overlay_cache            = None
        self.overlay_counter          = 0
        self.overlay_interval         = 2   # rebuild overlay every N frames

        self.perf = {'fps': 0.0, 'inference_ms': 0.0,
                     'latency_ms': 0.0, 'frames': 0}

        self._load_model()

    def _load_model(self):
        print(f"\nLoading DentScanNet from: {self.model_path}")

        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False
        )
        print("  Model loaded")

        # Wrap in tf.function for optimised GPU inference
        @tf.function(reduce_retracing=True)
        def _predict(x):
            return self.model(x, training=False)

        self.predict_fn = _predict

        # Warmup: compile the graph before live use
        print("  Warming up...")
        dummy = tf.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.float32)
        for _ in range(3):
            self.predict_fn(dummy)
        print("  Ready\n")

    def preprocess(self, frame):
        """Convert BGR to RGB, resize to 256x256, normalise, add batch dim."""
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized   = cv2.resize(rgb, (IMAGE_WIDTH, IMAGE_HEIGHT),
                               interpolation=cv2.INTER_LINEAR)
        normalised = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalised, axis=0)

    def predict(self, frame_batch):

        return self.predict_fn(frame_batch)

    def post_process(self, predictions, target_shape):
        # channel 1 = foreground probability for each of the 6 outputs
        processed = {}
        self.point_coords_cache.clear()

        # Map feature name to its fixed output index (ALL_FEATURES order).
        # Keeps predictions correct even when --features is a subset.
        feature_to_index = {feat: i for i, feat in enumerate(ALL_FEATURES)}

        try:
            for feature in self.features:
                if feature not in feature_to_index:
                    print(f"  Unknown feature skipped: {feature}")
                    continue

                pred     = predictions[feature_to_index[feature]].numpy()[0]
                prob_map = pred[..., 1] if pred.ndim == 3 else pred

                prob_resized = cv2.resize(
                    prob_map, (target_shape[1], target_shape[0]),
                    interpolation=cv2.INTER_LINEAR)

                threshold = THRESHOLDS.get(feature, 0.25)
                mask = (prob_resized > threshold).astype(np.uint8)
                processed[feature] = mask

                if feature in POINT_FEATURES and np.any(mask):
                    ys, xs = np.where(mask)
                    self.point_coords_cache[feature] = (
                        int(xs.mean()), int(ys.mean()))

        except Exception as e:
            print(f"  Post-process error: {e}")

        return processed

    def _compute_measurements(self):
        # iGH = GM-ABC, iGR = CEJ-GM, iABL = CEJ-ABC  (Fig. 5)
        cache = self.point_coords_cache
        if not all(k in cache for k in ('GM', 'CEJ', 'ABC')):
            return {}

        cej_x, cej_y = cache['CEJ']
        gm_x,  gm_y  = cache['GM']
        abc_x, abc_y = cache['ABC']

        def dist_px(ax, ay, bx, by):
            return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

        import math
        return {
            'iGH' : dist_px(gm_x,  gm_y,  abc_x, abc_y) / self.pixels_per_mm,
            'iGR' : dist_px(cej_x, cej_y, gm_x,  gm_y)  / self.pixels_per_mm,
            'iABL': dist_px(cej_x, cej_y, abc_x, abc_y)  / self.pixels_per_mm,
        }

    def _get_measurements_throttled(self):
        self.measurements_counter += 1
        if (self.measurements_counter >= self.measurements_interval
                or not self.measurements_cache):
            self.measurements_cache  = self._compute_measurements()
            self.measurements_counter = 0
        return self.measurements_cache

    def _build_region_overlay(self, processed, target_shape):
        overlay = np.zeros((*target_shape, 3), dtype=np.uint8)
        for feat in REGION_FEATURES:
            if feat not in processed:
                continue
            mask = processed[feat]
            if mask.ndim > 2:
                mask = mask.squeeze()
            color = FEATURE_COLORS.get(feat, (128, 128, 128))
            if np.any(mask):
                overlay[mask > 0] = color
        return overlay

    def _draw_landmarks(self, frame):
        """Draw filled circles and labels for GM, CEJ, ABC."""
        for feat in POINT_FEATURES:
            if feat not in self.point_coords_cache:
                continue
            cx, cy = self.point_coords_cache[feat]
            color  = FEATURE_COLORS.get(feat, (255, 255, 255))
            cv2.circle(frame, (cx, cy), 15, color,     -1)  # filled dot
            cv2.circle(frame, (cx, cy), 17, (255, 255, 255), 2)  # white ring
            cv2.putText(frame, feat, (cx - 20, cy - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_measurement_lines(self, frame, measurements):
        """Draw connecting lines between landmark pairs for each measurement."""
        cache = self.point_coords_cache

        # iGR: CEJ → GM
        if 'iGR' in measurements and 'CEJ' in cache and 'GM' in cache:
            cv2.line(frame, cache['CEJ'], cache['GM'],
                     CLINICAL_COLORS['iGR'], 2)
            mid = ((cache['CEJ'][0] + cache['GM'][0]) // 2,
                   (cache['CEJ'][1] + cache['GM'][1]) // 2)
            cv2.putText(frame, f"iGR {measurements['iGR']:.2f}mm",
                        (mid[0] + 8, mid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        CLINICAL_COLORS['iGR'], 2)

        # iGH: GM → ABC
        if 'iGH' in measurements and 'GM' in cache and 'ABC' in cache:
            cv2.line(frame, cache['GM'], cache['ABC'],
                     CLINICAL_COLORS['iGH'], 2)
            mid = ((cache['GM'][0]  + cache['ABC'][0]) // 2,
                   (cache['GM'][1]  + cache['ABC'][1]) // 2)
            cv2.putText(frame, f"iGH {measurements['iGH']:.2f}mm",
                        (mid[0] + 8, mid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        CLINICAL_COLORS['iGH'], 2)

        # iABL: CEJ → ABC
        if 'iABL' in measurements and 'CEJ' in cache and 'ABC' in cache:
            cv2.line(frame, cache['CEJ'], cache['ABC'],
                     CLINICAL_COLORS['iABL'], 2)
            mid = ((cache['CEJ'][0] + cache['ABC'][0]) // 2,
                   (cache['CEJ'][1] + cache['ABC'][1]) // 2)
            cv2.putText(frame, f"iABL {measurements['iABL']:.2f}mm",
                        (mid[0] + 8, mid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        CLINICAL_COLORS['iABL'], 2)

    def _draw_measurements_panel(self, frame, measurements):
        """Semi-transparent panel showing iGH, iGR, iABL values."""
        if not measurements:
            return
        h, w  = frame.shape[:2]
        ph, pw = 110, 280
        x0    = w - pw - 10
        y0    = h - ph - 10

        roi    = frame[y0:y0+ph, x0:x0+pw]
        bg     = np.zeros_like(roi)
        roi[:] = cv2.addWeighted(roi, 0.65, bg, 0.35, 0)

        y = 22
        cv2.putText(roi, "Clinical Indices", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        y += 26
        for key in ('iGH', 'iGR', 'iABL'):
            if key in measurements:
                cv2.putText(roi, f"{key}: {measurements[key]:.2f} mm",
                            (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, CLINICAL_COLORS[key], 2)
                y += 24

    def _draw_perf_panel(self, frame):
        """Semi-transparent performance panel (top-left)."""
        roi = frame[10:115, 10:320]
        bg  = np.zeros_like(roi)
        roi[:] = cv2.addWeighted(roi, 0.8, bg, 0.2, 0)

        fps_col = (0, 255, 0) if self.perf['fps'] >= 25 else (0, 165, 255)
        cv2.putText(roi, f"FPS:       {self.perf['fps']:.1f}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_col, 2)
        cv2.putText(roi, f"Inference: {self.perf['inference_ms']:.0f} ms",
                    (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 255, 255), 1)
        cv2.putText(roi, f"Latency:   {self.perf['latency_ms']:.0f} ms",
                    (8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 255, 255), 1)
        cv2.putText(roi, f"Frames:    {self.perf['frames']}",
                    (8, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 200, 200), 1)
        cv2.putText(roi, "DentScanNet | ",
                    (8, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (160, 160, 160), 1)

    def annotate(self, frame, processed):

        out = frame.copy()


        self.overlay_counter += 1
        if self.overlay_counter >= self.overlay_interval or self.overlay_cache is None:
            self.overlay_cache   = self._build_region_overlay(
                processed, frame.shape[:2])
            self.overlay_counter = 0
        if self.overlay_cache is not None:
            out = cv2.addWeighted(out, 0.75, self.overlay_cache, 0.25, 0)


        self._draw_landmarks(out)


        if self.show_clinical:
            measurements = self._get_measurements_throttled()
            self._draw_measurement_lines(out, measurements)
            self._draw_measurements_panel(out, measurements)


        self._draw_perf_panel(out)

        return out

    def run(self, capture):
        print(f"DentScanNet live  |  model: {self.model_path}")
        print(f"features: {self.features}  |  q=quit  s=screenshot")

        cv2.namedWindow("DentScanNet — Real-Time", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("DentScanNet — Real-Time", 1200, 800)

        fps_counter  = 0
        fps_t0       = time.time()
        latency_buf  = deque(maxlen=30)
        stale_frames = 0

        try:
            while True:
                t_frame = time.time()

                frame = capture.get_frame()
                if frame is None:
                    stale_frames += 1
                    if stale_frames > 100:
                        print("Connection lost — exiting")
                        break
                    time.sleep(0.01)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                stale_frames = 0
                out_frame = frame.copy()

                try:

                    t_infer  = time.time()
                    batch    = self.preprocess(frame)
                    preds    = self.predict(batch)
                    infer_ms = (time.time() - t_infer) * 1000


                    processed = self.post_process(preds, frame.shape[:2])
                    out_frame = self.annotate(frame, processed)


                    lat_ms = (time.time() - t_frame) * 1000
                    latency_buf.append(lat_ms)

                    self.perf['inference_ms'] = infer_ms
                    self.perf['latency_ms']   = sum(latency_buf) / len(latency_buf)
                    self.perf['frames']      += 1

                    fps_counter += 1
                    now = time.time()
                    if now - fps_t0 >= 1.0:
                        self.perf['fps'] = fps_counter / (now - fps_t0)
                        fps_counter      = 0
                        fps_t0           = now
                        print(f"  FPS: {self.perf['fps']:.1f}  "
                              f"Inference: {infer_ms:.0f} ms  "
                              f"Latency: {self.perf['latency_ms']:.0f} ms")

                    cv2.imshow("DentScanNet — Real-Time", out_frame)

                except Exception as e:
                    print(f"  Frame error: {e}")
                    cv2.imshow("DentScanNet — Real-Time", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    fname = f"dentscannet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(fname, out_frame)
                    print(f"  Saved: {fname}")

        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            cv2.destroyAllWindows()
            print(f"\nSession complete — {self.perf['frames']} frames  "
                  f"avg {self.perf['fps']:.1f} FPS")

def main():
    import math   # needed inside _compute_measurements; also ensure available here

    parser = argparse.ArgumentParser(
        description='DentScanNet — Real-Time Video Annotation '
                    '(Amjadian et al.)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Architecture: MKIR Encoder + GCAM + LFCR Decoder
Custom layers: MKIRBlock, MKIRABlock, GCAMBlock, LFGate, CReFGate

Examples
────────
  python live_dentscannet_realtime.py --model_path DentScanNet_best.h5

  python live_dentscannet_realtime.py \\
      --model_path DentScanNet_best.h5 \\
      --device_id 1 \\
      --crop_x 800 --crop_y 220 --crop_width 610 --crop_height 640

  python live_dentscannet_realtime.py \\
      --model_path DentScanNet_best.h5 --disable_clinical

Controls
────────
  q  quit
  s  save screenshot
        """)

    parser.add_argument('--model_path',      type=str, required=True,
                        help='Path to trained DentScanNet checkpoint (.h5)')
    parser.add_argument('--device_id',       type=int, default=1,
                        help='Video capture device ID (default: 1)')
    parser.add_argument('--crop_x',          type=int, default=800)
    parser.add_argument('--crop_y',          type=int, default=220)
    parser.add_argument('--crop_width',      type=int, default=610)
    parser.add_argument('--crop_height',     type=int, default=640)
    parser.add_argument('--pixels_per_mm', type=float, default=PIXELS_PER_MM,
                        help=f'Pixels per mm in the captured/cropped frame. '
                             f'System-specific — measure with a calibration '
                             f'phantom before clinical use. '
                             f'Default: {PIXELS_PER_MM} (see README).')
    parser.add_argument('--disable_clinical', action='store_true',
                        help='Disable clinical measurement overlay')
    parser.add_argument('--features',        type=str, nargs='+', default=None,
                        help='Subset of features to display (default: all 6)')

    args = parser.parse_args()

    print("DentScanNet — real-time periodontal annotation")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU: {len(gpus)} device(s) detected")
    else:
        print("WARNING: No GPU detected — performance will be limited")
        if input("Continue on CPU? (y/n): ").strip().lower() != 'y':
            return

    capture   = None
    annotator = None

    try:

        print("\n[1/3] Initialising capture...")
        capture = HD60XCapture(
            device_id   = args.device_id,
            crop_x      = args.crop_x,
            crop_y      = args.crop_y,
            crop_width  = args.crop_width,
            crop_height = args.crop_height,
        )
        capture.start()
        time.sleep(2)   # let capture stabilise

        test = capture.get_frame()
        if test is None or test.size == 0:
            print("ERROR: cannot read frames from capture device")
            return
        print(f"  Capture OK: {test.shape}")


        print("\n[2/3] Loading DentScanNet model...")
        annotator = DentScanNetAnnotator(
            model_path   = args.model_path,
            features     = args.features,
            show_clinical = not args.disable_clinical,
            pixels_per_mm = args.pixels_per_mm,
        )


        print("\n[3/3] Starting real-time annotation...")
        annotator.run(capture)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if capture:
            capture.stop()
        cv2.destroyAllWindows()
        gc.collect()

if __name__ == '__main__':
    main()
