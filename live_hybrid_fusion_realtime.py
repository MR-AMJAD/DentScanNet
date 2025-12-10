# live_hybrid_fusion_realtime.py
# Live video annotation for HYBRID FUSION model
# Optimized for Elgato HD60 X capture card with Hybrid Cross Fusion architecture

import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_CACHE_DISABLE'] = '0'
os.environ['OMP_NUM_THREADS'] = '2'

import cv2
import numpy as np
import tensorflow as tf
import argparse
import time
from collections import deque
from datetime import datetime
import threading
import queue
import math
import gc

cv2.setNumThreads(2)

# Import model components
from train_hybrid_production import (
    HybridCrossFusion as _HybridCrossFusion,
    build_hybrid_fusion_model,
    ALL_FEATURES,
    POINT_FEATURES,
    REGION_FEATURES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CLASSES,
    dice_coefficient_metric
)

from improved_architectural_components import (
    SelectivePyramidPooling as _SelectivePyramidPooling,
)

# Wrappers to handle 'trainable' argument during deserialization
@tf.keras.utils.register_keras_serializable(package="Custom")
class SelectivePyramidPooling(_SelectivePyramidPooling):
    @classmethod
    def from_config(cls, config):
        config.pop('trainable', None)  # Remove trainable if present
        config.pop('dtype', None)  # Remove dtype if present
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="Custom")
class HybridCrossFusion(_HybridCrossFusion):
    @classmethod
    def from_config(cls, config):
        config.pop('trainable', None)  # Remove trainable if present
        config.pop('dtype', None)  # Remove dtype if present
        return cls(**config)

# Feature colors for visualization
FEATURE_COLORS = {
    'GINGIVA': (255, 100, 100),
    'TOOTH': (200, 200, 255),
    'BONE': (100, 255, 100),
    'CEJ': (255, 255, 0),
    'ABC': (255, 0, 255),
    'GM': (0, 255, 255),
}

CLINICAL_COLORS = {
    'iGR': (0, 255, 255),
    'iGH': (0, 165, 255),
    'iABL': (0, 0, 255),
}


class HD60XOptimizedCapture:
    """Optimized for Elgato HD60 X capture card"""
    def __init__(self, device_id=0, crop_x=800, crop_y=220, crop_width=610, crop_height=640):
        self.device_id = device_id
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=1)  # Minimal buffer
        self.capture_thread = None
        self.last_frame = None
        
    def start_capture(self):
        print(f"Initializing HD60 X (device {self.device_id})...")
        
        # Try MSMF first (best for HD60 X on Windows)
        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            print("MSMF failed, trying DSHOW...")
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open HD60 X on device {self.device_id}")
        
        # HD60 X optimal settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1924)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 980)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Try uncompressed format for lower latency
        yuv2_codec = cv2.VideoWriter_fourcc(*'YUY2')
        self.cap.set(cv2.CAP_PROP_FOURCC, yuv2_codec)
        
        # If YUY2 fails, try MJPG
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            print("YUY2 failed, trying MJPG...")
            mjpg_codec = cv2.VideoWriter_fourcc(*'MJPG')
            self.cap.set(cv2.CAP_PROP_FOURCC, mjpg_codec)
            ret, test_frame = self.cap.read()
        
        if ret and test_frame is not None:
            self.last_frame = test_frame
            print("HD60 X initialized")
        else:
            raise ValueError("Cannot read frames from HD60 X")
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Capture: {width}x{height} @ {fps} FPS")
        
        # Validate crop
        if self.crop_x + self.crop_width > width:
            self.crop_width = width - self.crop_x
        if self.crop_y + self.crop_height > height:
            self.crop_height = height - self.crop_y
        
        print(f"Crop: {self.crop_width}x{self.crop_height}")
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._aggressive_capture_loop, daemon=True)
        self.capture_thread.start()
    
    def _aggressive_capture_loop(self):
        """Aggressively grab frames to minimize latency"""
        consecutive_failures = 0
        
        while self.is_running and consecutive_failures < 30:
            ret = self.cap.grab()
            if not ret:
                consecutive_failures += 1
                continue
            
            ret, frame = self.cap.retrieve()
            if not ret or frame is None:
                consecutive_failures += 1
                continue
            
            consecutive_failures = 0
            
            # Crop immediately
            cropped = frame[self.crop_y:self.crop_y+self.crop_height, 
                          self.crop_x:self.crop_x+self.crop_width]
            
            self.last_frame = cropped
            
            # Clear queue aggressively
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    break
            
            try:
                self.frame_queue.put(cropped, block=False)
            except:
                pass
        
        if consecutive_failures >= 30:
            print("HD60 X connection lost")
            self.is_running = False
    
    def get_frame(self):
        try:
            return self.frame_queue.get(timeout=0.05)
        except queue.Empty:
            return self.last_frame
    
    def stop_capture(self):
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break


class RealTimeHybridFusionAnnotator:
    """Real-time annotator for Hybrid Fusion model"""
    
    def __init__(self, model_path, features=None, show_clinical_measurements=True, 
                 pixels_per_mm=23.2):
        self.model_path = model_path
        self.show_clinical_measurements = show_clinical_measurements
        self.pixels_per_mm = pixels_per_mm
        
        self.features = features if features else ALL_FEATURES
        self.model = None
        self.predict_fn = None
        
        self.measurements_cache = None
        self.measurements_frame_counter = 0
        self.measurements_update_interval = 3
        
        self.overlay_cache = None
        self.overlay_frame_counter = 0
        self.overlay_update_interval = 2
        
        self.point_coords_cache = {}
        
        self.performance_stats = {
            'fps': 0,
            'inference_time': 0,
            'avg_latency': 0,
            'total_frames': 0,
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load the Hybrid Fusion model"""
        print(f"\nLoading Hybrid Fusion model from: {self.model_path}")
        
        # Register custom objects
        custom_objects = {
            'HybridCrossFusion': HybridCrossFusion,
            'SelectivePyramidPooling': SelectivePyramidPooling,
            'dice_coefficient_metric': dice_coefficient_metric,
        }
        
        try:
            # Load model with custom objects
            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects=custom_objects,
                compile=False
            )
            print("✓ Model loaded successfully")
            
            # Create optimized predict function with tf.function
            print("Creating optimized inference function...")
            @tf.function(reduce_retracing=True)
            def predict_fn(x):
                return self.model(x, training=False)
            
            self.predict_fn = predict_fn
            
            # Warmup with multiple calls to compile the graph
            print("Warming up model...")
            dummy_input = tf.random.normal((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
            for _ in range(3):
                _ = self.predict_fn(dummy_input)
            print("✓ Model ready")
            
        except Exception as e:
            print(f"ERROR loading model: {e}")
            raise
    
    def preprocess_frame_fast(self, frame):
        """Fast preprocessing for real-time inference"""
        try:
            # Resize to model input size
            resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT), 
                               interpolation=cv2.INTER_LINEAR)
            
            # Normalize
            normalized = resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            batch = np.expand_dims(normalized, axis=0)
            
            return batch
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def predict_single_frame(self, frame_batch):
        """Run inference on a single frame"""
        try:
            predictions = self.predict_fn(frame_batch)
            return predictions
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def post_process_predictions(self, predictions, target_shape):
        """Post-process predictions to target shape"""
        processed = {}
        
        if predictions is None:
            return processed
        
        # Feature-specific thresholds (from working code)
        thresholds = {
            'GINGIVA': 0.24, 'TOOTH': 0.30, 'BONE': 0.08,
            'GM': 0.18, 'CEJ': 0.28, 'ABC': 0.20
        }
        
        try:
            # Clear all point coordinates at start of each frame
            for feature in POINT_FEATURES:
                if feature in self.point_coords_cache:
                    del self.point_coords_cache[feature]
            
            for i, feature in enumerate(self.features):
                if i < len(predictions):
                    pred = predictions[i].numpy()[0]  # Remove batch dimension
                    
                    # Extract probability map - use channel 1 if multi-channel
                    if pred.ndim == 3 and pred.shape[-1] >= 2:
                        prob_map = pred[:, :, 1]  # Use channel 1 (class probability)
                    elif pred.ndim == 3:
                        prob_map = pred[:, :, 0]
                    else:
                        prob_map = pred
                    
                    # Resize to target shape
                    resized = cv2.resize(prob_map, (target_shape[1], target_shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
                    
                    # Apply feature-specific threshold
                    threshold = thresholds.get(feature, 0.25)
                    mask = (resized > threshold).astype(np.uint8)
                    
                    processed[feature] = mask
                    
                    # Update point coordinates ONLY if detected for point features
                    if feature in POINT_FEATURES:
                        if np.any(mask > 0):
                            y_coords, x_coords = np.where(mask > 0)
                            cy = int(np.mean(y_coords))
                            cx = int(np.mean(x_coords))
                            self.point_coords_cache[feature] = (cx, cy)
        
        except Exception as e:
            print(f"Post-processing error: {e}")
        
        return processed
    
    def calculate_clinical_measurements(self, predictions):
        """Calculate clinical measurements from predictions"""
        measurements = {}
        
        try:
            # Check if we have the required features
            if 'CEJ' not in self.point_coords_cache or 'GM' not in self.point_coords_cache:
                return measurements
            
            if 'ABC' not in self.point_coords_cache:
                return measurements
            
            cej_x, cej_y = self.point_coords_cache['CEJ']
            gm_x, gm_y = self.point_coords_cache['GM']
            abc_x, abc_y = self.point_coords_cache['ABC']
            
            # Calculate distances in pixels
            igr_pixels = abs(cej_y - gm_y)
            igh_pixels = abs(cej_y - abc_y)
            iabl_pixels = abs(gm_y - abc_y)
            
            # Convert to mm
            measurements['iGR'] = igr_pixels / self.pixels_per_mm
            measurements['iGH'] = igh_pixels / self.pixels_per_mm
            measurements['iABL'] = iabl_pixels / self.pixels_per_mm
            
        except Exception as e:
            print(f"Measurement calculation error: {e}")
        
        return measurements
    
    def get_measurements_throttled(self, predictions):
        """Get measurements with throttling to reduce computation"""
        self.measurements_frame_counter += 1
        
        if self.measurements_frame_counter >= self.measurements_update_interval or self.measurements_cache is None:
            self.measurements_cache = self.calculate_clinical_measurements(predictions)
            self.measurements_frame_counter = 0
        
        return self.measurements_cache if self.measurements_cache else {}
    
    def draw_clinical_measurements(self, frame, measurements):
        """Draw measurement lines on frame"""
        if not measurements:
            return frame
        
        annotated = frame.copy()
        
        try:
            if 'CEJ' in self.point_coords_cache and 'GM' in self.point_coords_cache:
                cej_x, cej_y = self.point_coords_cache['CEJ']
                gm_x, gm_y = self.point_coords_cache['GM']
                
                if 'iGR' in measurements:
                    cv2.line(annotated, (cej_x, cej_y), (gm_x, gm_y), 
                           CLINICAL_COLORS['iGR'], 2)
                    mid_x = (cej_x + gm_x) // 2
                    mid_y = (cej_y + gm_y) // 2
                    cv2.putText(annotated, f"iGR: {measurements['iGR']:.2f}mm",
                              (mid_x + 10, mid_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                              CLINICAL_COLORS['iGR'], 2)
            
            if 'CEJ' in self.point_coords_cache and 'ABC' in self.point_coords_cache:
                cej_x, cej_y = self.point_coords_cache['CEJ']
                abc_x, abc_y = self.point_coords_cache['ABC']
                
                if 'iGH' in measurements:
                    cv2.line(annotated, (cej_x, cej_y), (abc_x, abc_y),
                           CLINICAL_COLORS['iGH'], 2)
            
            if 'GM' in self.point_coords_cache and 'ABC' in self.point_coords_cache:
                gm_x, gm_y = self.point_coords_cache['GM']
                abc_x, abc_y = self.point_coords_cache['ABC']
                
                if 'iABL' in measurements:
                    cv2.line(annotated, (gm_x, gm_y), (abc_x, abc_y),
                           CLINICAL_COLORS['iABL'], 2)
        
        except Exception as e:
            print(f"Drawing error: {e}")
        
        return annotated
    
    def add_clinical_measurements_overlay(self, frame, measurements):
        """Add measurement overlay panel"""
        if not measurements:
            return frame
        
        overlay_height = 120
        overlay_width = 300
        overlay_y = frame.shape[0] - overlay_height - 10
        overlay_x = frame.shape[1] - overlay_width - 10
        
        overlay_region = frame[overlay_y:overlay_y+overlay_height, 
                              overlay_x:overlay_x+overlay_width].copy()
        overlay_bg = np.zeros_like(overlay_region)
        overlay_region = cv2.addWeighted(overlay_region, 0.7, overlay_bg, 0.3, 0)
        
        y_offset = 25
        cv2.putText(overlay_region, "Clinical Measurements", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        y_offset += 25
        if 'iGR' in measurements:
            cv2.putText(overlay_region, f"iGR:  {measurements['iGR']:.2f} mm",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, CLINICAL_COLORS['iGR'], 2)
            y_offset += 25
        
        if 'iGH' in measurements:
            cv2.putText(overlay_region, f"iGH:  {measurements['iGH']:.2f} mm",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, CLINICAL_COLORS['iGH'], 2)
            y_offset += 25
        
        if 'iABL' in measurements:
            cv2.putText(overlay_region, f"iABL: {measurements['iABL']:.2f} mm",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, CLINICAL_COLORS['iABL'], 2)
        
        frame[overlay_y:overlay_y+overlay_height, 
              overlay_x:overlay_x+overlay_width] = overlay_region
        
        return frame
    
    def _create_overlay(self, processed_predictions, target_shape):
        """Create colored overlay from predictions"""
        overlay = np.zeros((*target_shape, 3), dtype=np.uint8)
        
        for feature in REGION_FEATURES:
            if feature in processed_predictions:
                mask = processed_predictions[feature]
                
                # Ensure mask is 2D
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                elif mask.ndim > 3:
                    mask = mask.squeeze()
                    if mask.ndim == 3:
                        mask = mask[:, :, 0]
                    
                color = FEATURE_COLORS[feature]
                if np.any(mask > 0):
                    indices = np.where(mask > 0)
                    if len(indices) == 2:
                        rows, cols = indices
                        overlay[rows, cols] = color
        
        return overlay
    
    def create_annotated_frame(self, frame, processed_predictions):
        """Create annotated frame with overlays"""
        annotated = frame.copy()
        
        # Update overlay cache periodically
        self.overlay_frame_counter += 1
        if self.overlay_frame_counter >= self.overlay_update_interval or self.overlay_cache is None:
            self.overlay_cache = self._create_overlay(processed_predictions, frame.shape[:2])
            self.overlay_frame_counter = 0
        
        # Blend overlay
        if self.overlay_cache is not None:
            annotated = cv2.addWeighted(annotated, 0.75, self.overlay_cache, 0.25, 0)
        
        # Draw point features
        for feature in POINT_FEATURES:
            if feature in self.point_coords_cache:
                cx, cy = self.point_coords_cache[feature]
                color = FEATURE_COLORS[feature]
                
                cv2.circle(annotated, (cx, cy), 15, color, -1)
                cv2.circle(annotated, (cx, cy), 17, (255, 255, 255), 3)
                cv2.putText(annotated, feature, (cx-20, cy-25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add clinical measurements
        if self.show_clinical_measurements:
            measurements = self.get_measurements_throttled(processed_predictions)
            annotated = self.draw_clinical_measurements(annotated, measurements)
            annotated = self.add_clinical_measurements_overlay(annotated, measurements)
        
        return annotated
    
    def add_performance_overlay(self, frame):
        """Add performance statistics overlay"""
        overlay_region = frame[10:110, 10:310].copy()
        overlay_bg = np.zeros_like(overlay_region)
        overlay_region = cv2.addWeighted(overlay_region, 0.8, overlay_bg, 0.2, 0)
        
        fps_color = (0, 255, 0) if self.performance_stats['fps'] >= 50 else (255, 255, 0)
        
        cv2.putText(overlay_region, f"FPS: {self.performance_stats['fps']:.1f}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        cv2.putText(overlay_region, f"Inference: {self.performance_stats['inference_time']:.0f}ms", 
                   (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(overlay_region, f"Latency: {self.performance_stats['avg_latency']:.0f}ms", 
                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(overlay_region, f"Frames: {self.performance_stats['total_frames']}", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        frame[10:110, 10:310] = overlay_region
        return frame
    
    def process_real_time_video(self, video_capture):
        """Main processing loop for real-time video"""
        print("\n" + "="*70)
        print("Starting Hybrid Fusion real-time processing")
        print("Target: 60+ FPS")
        print("="*70)
        
        fps_counter = 0
        fps_start_time = time.time()
        latency_samples = deque(maxlen=30)
        
        cv2.namedWindow("Hybrid Fusion - Real-time", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hybrid Fusion - Real-time", 1200, 800)
        
        frame_skip_counter = 0
        
        try:
            frame_count = 0
            while True:
                frame_count += 1
                    
                frame_start_time = time.time()
                
                frame = video_capture.get_frame()
                if frame is None:
                    frame_skip_counter += 1
                    if frame_skip_counter > 100:
                        print("Connection lost")
                        break
                    time.sleep(0.01)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    continue
                
                frame_skip_counter = 0
                
                try:
                    # Inference
                    inference_start = time.time()
                    
                    frame_batch = self.preprocess_frame_fast(frame)
                    if frame_batch is None:
                        continue
                    
                    predictions = self.predict_single_frame(frame_batch)
                    if predictions is None:
                        continue
                        
                    inference_time = (time.time() - inference_start) * 1000
                    
                    # Post-process
                    processed_predictions = self.post_process_predictions(predictions, frame.shape[:2])
                    annotated_frame = self.create_annotated_frame(frame, processed_predictions)
                    
                    # Update performance stats
                    total_latency = (time.time() - frame_start_time) * 1000
                    latency_samples.append(total_latency)
                    
                    self.performance_stats['inference_time'] = inference_time
                    self.performance_stats['avg_latency'] = sum(latency_samples) / len(latency_samples)
                    self.performance_stats['total_frames'] += 1
                    
                    # Update FPS
                    fps_counter += 1
                    current_time = time.time()
                    if current_time - fps_start_time >= 1.0:
                        self.performance_stats['fps'] = fps_counter / (current_time - fps_start_time)
                        fps_counter = 0
                        fps_start_time = current_time
                        print(f"FPS: {self.performance_stats['fps']:.1f}, Latency: {self.performance_stats['avg_latency']:.0f}ms")
                    
                    # Add overlays and display
                    annotated_frame = self.add_performance_overlay(annotated_frame)
                    cv2.imshow("Hybrid Fusion - Real-time", annotated_frame)
                    
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    import traceback
                    traceback.print_exc()
                    # Show original frame on error
                    cv2.imshow("Hybrid Fusion - Real-time", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"hybrid_fusion_{timestamp}.png"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"Saved: {filename}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cv2.destroyAllWindows()
            print(f"\nFinal FPS: {self.performance_stats['fps']:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time Video Annotation with Hybrid Fusion Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Basic usage
  python live_hybrid_fusion_realtime.py \\
      --model_path "./hybrid_fusion_best.h5" \\
      --device_id 1

  # Custom crop settings
  python live_hybrid_fusion_realtime.py \\
      --model_path "./model.h5" \\
      --device_id 0 \\
      --crop_x 800 --crop_y 220 \\
      --crop_width 610 --crop_height 640

  # Disable clinical measurements
  python live_hybrid_fusion_realtime.py \\
      --model_path "./model.h5" \\
      --disable_clinical

Controls:
  - Press 'q' to quit
  - Press 's' to save screenshot
        """
    )
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained Hybrid Fusion model (.h5 file)")
    parser.add_argument("--device_id", type=int, default=1,
                       help="Video capture device ID (default: 1)")
    parser.add_argument("--features", type=str, nargs='+', default=None,
                       help="Features to predict (default: all)")
    parser.add_argument("--crop_x", type=int, default=800,
                       help="Crop X coordinate (default: 800)")
    parser.add_argument("--crop_y", type=int, default=220,
                       help="Crop Y coordinate (default: 220)")
    parser.add_argument("--crop_width", type=int, default=610,
                       help="Crop width (default: 610)")
    parser.add_argument("--crop_height", type=int, default=640,
                       help="Crop height (default: 640)")
    parser.add_argument("--disable_clinical", action="store_true",
                       help="Disable clinical measurements")
    parser.add_argument("--pixels_per_mm", type=float, default=152.25,
                       help="Pixels per millimeter (default: 152.25)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("HYBRID FUSION - REAL-TIME VIDEO ANNOTATION")
    print("="*70)
    print("Optimized for Elgato HD60 X Capture Card")
    print("Architecture: Hybrid Cross Fusion (Spatial + Channel)")
    print("="*70)
    
    video_capture = None
    
    try:
        # Check GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if not gpus:
            print("\nWARNING: No GPU detected. Performance may be limited.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
        else:
            print(f"\n✓ GPU detected: {len(gpus)} device(s)")
        
        # Initialize capture
        print("\n[1/3] Initializing HD60 X capture...")
        video_capture = HD60XOptimizedCapture(
            device_id=args.device_id,
            crop_x=args.crop_x,
            crop_y=args.crop_y,
            crop_width=args.crop_width,
            crop_height=args.crop_height
        )
        
        video_capture.start_capture()
        time.sleep(2)  # Allow capture to stabilize
        
        # Test capture
        test_frame = video_capture.get_frame()
        if test_frame is None or test_frame.size == 0:
            print("ERROR: Cannot capture frames from HD60 X")
            return
        print(f"✓ Capture OK: {test_frame.shape}")
        
        # Load model
        print("\n[2/3] Loading Hybrid Fusion model...")
        annotator = RealTimeHybridFusionAnnotator(
            model_path=args.model_path,
            features=args.features,
            show_clinical_measurements=not args.disable_clinical,
            pixels_per_mm=args.pixels_per_mm
        )
        
        print("\n[3/3] System ready")
        print("\n" + "="*70)
        print("READY - Press 'q' to quit, 's' to save screenshot")
        print("="*70 + "\n")
        
        # Start processing
        annotator.process_real_time_video(video_capture)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if video_capture:
            video_capture.stop_capture()
        
        cv2.destroyAllWindows()
        gc.collect()
        print("\nSession complete")


if __name__ == "__main__":
    main()