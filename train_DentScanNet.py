import os
import argparse
import json
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
import numpy as np
from datetime import datetime
import time

# Import base components
from standalone_production_training import (
    stable_reparameterizable_block,
    ssm_v1_original,
    create_optimized_feature_heads,
    OptimizedDataLoader,
    ALL_FEATURES,
    POINT_FEATURES,
    REGION_FEATURES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CLASSES,
    dice_coefficient_metric,
)

# Import improved components
from improved_architectural_components import (
    SelectivePyramidPooling,
)

mixed_precision.set_global_policy('mixed_float16')




# ============================================================================
# HYBRID FUSION LAYER
# ============================================================================

class HybridCrossFusion(layers.Layer):
    """
    Hybrid Cross Fusion: Spatial Attention + Channel Gating
    
    Combines both WHERE (spatial) and WHAT (channels) to fuse.
    Best performing fusion strategy based on comprehensive evaluation.
    
    Args:
        spatial_reduction: Reduction ratio for spatial attention (default: 8)
        channel_reduction: Reduction ratio for channel gating (default: 4)
        use_residual: Add residual connection (default: True)
    """
    
    def __init__(self, spatial_reduction=8, channel_reduction=4, 
                 use_residual=True, **kwargs):
        super().__init__(**kwargs)
        self.spatial_reduction = spatial_reduction
        self.channel_reduction = channel_reduction
        self.use_residual = use_residual
        
    def build(self, input_shape):
        detail_shape = input_shape[0]
        semantic_shape = input_shape[1]
        semantic_up_shape = input_shape[2]
        detail_down_shape = input_shape[3]
        
        detail_filters = detail_shape[-1]
        semantic_filters = semantic_shape[-1]
        semantic_up_filters = semantic_up_shape[-1]
        detail_down_filters = detail_down_shape[-1]
        
        # ===== DETAIL BRANCH =====
        # Spatial attention
        self.detail_spatial_conv1 = layers.Conv2D(
            filters=max(8, detail_filters // self.spatial_reduction),
            kernel_size=3, padding='same', activation='relu',
            kernel_initializer='he_normal', name='detail_spatial_conv1'
        )
        self.detail_spatial_conv2 = layers.Conv2D(
            filters=1, kernel_size=1, padding='same', activation='sigmoid',
            kernel_initializer='glorot_uniform', name='detail_spatial_conv2'
        )
        
        # Channel gating
        self.detail_gap = layers.GlobalAveragePooling2D(name='detail_gap')
        self.detail_channel_fc1 = layers.Dense(
            units=max(8, (detail_filters + semantic_up_filters) // self.channel_reduction),
            activation='relu', kernel_initializer='he_normal',
            name='detail_channel_fc1'
        )
        self.detail_channel_fc2 = layers.Dense(
            units=semantic_up_filters, activation='sigmoid',
            kernel_initializer='glorot_uniform', name='detail_channel_fc2'
        )
        
        # ===== SEMANTIC BRANCH =====
        # Spatial attention
        self.semantic_spatial_conv1 = layers.Conv2D(
            filters=max(8, semantic_filters // self.spatial_reduction),
            kernel_size=3, padding='same', activation='relu',
            kernel_initializer='he_normal', name='semantic_spatial_conv1'
        )
        self.semantic_spatial_conv2 = layers.Conv2D(
            filters=1, kernel_size=1, padding='same', activation='sigmoid',
            kernel_initializer='glorot_uniform', name='semantic_spatial_conv2'
        )
        
        # Channel gating
        self.semantic_gap = layers.GlobalAveragePooling2D(name='semantic_gap')
        self.semantic_channel_fc1 = layers.Dense(
            units=max(8, (semantic_filters + detail_down_filters) // self.channel_reduction),
            activation='relu', kernel_initializer='he_normal',
            name='semantic_channel_fc1'
        )
        self.semantic_channel_fc2 = layers.Dense(
            units=detail_down_filters, activation='sigmoid',
            kernel_initializer='glorot_uniform', name='semantic_channel_fc2'
        )
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        detail, semantic, semantic_up, detail_down = inputs
        
        # ===== DETAIL BRANCH =====
        detail_context = layers.Concatenate()([detail, semantic_up])
        
        # Spatial attention
        spatial_attn = self.detail_spatial_conv1(detail_context)
        spatial_attn = self.detail_spatial_conv2(spatial_attn)
        
        # Channel gating
        channel_context = self.detail_gap(detail_context)
        channel_gate = self.detail_channel_fc1(channel_context)
        channel_gate = self.detail_channel_fc2(channel_gate)
        channel_gate = layers.Reshape((1, 1, -1))(channel_gate)
        
        # Combined attention: spatial * channel
        combined_attn = spatial_attn * channel_gate
        
        if self.use_residual:
            detail_fused = detail + combined_attn * semantic_up
        else:
            detail_fused = detail * (1 - combined_attn) + combined_attn * semantic_up
        
        # ===== SEMANTIC BRANCH =====
        semantic_context = layers.Concatenate()([semantic, detail_down])
        
        spatial_attn = self.semantic_spatial_conv1(semantic_context)
        spatial_attn = self.semantic_spatial_conv2(spatial_attn)
        
        channel_context = self.semantic_gap(semantic_context)
        channel_gate = self.semantic_channel_fc1(channel_context)
        channel_gate = self.semantic_channel_fc2(channel_gate)
        channel_gate = layers.Reshape((1, 1, -1))(channel_gate)
        
        combined_attn = spatial_attn * channel_gate
        
        if self.use_residual:
            semantic_fused = semantic + combined_attn * detail_down
        else:
            semantic_fused = semantic * (1 - combined_attn) + combined_attn * detail_down
        
        return detail_fused, semantic_fused
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'spatial_reduction': self.spatial_reduction,
            'channel_reduction': self.channel_reduction,
            'use_residual': self.use_residual
        })
        return config


# ============================================================================
# MODEL ARCHITECTURE - HYBRID FUSION PRODUCTION
# ============================================================================

def build_hybrid_fusion_model(input_shape=(256, 256, 3), features=ALL_FEATURES,
                               spatial_reduction=8, channel_reduction=4):
    """
    Build the production Hybrid Fusion model
    
    Architecture:
    - Stem: 32 filters, stride 2
    - Detail Branch: 32 filters, 2 reparam blocks
    - Semantic Branch: 64→96 filters, SSM, Selective Pyramid
    - Hybrid Cross Fusion: Spatial Attention (WHERE) + Channel Gating (WHAT)
    - Final: Second SSM, upsampling, multi-head outputs
    
    Args:
        input_shape: Input image shape (H, W, C)
        features: List of features to predict
        spatial_reduction: Spatial attention reduction ratio (default: 8)
        channel_reduction: Channel gating reduction ratio (default: 4)
    
    Returns:
        model: Keras Model
        feature_outputs: Dictionary of output layers
    """
    
    print("Building Hybrid Fusion Production Model...")
    print(f"  Architecture:")
    print(f"    - Reparam blocks (stable gradient flow)")
    print(f"    - SSM blocks (global reasoning)")
    print(f"    - Selective Pyramid (pool_sizes=[1,2,3])")
    print(f"    - Hybrid Fusion (spatial={spatial_reduction}, channel={channel_reduction})")
    
    inputs = layers.Input(input_shape, name='input')
    
    # ========== STEM ==========
    x = layers.Conv2D(32, 3, strides=2, padding='same',
                      kernel_initializer='he_normal', name='stem_conv')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.ReLU(name='stem_relu')(x)
    
    # ========== DETAIL BRANCH ==========
    # Preserves fine-grained spatial information for point detection
    detail = layers.Conv2D(32, 3, padding='same',
                          kernel_initializer='he_normal', name='detail_conv')(x)
    detail = layers.BatchNormalization(name='detail_bn')(detail)
    detail = layers.ReLU(name='detail_relu')(detail)
    
    for i in range(2):
        detail = stable_reparameterizable_block(detail, 32, stride=1,
                                               training=True,
                                               block_id=f'detail_block_{i}')
    
    # ========== SEMANTIC BRANCH ==========
    # Captures global context for region segmentation
    semantic = layers.Conv2D(64, 3, strides=2, padding='same',
                            kernel_initializer='he_normal', name='semantic_conv')(x)
    semantic = layers.BatchNormalization(name='semantic_bn')(semantic)
    semantic = layers.ReLU(name='semantic_relu')(semantic)
    
    semantic = stable_reparameterizable_block(semantic, 64, stride=1,
                                             training=True,
                                             block_id='semantic_1')
    
    # ========== SSM BLOCK ==========
    semantic = ssm_v1_original(semantic, 96, 32, 'ssm_1')
    
    # ========== SELECTIVE PYRAMID POOLING ==========
    semantic = SelectivePyramidPooling(
        filters=96, 
        pool_sizes=[1, 2, 3],
        preserve_detail=True,
        name='selective_pyramid'
    )(semantic)
    
    print("  ✓ Selective Pyramid: pool_sizes=[1,2,3]")
    
    # ========== CROSS-PATH CONNECTIONS ==========
    semantic_up = layers.Conv2DTranspose(32, 3, strides=2, padding='same',
                                        kernel_initializer='he_normal',
                                        name='cross_up')(semantic)
    semantic_up = layers.Lambda(lambda x: tf.image.resize(
        x[0], tf.shape(x[1])[1:3], method='bilinear'
    ), name='match_detail_shape')([semantic_up, detail])
    
    detail_down = layers.Conv2D(96, 3, strides=2, padding='same',
                               kernel_initializer='he_normal',
                               name='cross_down')(detail)
    detail_down = layers.Lambda(lambda x: tf.image.resize(
        x[0], tf.shape(x[1])[1:3], method='bilinear'
    ), name='match_semantic_shape')([detail_down, semantic])
    
    # ========== HYBRID CROSS FUSION ==========
    # Combines spatial attention (WHERE) + channel gating (WHAT)
    detail_fused, semantic_fused = HybridCrossFusion(
        spatial_reduction=spatial_reduction,
        channel_reduction=channel_reduction,
        use_residual=True,
        name='hybrid_fusion'
    )([detail, semantic, semantic_up, detail_down])
    
    print(f"  ✓ Hybrid Fusion: spatial_reduction={spatial_reduction}, channel_reduction={channel_reduction}")
    
    # ========== FINAL PROCESSING ==========
    detail_final = stable_reparameterizable_block(detail_fused, 48, stride=1,
                                                 training=True,
                                                 block_id='detail_final')
    semantic_final = stable_reparameterizable_block(semantic_fused, 96, stride=2,
                                                   training=True,
                                                   block_id='semantic_final')
    
    # Second SSM block for additional semantic reasoning
    semantic_final = ssm_v1_original(semantic_final, 128, 48, 'ssm_2')
    
    # ========== UPSAMPLING ==========
    semantic_up_final = layers.Conv2DTranspose(48, 3, strides=4, padding='same',
                                              kernel_initializer='he_normal',
                                              name='final_up')(semantic_final)
    semantic_up_final = layers.Lambda(lambda x: tf.image.resize(
        x[0], tf.shape(x[1])[1:3], method='bilinear'
    ), name='final_match')([semantic_up_final, detail_final])
    
    # ========== FINAL FUSION ==========
    shared_features = layers.Concatenate(name='shared_concat')([detail_final, semantic_up_final])
    shared_features = layers.Conv2D(64, 3, padding='same',
                                   kernel_initializer='he_normal',
                                   name='shared_conv')(shared_features)
    shared_features = layers.BatchNormalization(name='shared_bn')(shared_features)
    shared_features = layers.ReLU(name='shared_relu')(shared_features)
    
    # ========== FINAL UPSAMPLING TO INPUT SIZE ==========
    shared_features = layers.Conv2DTranspose(32, 3, strides=2, padding='same',
                                            kernel_initializer='he_normal',
                                            name='final_upsample')(shared_features)
    shared_features = layers.Lambda(lambda x: tf.image.resize(
        x, [input_shape[0], input_shape[1]], method='bilinear'
    ), name='resize_final')(shared_features)
    
    # ========== OUTPUT HEADS (one per feature) ==========
    feature_outputs = create_optimized_feature_heads(shared_features, features)
    
    model = models.Model(inputs, list(feature_outputs.values()),
                        name='HybridFusion_Production')
    
    # Set output names
    for i, feature in enumerate(features):
        model.output_names[i] = f'{feature}_output'
    
    print(f"  ✓ Model built: {model.count_params():,} parameters\n")
    
    return model, feature_outputs


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def weighted_dice_loss(y_true, y_pred, class_weights):
    """Weighted dice loss for imbalanced classes"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Handle NaN values
    y_pred = tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred), y_pred)
    y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    
    # Clip predictions for numerical stability
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    total_loss = 0.0
    total_weight = tf.reduce_sum(class_weights)
    
    # Calculate dice for each class
    for class_idx, weight in enumerate(class_weights):
        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]
        
        y_true_flat = tf.reshape(y_true_class, [-1])
        y_pred_flat = tf.reshape(y_pred_class, [-1])
        
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
        
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice = tf.where(tf.math.is_nan(dice), 0.0, dice)
        dice = tf.clip_by_value(dice, 0.0, 1.0)
        
        total_loss += weight * (1.0 - dice)
    
    normalized_loss = total_loss / total_weight
    normalized_loss = tf.where(tf.math.is_nan(normalized_loss), 1.0, normalized_loss)
    
    return tf.cast(normalized_loss, tf.float32)


def create_loss_functions(features=ALL_FEATURES):
    """Create loss functions for all features"""
    losses = {}
    loss_weights = {}
    metrics = {}
    
    for feature in features:
        output_name = f'{feature}_output'
        
        # Create dice loss with 1:20 class weights (background:foreground)
        def create_dice_loss():
            def loss_fn(y_true, y_pred):
                return weighted_dice_loss(y_true, y_pred, [1.0, 20.0])
            return loss_fn
        
        losses[output_name] = create_dice_loss()
        loss_weights[output_name] = 1.0
        metrics[output_name] = [dice_coefficient_metric, 'accuracy']
    
    return losses, loss_weights, metrics


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_hybrid_fusion(data_dir, output_dir, epochs=30, batch_size=4, 
                       learning_rate=1e-4, seed=42,
                       spatial_reduction=8, channel_reduction=4):
    """
    Train the Hybrid Fusion production model
    
    Args:
        data_dir: Path to data directory (must contain train/ and test/)
        output_dir: Output directory for results
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        seed: Random seed for reproducibility
        spatial_reduction: Spatial attention reduction ratio
        channel_reduction: Channel gating reduction ratio
    
    Returns:
        Dictionary with training results
    """
    
    print("="*80)
    print(f"TRAINING HYBRID FUSION MODEL (Seed: {seed})")
    print("="*80)
    
    # Set seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Build model
    model, _ = build_hybrid_fusion_model(
        (IMAGE_HEIGHT, IMAGE_WIDTH, 3), 
        ALL_FEATURES,
        spatial_reduction=spatial_reduction,
        channel_reduction=channel_reduction
    )
    
    # Compile
    losses, loss_weights, metrics = create_loss_functions(ALL_FEATURES)
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0
    )
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    print("\nModel compiled with:")
    print(f"  Optimizer: Adam (lr={learning_rate}, clipnorm=1.0)")
    print(f"  Loss: Weighted Dice (1:20 class weights)")
    print(f"  Metrics: Dice coefficient, Accuracy\n")
    
    # Data loaders
    train_path = os.path.join(data_dir, 'train')
    test_path = os.path.join(data_dir, 'test')
    
    print("Loading data...")
    train_loader = OptimizedDataLoader(
        train_path, ALL_FEATURES, batch_size, (IMAGE_HEIGHT, IMAGE_WIDTH),
        enable_point_dilation=True, dilation_radius=4, is_training=True
    )
    
    test_loader = OptimizedDataLoader(
        test_path, ALL_FEATURES, batch_size, (IMAGE_HEIGHT, IMAGE_WIDTH),
        enable_point_dilation=True, dilation_radius=4, is_training=False
    )
    
    train_steps = max(1, len(train_loader.image_files) // batch_size)
    test_steps = max(1, len(test_loader.image_files) // batch_size)
    
    print(f"  Train: {len(train_loader.image_files)} images ({train_steps} steps/epoch)")
    print(f"  Test:  {len(test_loader.image_files)} images ({test_steps} steps/epoch)\n")
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, f'hybrid_fusion_best_seed{seed}.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            verbose=1,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=6,
            verbose=1,
            mode='min',
            min_lr=1e-7
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, f'training_log_seed{seed}.csv')
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, f'logs_seed{seed}'),
            histogram_freq=0,
            write_graph=False
        )
    ]
    
    # Train
    print("="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    start_time = time.time()
    
    history = model.fit(
        train_loader(),
        validation_data=test_loader(),
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=test_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total time: {training_time/3600:.2f} hours")
    print(f"Epochs trained: {len(history.history['loss'])}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = model.evaluate(test_loader(), steps=test_steps, return_dict=True)
    
    # Extract metrics
    feature_performance = {}
    for feature in ALL_FEATURES:
        dice_key = f'{feature}_output_dice_coefficient_metric'
        if dice_key in results:
            feature_performance[feature] = results[dice_key]
    
    avg_dice = np.mean(list(feature_performance.values()))
    point_dice = np.mean([feature_performance[f] for f in POINT_FEATURES
                         if f in feature_performance])
    region_dice = np.mean([feature_performance[f] for f in REGION_FEATURES
                          if f in feature_performance])
    
    # Save results
    final_results = {
        'model_name': 'hybrid_fusion_production',
        'seed': seed,
        'architecture': {
            'fusion': 'HybridCrossFusion (Spatial Attention + Channel Gating)',
            'spatial_reduction': spatial_reduction,
            'channel_reduction': channel_reduction,
            'pyramid': 'SelectivePyramidPooling (pool_sizes=[1,2,3])',
            'reparam': 'Enabled',
            'ssm': 'Enabled'
        },
        'parameters': model.count_params(),
        'training_time_seconds': training_time,
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        
        # Performance
        'test_average_dice': float(avg_dice),
        'test_point_dice': float(point_dice),
        'test_region_dice': float(region_dice),
        'test_feature_performance': {k: float(v) for k, v in feature_performance.items()},
        
        # Expected vs actual
        'expected_overall': 76.67,
        'expected_points': 77.18,
        'expected_regions': 76.15,
        'vs_baseline': {
            'baseline_overall': 62.28,
            'improvement_overall': float((avg_dice * 100) - 62.28),
            'improvement_points': float((point_dice * 100) - 54.13),
            'improvement_regions': float((region_dice * 100) - 70.43),
        }
    }
    
    results_path = os.path.join(output_dir, f'results_seed{seed}.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Overall Dice: {avg_dice:.4f} ({avg_dice*100:.2f}%)")
    print(f"Point Dice:   {point_dice:.4f} ({point_dice*100:.2f}%)")
    print(f"Region Dice:  {region_dice:.4f} ({region_dice*100:.2f}%)")
    print(f"\nImprovement vs Baseline:")
    print(f"  Overall: {final_results['vs_baseline']['improvement_overall']:+.2f}%")
    print(f"  Points:  {final_results['vs_baseline']['improvement_points']:+.2f}%")
    print(f"  Regions: {final_results['vs_baseline']['improvement_regions']:+.2f}%")
    print(f"\nResults saved to: {results_path}")
    print(f"Model saved to: {os.path.join(output_dir, f'hybrid_fusion_best_seed{seed}.h5')}")
    print(f"{'='*80}\n")
    
    # Cleanup
    tf.keras.backend.clear_session()
    del model
    
    return final_results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train Hybrid Fusion Production Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
HYBRID FUSION - BEST PERFORMING ARCHITECTURE

Combines:
  • Spatial Attention (WHERE to fuse) - Learns pixel-level fusion decisions
  • Channel Gating (WHAT to fuse) - Learns feature-level fusion decisions

Expected Performance (based on test evaluation):
  - Overall: ~76.67% (+14.39% vs baseline)
  - Points:  ~77.18% (+23.05% vs baseline)
  - Regions: ~76.15% (+5.72% vs baseline)

Examples:

  # Train with default settings
  python train_hybrid_production.py \\
      --data_dir "D:/Jokerst group/package2/combined_data2/processed" \\
      --output_dir "./hybrid_fusion_production"

  # Train multiple seeds for robust evaluation
  python train_hybrid_production.py \\
      --data_dir "./data" \\
      --output_dir "./hybrid_fusion_multi_seed" \\
      --seeds 42 123 456 \\
      --epochs 30

  # Custom fusion parameters
  python train_hybrid_production.py \\
      --data_dir "./data" \\
      --output_dir "./hybrid_custom" \\
      --spatial_reduction 16 \\
      --channel_reduction 8 \\
      --epochs 40
        """
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory (must contain train/ and test/)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                       help='Random seeds (default: 42)')
    parser.add_argument('--spatial_reduction', type=int, default=8,
                       help='Spatial attention reduction ratio (default: 8)')
    parser.add_argument('--channel_reduction', type=int, default=4,
                       help='Channel gating reduction ratio (default: 4)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        exit(1)
    
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')
    
    if not os.path.exists(train_dir):
        print(f"ERROR: Train directory not found: {train_dir}")
        exit(1)
    if not os.path.exists(test_dir):
        print(f"ERROR: Test directory not found: {test_dir}")
        exit(1)
    
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Seeds: {args.seeds}")
    print(f"Spatial reduction: {args.spatial_reduction}")
    print(f"Channel reduction: {args.channel_reduction}")
    print("="*80)
    
    input("\nPress Enter to start training...")
    
    # Train for each seed
    all_results = []
    
    for seed in args.seeds:
        try:
            results = train_hybrid_fusion(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=seed,
                spatial_reduction=args.spatial_reduction,
                channel_reduction=args.channel_reduction
            )
            all_results.append(results)
            
        except Exception as e:
            print(f"\nERROR training seed {seed}: {e}")
            import traceback
            traceback.print_exc()
    
    # Aggregate results if multiple seeds
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("AGGREGATE RESULTS (MULTIPLE SEEDS)")
        print("="*80)
        
        overall_scores = [r['test_average_dice'] * 100 for r in all_results]
        point_scores = [r['test_point_dice'] * 100 for r in all_results]
        region_scores = [r['test_region_dice'] * 100 for r in all_results]
        
        print(f"Overall: {np.mean(overall_scores):.2f}% ± {np.std(overall_scores):.2f}%")
        print(f"Points:  {np.mean(point_scores):.2f}% ± {np.std(point_scores):.2f}%")
        print(f"Regions: {np.mean(region_scores):.2f}% ± {np.std(region_scores):.2f}%")
        
        # Save aggregate summary
        aggregate_summary = {
            'n_seeds': len(all_results),
            'seeds': args.seeds,
            'overall': {
                'mean': float(np.mean(overall_scores)),
                'std': float(np.std(overall_scores)),
                'min': float(np.min(overall_scores)),
                'max': float(np.max(overall_scores))
            },
            'points': {
                'mean': float(np.mean(point_scores)),
                'std': float(np.std(point_scores)),
                'min': float(np.min(point_scores)),
                'max': float(np.max(point_scores))
            },
            'regions': {
                'mean': float(np.mean(region_scores)),
                'std': float(np.std(region_scores)),
                'min': float(np.min(region_scores)),
                'max': float(np.max(region_scores))
            },
            'all_results': all_results
        }
        
        summary_path = os.path.join(args.output_dir, 'aggregate_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(aggregate_summary, f, indent=2)
        
        print(f"\nAggregate summary saved to: {summary_path}")
        print("="*80)
    
    print("\n✅ Training complete!")
    print(f"Results saved in: {args.output_dir}\n")


if __name__ == '__main__':
    main()