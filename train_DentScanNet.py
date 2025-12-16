
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
    OptimizedDataLoader,
    ALL_FEATURES,
    POINT_FEATURES,
    REGION_FEATURES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CLASSES,
)

from improved_architectural_components import (
    SelectivePyramidPooling,
)

mixed_precision.set_global_policy('mixed_float16')


# ============================================================================
# LOSS FUNCTIONS (same as before)
# ============================================================================

def dice_loss(y_true, y_pred, smooth=1e-6):
    """Standard Dice loss"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    if len(y_true.shape) > 3 and y_true.shape[-1] == 2:
        y_true = y_true[..., 1]
        y_pred = y_pred[..., 1]
    
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-6):
    """Tversky loss"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    if len(y_true.shape) > 3 and y_true.shape[-1] == 2:
        y_true = y_true[..., 1]
        y_pred = y_pred[..., 1]
    
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    true_pos = tf.reduce_sum(y_true_f * y_pred_f)
    false_pos = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    false_neg = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    
    tversky_index = (true_pos + smooth) / (true_pos + alpha * false_pos + beta * false_neg + smooth)
    
    return 1.0 - tversky_index


def focal_tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, gamma=1.5, smooth=1e-6):
    """Focal Tversky loss"""
    tversky = tversky_loss(y_true, y_pred, alpha=alpha, beta=beta, smooth=smooth)
    focal_tversky = tf.pow(tversky, gamma)
    return focal_tversky


def hybrid_dice_tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, gamma=1.5):
    """Hybrid Dice-Tversky loss (Equation 12)"""
    dice = dice_loss(y_true, y_pred)
    focal_tversky = focal_tversky_loss(y_true, y_pred, alpha=alpha, beta=beta, gamma=gamma)
    hybrid = 0.5 * dice + 0.5 * focal_tversky
    return hybrid


def focal_binary_crossentropy(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Focal loss"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    if len(y_true.shape) > 3 and y_true.shape[-1] == 2:
        y_true = y_true[..., 1]
        y_pred = y_pred[..., 1]
    
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    bce = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    focal_weight = tf.pow(1 - p_t, gamma)
    alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    focal_loss = alpha_t * focal_weight * bce
    
    return tf.reduce_mean(focal_loss)


def online_hard_example_mining_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25, 
                                          ohem_ratio=0.25, min_kept=100):
    """Focal loss with OHEM (Equation 13)"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    if len(y_true.shape) > 3 and y_true.shape[-1] == 2:
        y_true = y_true[..., 1]
        y_pred = y_pred[..., 1]
    
    batch_size = tf.shape(y_true)[0]
    
    y_true_flat = tf.reshape(y_true, [batch_size, -1])
    y_pred_flat = tf.reshape(y_pred, [batch_size, -1])
    
    epsilon = tf.keras.backend.epsilon()
    y_pred_flat = tf.clip_by_value(y_pred_flat, epsilon, 1.0 - epsilon)
    
    bce = - (y_true_flat * tf.math.log(y_pred_flat) + (1 - y_true_flat) * tf.math.log(1 - y_pred_flat))
    p_t = tf.where(tf.equal(y_true_flat, 1), y_pred_flat, 1 - y_pred_flat)
    focal_weight = tf.pow(1 - p_t, gamma)
    alpha_t = tf.where(tf.equal(y_true_flat, 1), alpha, 1 - alpha)
    per_pixel_loss = alpha_t * focal_weight * bce
    
    num_pixels = tf.shape(per_pixel_loss)[1]
    num_keep = tf.maximum(
        tf.cast(tf.cast(num_pixels, tf.float32) * ohem_ratio, tf.int32),
        min_kept
    )
    
    top_k_losses, _ = tf.nn.top_k(per_pixel_loss, k=num_keep, sorted=False)
    ohem_loss = tf.reduce_mean(top_k_losses)
    
    return ohem_loss


def soft_argmax_2d(heatmap):
    """
    FIXED: Differentiable soft-argmax with proper dtype handling
    
    Key fix: Cast coordinate grids to match input dtype (FP16 or FP32)
    """
    if len(heatmap.shape) == 4:
        if heatmap.shape[-1] == 2:
            heatmap = heatmap[..., 1]  # Extract foreground
        else:
            heatmap = tf.squeeze(heatmap, axis=-1)
    
    # Get input dtype (will be float16 with mixed precision)
    input_dtype = heatmap.dtype
    
    batch_size = tf.shape(heatmap)[0]
    height = tf.shape(heatmap)[1]
    width = tf.shape(heatmap)[2]
    
    heatmap_flat = tf.reshape(heatmap, [batch_size, height * width])
    
    # Softmax in float32 for numerical stability
    heatmap_flat_f32 = tf.cast(heatmap_flat, tf.float32)
    weights = tf.nn.softmax(heatmap_flat_f32, axis=1)
    weights = tf.reshape(weights, [batch_size, height, width])
    
    # Create coordinate grids in float32
    x_coords = tf.range(width, dtype=tf.float32)
    y_coords = tf.range(height, dtype=tf.float32)
    
    x_grid = tf.tile(tf.reshape(x_coords, [1, 1, width]), [batch_size, height, 1])
    y_grid = tf.tile(tf.reshape(y_coords, [1, height, 1]), [batch_size, 1, width])
    
    # Compute weighted average (all in float32)
    x_pred = tf.reduce_sum(weights * x_grid, axis=[1, 2])
    y_pred = tf.reduce_sum(weights * y_grid, axis=[1, 2])
    
    coords = tf.stack([x_pred, y_pred], axis=1)
    
    # Keep in float32 for coordinate regression (Huber loss expects float32)
    return coords


def huber_coordinate_loss(y_true, y_pred, delta=1.0):
    """Huber loss for coordinates (Equation 14)"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    error = tf.abs(y_true - y_pred)
    
    quadratic = 0.5 * tf.square(error)
    linear = delta * (error - 0.5 * delta)
    
    huber = tf.where(error <= delta, quadratic, linear)
    
    loss = tf.reduce_mean(tf.reduce_sum(huber, axis=1))
    
    return loss


# ============================================================================
# COORDINATE EXTRACTOR LAYER (FIXED)
# ============================================================================

class SoftArgmaxCoordinateExtractor(layers.Layer):
    """
    FIXED: Extracts coordinates with proper dtype handling
    Output is always float32 (for Huber loss)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, heatmap):
        # soft_argmax_2d handles dtype internally and returns float32
        return soft_argmax_2d(heatmap)
    
    def get_config(self):
        return super().get_config()


# ============================================================================
# HYBRID FUSION (unchanged)
# ============================================================================

class HybridCrossFusion(layers.Layer):
    """Hybrid Cross Fusion: Spatial Attention + Channel Gating"""
    
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
        
        # Detail branch
        self.detail_spatial_conv1 = layers.Conv2D(
            filters=max(8, detail_filters // self.spatial_reduction),
            kernel_size=3, padding='same', activation='relu',
            kernel_initializer='he_normal', name='detail_spatial_conv1'
        )
        self.detail_spatial_conv2 = layers.Conv2D(
            filters=1, kernel_size=1, padding='same', activation='sigmoid',
            kernel_initializer='glorot_uniform', name='detail_spatial_conv2'
        )
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
        
        # Semantic branch
        self.semantic_spatial_conv1 = layers.Conv2D(
            filters=max(8, semantic_filters // self.spatial_reduction),
            kernel_size=3, padding='same', activation='relu',
            kernel_initializer='he_normal', name='semantic_spatial_conv1'
        )
        self.semantic_spatial_conv2 = layers.Conv2D(
            filters=1, kernel_size=1, padding='same', activation='sigmoid',
            kernel_initializer='glorot_uniform', name='semantic_spatial_conv2'
        )
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
        
        detail_context = layers.Concatenate()([detail, semantic_up])
        spatial_attn = self.detail_spatial_conv1(detail_context)
        spatial_attn = self.detail_spatial_conv2(spatial_attn)
        channel_context = self.detail_gap(detail_context)
        channel_gate = self.detail_channel_fc1(channel_context)
        channel_gate = self.detail_channel_fc2(channel_gate)
        channel_gate = layers.Reshape((1, 1, -1))(channel_gate)
        combined_attn = spatial_attn * channel_gate
        
        if self.use_residual:
            detail_fused = detail + combined_attn * semantic_up
        else:
            detail_fused = detail * (1 - combined_attn) + combined_attn * semantic_up
        
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
# MODEL ARCHITECTURE (same structure, uses fixed SoftArgmax)
# ============================================================================

def build_complete_manuscript_model(input_shape=(256, 256, 3), features=ALL_FEATURES,
                                   spatial_reduction=8, channel_reduction=4):
    """Build COMPLETE manuscript-consistent model with dtype fix"""
    
        
    inputs = layers.Input(input_shape, name='input')
    
    # ========== STEM ==========
    x = layers.Conv2D(32, 3, strides=2, padding='same',
                      kernel_initializer='he_normal', name='stem_conv')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.ReLU(name='stem_relu')(x)
    
    # ========== DETAIL BRANCH ==========
    detail = layers.Conv2D(32, 3, padding='same',
                          kernel_initializer='he_normal', name='detail_conv')(x)
    detail = layers.BatchNormalization(name='detail_bn')(detail)
    detail = layers.ReLU(name='detail_relu')(detail)
    
    for i in range(2):
        detail = stable_reparameterizable_block(detail, 32, stride=1,
                                               training=True,
                                               block_id=f'detail_block_{i}')
    
    # ========== SEMANTIC BRANCH ==========
    semantic = layers.Conv2D(64, 3, strides=2, padding='same',
                            kernel_initializer='he_normal', name='semantic_conv')(x)
    semantic = layers.BatchNormalization(name='semantic_bn')(semantic)
    semantic = layers.ReLU(name='semantic_relu')(semantic)
    
    semantic = stable_reparameterizable_block(semantic, 64, stride=1,
                                             training=True,
                                             block_id='semantic_1')
    
    semantic = ssm_v1_original(semantic, 96, 32, 'ssm_1')
    
    semantic = SelectivePyramidPooling(
        filters=96, 
        pool_sizes=[1, 2, 3],
        preserve_detail=True,
        name='selective_pyramid'
    )(semantic)
    
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
    detail_fused, semantic_fused = HybridCrossFusion(
        spatial_reduction=spatial_reduction,
        channel_reduction=channel_reduction,
        use_residual=True,
        name='hybrid_fusion'
    )([detail, semantic, semantic_up, detail_down])
    
    # ========== FINAL PROCESSING ==========
    detail_final = stable_reparameterizable_block(detail_fused, 48, stride=1,
                                                 training=True,
                                                 block_id='detail_final')
    semantic_final = stable_reparameterizable_block(semantic_fused, 96, stride=2,
                                                   training=True,
                                                   block_id='semantic_final')
    
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
    
    # ========== OUTPUT HEADS ==========
    outputs = []
    output_names = []
    
    for feature in features:
        feature_head = layers.Conv2D(16, 3, padding='same',
                                    kernel_initializer='he_normal',
                                    name=f'{feature}_head_conv')(shared_features)
        feature_head = layers.BatchNormalization(name=f'{feature}_head_bn')(feature_head)
        feature_head = layers.ReLU(name=f'{feature}_head_relu')(feature_head)
        
        if feature in REGION_FEATURES:
            # REGION: Binary mask only
            mask = layers.Conv2D(2, 1, padding='same', activation='sigmoid',
                                kernel_initializer='glorot_uniform',
                                name=f'{feature}_mask')(feature_head)
            outputs.append(mask)
            output_names.append(f'{feature}_mask')
            
        elif feature in POINT_FEATURES:
            # POINT: Heatmap + Coordinate
            
            # Heatmap output (FP16)
            heatmap = layers.Conv2D(2, 1, padding='same', activation='sigmoid',
                                   kernel_initializer='glorot_uniform',
                                   name=f'{feature}_heatmap')(feature_head)
            outputs.append(heatmap)
            output_names.append(f'{feature}_heatmap')
            
            # Coordinate extraction (returns FP32)
            coord_extractor = SoftArgmaxCoordinateExtractor(name=f'{feature}_coord_extractor')
            coords = coord_extractor(heatmap)  # Handles FP16→FP32 internally
            outputs.append(coords)
            output_names.append(f'{feature}_coord')
    
    # Create model
    model = models.Model(inputs, outputs, name='DentScanNet_Complete_FIXED')
    
    for i, name in enumerate(output_names):
        model.output_names[i] = name
    

    
    return model, output_names


# ============================================================================
# LOSS CREATION (unchanged)
# ============================================================================

def create_complete_losses(output_names, loss_weights=None):
    """Create losses matching manuscript"""
    
    if loss_weights is None:
        loss_weights = {
            'region': 1.0,
            'heatmap': 1.0,
            'coord': 0.5,
        }
    
    losses = {}
    weights = {}
    metrics = {}
    
    for output_name in output_names:
        if '_mask' in output_name:
            def create_region_loss():
                def loss_fn(y_true, y_pred):
                    return hybrid_dice_tversky_loss(y_true, y_pred,
                                                   alpha=0.3, beta=0.7, gamma=1.5)
                return loss_fn
            
            losses[output_name] = create_region_loss()
            weights[output_name] = loss_weights['region']
            metrics[output_name] = []
            
        elif '_heatmap' in output_name:
            def create_heatmap_loss():
                def loss_fn(y_true, y_pred):
                    return online_hard_example_mining_focal_loss(y_true, y_pred,
                                                                gamma=2.0, alpha=0.25,
                                                                ohem_ratio=0.25, min_kept=100)
                return loss_fn
            
            losses[output_name] = create_heatmap_loss()
            weights[output_name] = loss_weights['heatmap']
            metrics[output_name] = []
            
        elif '_coord' in output_name:
            def create_coord_loss():
                def loss_fn(y_true, y_pred):
                    return huber_coordinate_loss(y_true, y_pred, delta=1.0)
                return loss_fn
            
            losses[output_name] = create_coord_loss()
            weights[output_name] = loss_weights['coord']
            metrics[output_name] = []
    
    return losses, weights, metrics


# ============================================================================
# DATA LOADER (unchanged)
# ============================================================================

class ManuscriptConsistentDataLoader:
    """Data loader that provides coordinates"""
    
    def __init__(self, base_loader):
        self.base_loader = base_loader
    
    def __call__(self):
        for batch in self.base_loader():
            images, masks_list = batch
            
            targets = {}
            
            for i, feature in enumerate(ALL_FEATURES):
                mask = masks_list[i]
                
                if feature in REGION_FEATURES:
                    targets[f'{feature}_mask'] = mask
                    
                elif feature in POINT_FEATURES:
                    targets[f'{feature}_heatmap'] = mask
                    
                    # Extract coordinates (FP32)
                    coords = self.extract_coordinates_from_mask(mask)
                    targets[f'{feature}_coord'] = coords
            
            yield images, targets
    
    @staticmethod
    def extract_coordinates_from_mask(mask):
        """Extract centroid from mask (returns FP32)"""
        if len(mask.shape) == 4 and mask.shape[-1] == 2:
            mask = mask[..., 1]
        
        batch_size = tf.shape(mask)[0]
        height = tf.shape(mask)[1]
        width = tf.shape(mask)[2]
        
        # All in float32
        x_coords = tf.range(width, dtype=tf.float32)
        y_coords = tf.range(height, dtype=tf.float32)
        
        x_grid = tf.tile(tf.reshape(x_coords, [1, 1, width]), [batch_size, height, 1])
        y_grid = tf.tile(tf.reshape(y_coords, [1, height, 1]), [batch_size, 1, width])
        
        mask = tf.cast(mask, tf.float32)
        total_mass = tf.reduce_sum(mask, axis=[1, 2]) + 1e-6
        
        x_center = tf.reduce_sum(mask * x_grid, axis=[1, 2]) / total_mass
        y_center = tf.reduce_sum(mask * y_grid, axis=[1, 2]) / total_mass
        
        coords = tf.stack([x_center, y_center], axis=1)
        
        return coords


# ============================================================================
# TRAINING FUNCTION (same as before)
# ============================================================================

def train_complete_manuscript_model(data_dir, output_dir, epochs=30, batch_size=4,
                                   learning_rate=1e-4, seed=42,
                                   spatial_reduction=8, channel_reduction=4,
                                   loss_weights=None):
    """Train with COMPLETE manuscript-consistent implementation (FIXED)"""

    
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Build model
    model, output_names = build_complete_manuscript_model(
        (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        ALL_FEATURES,
        spatial_reduction=spatial_reduction,
        channel_reduction=channel_reduction
    )
    
    # Create losses
    losses, loss_weights_dict, metrics = create_complete_losses(output_names, loss_weights)
    
    # Compile
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
        loss_weights=loss_weights_dict,
        metrics=metrics
    )
    

    
    # Data loaders
    train_path = os.path.join(data_dir, 'train')
    test_path = os.path.join(data_dir, 'test')
    
    print("\nLoading data...")
    base_train_loader = OptimizedDataLoader(
        train_path, ALL_FEATURES, batch_size, (IMAGE_HEIGHT, IMAGE_WIDTH),
        enable_point_dilation=True, dilation_radius=4, is_training=True
    )
    
    base_test_loader = OptimizedDataLoader(
        test_path, ALL_FEATURES, batch_size, (IMAGE_HEIGHT, IMAGE_WIDTH),
        enable_point_dilation=True, dilation_radius=4, is_training=False
    )
    
    train_loader = ManuscriptConsistentDataLoader(base_train_loader)
    test_loader = ManuscriptConsistentDataLoader(base_test_loader)
    
    train_steps = max(1, len(base_train_loader.image_files) // batch_size)
    test_steps = max(1, len(base_test_loader.image_files) // batch_size)
    
    print(f"  Train: {len(base_train_loader.image_files)} images ({train_steps} steps/epoch)")
    print(f"  Test:  {len(base_test_loader.image_files)} images ({test_steps} steps/epoch)\n")
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, f'complete_best_seed{seed}.h5'),
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
    
    # Save results
    results = {
        'model_name': 'complete_manuscript_consistent_FIXED',
        'seed': seed,
        'parameters': model.count_params(),
        'training_time_seconds': training_time,
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'dtype_fix': 'Applied - FP32 coordinates, FP16 masks/heatmaps'
    }
    
    results_path = os.path.join(output_dir, f'results_seed{seed}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train COMPLETE Model (DTYPE FIXED)')
    
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--region_weight', type=float, default=1.0)
    parser.add_argument('--heatmap_weight', type=float, default=1.0)
    parser.add_argument('--coord_weight', type=float, default=0.5)
    
    args = parser.parse_args()
    
    loss_weights = {
        'region': args.region_weight,
        'heatmap': args.heatmap_weight,
        'coord': args.coord_weight,
    }
    
    print("\n" + "="*80)
    print("COMPLETE MANUSCRIPT-CONSISTENT TRAINING (DTYPE FIXED)")
    print("="*80)
    print(f"Fix applied: Mixed precision dtype handling in soft_argmax")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print("="*80)
    
    try:
        results = train_complete_manuscript_model(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            loss_weights=loss_weights
        )
        
        print("\n✅ Training complete with dtype fix!")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
