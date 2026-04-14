import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

SAVED_MODEL_DIR = "esrgan_saved_model"
OUT = "esrgan_int8.tflite"

# Representative dataset: ESRGAN input is 50x50x3 RGB image in [0, 255] float
# Use deterministic pseudo-random natural-image-like patches
def rep_data_gen():
    rng = np.random.default_rng(42)
    for _ in range(100):
        # Generate image-like data with smooth gradients + noise to emulate natural imagery
        base = rng.uniform(0, 255, size=(50, 50, 3)).astype(np.float32)
        # Low-pass via box blur to make it less noisy
        import scipy.ndimage as ndi  # not used - keep pure numpy fallback below
        yield [base[np.newaxis, ...]]

# Simpler representative dataset without scipy
def rep_data_gen_simple():
    rng = np.random.default_rng(42)
    for _ in range(100):
        img = rng.uniform(0, 255, size=(1, 50, 50, 3)).astype(np.float32)
        yield [img]

# Load SavedModel and build a concrete function with fixed 50x50x3 input
loaded = tf.saved_model.load(SAVED_MODEL_DIR)
concrete_fn = loaded.signatures["serving_default"]

@tf.function(input_signature=[tf.TensorSpec(shape=[1, 50, 50, 3], dtype=tf.float32)])
def fixed_shape_fn(x):
    return concrete_fn(input_0=x)

cf = fixed_shape_fn.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([cf], loaded)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data_gen_simple
# Full integer quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

print("Converting to int8...")
tflite_quant = converter.convert()

with open(OUT, "wb") as f:
    f.write(tflite_quant)

print(f"Wrote {OUT} ({len(tflite_quant)} bytes)")
