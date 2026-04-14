import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_PATH = "esrgan.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=== INPUT DETAILS ===")
for d in input_details:
    print(f"name={d['name']} shape={d['shape']} dtype={d['dtype']} quant={d['quantization']}")

print("\n=== OUTPUT DETAILS ===")
for d in output_details:
    print(f"name={d['name']} shape={d['shape']} dtype={d['dtype']} quant={d['quantization']}")

# Inspect all tensors
print("\n=== TENSOR DTYPE SUMMARY ===")
dtype_counts = {}
for t in interpreter.get_tensor_details():
    dt = str(t['dtype'])
    dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
for dt, c in dtype_counts.items():
    print(f"  {dt}: {c}")

# Determine float32
is_f32 = all(d['dtype'] == np.float32 for d in input_details + output_details)
print(f"\nIS FLOAT32 MODEL: {is_f32}")
