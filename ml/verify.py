import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

interp = tf.lite.Interpreter(model_path="esrgan_int8.tflite")
interp.allocate_tensors()

inp = interp.get_input_details()
out = interp.get_output_details()

print("=== QUANTIZED MODEL ===")
for d in inp:
    print(f"INPUT  name={d['name']} shape={d['shape']} dtype={d['dtype']} quant={d['quantization']}")
for d in out:
    print(f"OUTPUT name={d['name']} shape={d['shape']} dtype={d['dtype']} quant={d['quantization']}")

dtypes = {}
for t in interp.get_tensor_details():
    s = str(t['dtype'])
    dtypes[s] = dtypes.get(s, 0) + 1
print("Tensor dtypes:", dtypes)

# Run a test inference
sample = np.random.randint(-128, 128, size=inp[0]['shape'], dtype=np.int8)
interp.set_tensor(inp[0]['index'], sample)
interp.invoke()
y = interp.get_tensor(out[0]['index'])
print(f"Inference OK. Output shape={y.shape} dtype={y.dtype} min={y.min()} max={y.max()}")

size = os.path.getsize("esrgan_int8.tflite")
orig = os.path.getsize("esrgan.tflite")
print(f"\nSizes: original={orig} bytes, int8={size} bytes, ratio={size/orig:.2%}")
