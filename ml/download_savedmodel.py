import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TFHUB_CACHE_DIR'] = os.path.join(os.getcwd(), 'tfhub_cache')

import tensorflow as tf
import tensorflow_hub as hub

# ESRGAN 4x super-resolution - matches input [1,50,50,3] -> output [1,200,200,3]
URL = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

print(f"Downloading {URL} ...")
model = hub.load(URL)
print("Loaded. Signatures:", list(model.signatures.keys()) if hasattr(model, 'signatures') else "none")

# Find the cache dir and print it
cache_root = os.environ['TFHUB_CACHE_DIR']
print("Cache root:", cache_root)
for root, dirs, files in os.walk(cache_root):
    for f in files:
        print(os.path.join(root, f))

# Save as SavedModel in our own location
SAVED_MODEL_DIR = "esrgan_saved_model"
tf.saved_model.save(model, SAVED_MODEL_DIR)
print(f"Saved model dumped to {SAVED_MODEL_DIR}")
