
import traceback
import onnxruntime
import sys

print(f"Python version: {sys.version}")
print(f"ONNXRuntime version: {onnxruntime.__version__}")
print(f"Available providers: {onnxruntime.get_available_providers()}")

try:
    from insightface.app import FaceAnalysis
    print("Initializing FaceAnalysis (without allowed_modules restriction)...")
    # Remove allowed_modules to load all default models (including detection)
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("Preparing FaceAnalysis...")
    app.prepare(ctx_id=0)
    print("Success! Models loaded:", app.models.keys())
except Exception:
    traceback.print_exc()
