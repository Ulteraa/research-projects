import cv2, numpy as np, scipy
from calibgraph.geometry.se3 import compose,inverse,random_transform
rng=np.random.default_rng(42); T=random_transform(rng,max_rotation_deg=45,max_translation=.25)
err=np.linalg.norm(compose(T,inverse(T))-np.eye(4))
print("Environment check")
print("-----------------")
print("NumPy:",np.__version__)
print("SciPy:",scipy.__version__)
print("OpenCV:",cv2.__version__)
print(f"SE(3) inverse round-trip error: {err:.3e}")
if not hasattr(cv2,"calibrateHandEye"): raise RuntimeError("cv2.calibrateHandEye unavailable")
print("cv2.calibrateHandEye: available")
print("Phase 1 environment: PASS")
