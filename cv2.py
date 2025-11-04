# cv2.py — minimal stub for headless environments without libGL.so.1
import numpy as np

# --- basic constants ---
COLOR_BGR2RGB = 4
COLOR_RGB2BGR = 5
INTER_LINEAR = 1
FONT_HERSHEY_SIMPLEX = 0
LINE_AA = 16

# --- minimal image ops ---
def cvtColor(img, code):
    if code in (COLOR_BGR2RGB, COLOR_RGB2BGR):
        return img[..., ::-1]
    return img

def resize(img, dsize, interpolation=INTER_LINEAR):
    from PIL import Image
    pil = Image.fromarray(img)
    pil = pil.resize(dsize, Image.BILINEAR)
    return np.array(pil)

def imread(*args, **kwargs):
    raise NotImplementedError("imread unavailable in this environment")

def imwrite(*args, **kwargs):
    raise NotImplementedError("imwrite unavailable in this environment")

# --- threading & compatibility hooks ---
def setNumThreads(n: int):
    # silently ignore; avoids AttributeError in ultralytics.utils
    pass

def getBuildInformation():
    return "Headless OpenCV Stub — no GUI/OpenGL features available"
