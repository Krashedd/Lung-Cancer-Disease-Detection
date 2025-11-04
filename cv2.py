# cv2.py — full headless stub for Streamlit / libGL-free environments
# Allows Ultralytics and other OpenCV users to import without errors.

import numpy as np
from PIL import Image

# ======================
# Common constants
# ======================
COLOR_BGR2RGB = 4
COLOR_RGB2BGR = 5
INTER_LINEAR = 1
INTER_NEAREST = 0
FONT_HERSHEY_SIMPLEX = 0
LINE_AA = 16
IMREAD_COLOR = 1
IMREAD_GRAYSCALE = 0
IMWRITE_JPEG_QUALITY = 1
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4

# ======================
# Basic image functions
# ======================
def cvtColor(img, code):
    if code in (COLOR_BGR2RGB, COLOR_RGB2BGR):
        return img[..., ::-1]
    return img

def resize(img, dsize, interpolation=INTER_LINEAR):
    pil = Image.fromarray(img)
    pil = pil.resize(dsize, Image.BILINEAR)
    return np.array(pil)

def imread(path, flags=IMREAD_COLOR):
    raise NotImplementedError("cv2.imread not available in headless stub")

def imwrite(path, img, params=None):
    raise NotImplementedError("cv2.imwrite not available in headless stub")

def addWeighted(src1, alpha, src2, beta, gamma):
    return np.clip(src1 * alpha + src2 * beta + gamma, 0, 255).astype(np.uint8)

# ======================
# Threading / info
# ======================
def setNumThreads(n: int):
    pass

def getBuildInformation():
    return "OpenCV headless stub (no GUI / no libGL)."

# ======================
# GUI placeholders
# ======================
def imshow(winname, img):
    # do nothing
    return None

def waitKey(delay=0):
    # just return -1 as dummy
    return -1

def destroyAllWindows():
    pass

def namedWindow(winname, flags=0):
    pass

def destroyWindow(winname):
    pass

# ======================
# Misc placeholders
# ======================
def rectangle(*args, **kwargs): pass
def putText(*args, **kwargs): pass
def circle(*args, **kwargs): pass
def line(*args, **kwargs): pass
def VideoCapture(*args, **kwargs): raise NotImplementedError("VideoCapture unavailable in stub")
def flip(img, flipCode):
    return np.flip(img, axis=flipCode if flipCode in [0, 1] else -1)
