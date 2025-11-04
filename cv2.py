# cv2.py — minimal stub for headless environments (no libGL)
import numpy as np

# --- constants ---
COLOR_BGR2RGB = 4
COLOR_RGB2BGR = 5
INTER_LINEAR = 1
FONT_HERSHEY_SIMPLEX = 0
LINE_AA = 16

# --- basic image ops ---
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
    raise NotImplementedError("imread unavailable in headless mode")

def imwrite(*args, **kwargs):
    raise NotImplementedError("imwrite unavailable in headless mode")

# --- threading & misc ---
def setNumThreads(n: int):
    pass

def getBuildInformation():
    return "Headless OpenCV Stub — no GUI/OpenGL features available"

# --- GUI stubs ---
def imshow(window_name, img):
    # do nothing, silently skip
    return None

def waitKey(delay=0):
    # return arbitrary int to satisfy callers
    return -1

def destroyAllWindows():
    pass
