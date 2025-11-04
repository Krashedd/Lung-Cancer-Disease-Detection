# cv2.py — tiny stub to satisfy ultralytics in environments without libGL
import numpy as np

COLOR_BGR2RGB = 4
COLOR_RGB2BGR = 5
INTER_LINEAR = 1

def cvtColor(img, code):
    if code in (COLOR_BGR2RGB, COLOR_RGB2BGR):
        return img[..., ::-1]
    raise NotImplementedError("cvtColor code not supported in stub")

def resize(img, dsize, interpolation=INTER_LINEAR):
    # dsize is (width, height)
    from PIL import Image
    pil = Image.fromarray(img)
    pil = pil.resize(dsize, Image.BILINEAR)
    return np.array(pil)

def imread(*args, **kwargs):
    raise NotImplementedError("imread not available in this environment")

def imwrite(*args, **kwargs):
    raise NotImplementedError("imwrite not available in this environment")

# some attrs ulralytics/openCV code sometimes touches
FONT_HERSHEY_SIMPLEX = 0
LINE_AA = 16
