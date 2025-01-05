import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import math
import torch
import cv2
import random
import rawpy
import colorsys
from piqa.ssim import ssim
from piqa.utils.functional import gaussian_kernel


PI = math.pi
INV_PI = 1.0 / PI
INV_2PI = 1.0 / (PI * 2)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FLOAT_TYPE = torch.float32

torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(FLOAT_TYPE)


# region image operators ================

# return shape: (h, w, c), in linear space.
def readexr(path):
    assert path[-3:] == 'exr'
    img = torch.from_numpy(cv2.imread(path, cv2.IMREAD_UNCHANGED)[..., [2, 1, 0]]).type(FLOAT_TYPE)
    return img


# return shape: (h, w, c), in sRGB space.
def readimg(path):
    img = torch.from_numpy(cv2.imread(path)[..., [2, 1, 0]]).type(FLOAT_TYPE)
    img = img / 255
    return img


# input shape: (h, w, c), in linear space.
def writeexr(path, img):
    assert path[-3:] == 'exr'
    img = img.type(FLOAT_TYPE).cpu().numpy()
    img = img[..., [2, 1, 0]]
    cv2.imwrite(path, img)


# input shape: (h, w, c), in sRGB space.
def writeimg(path, img):
    img = torch.minimum(img, torch.tensor(1.0))
    img = (img * 255).type(torch.int)
    img = img.cpu().numpy()
    img = img[..., [2, 1, 0]]
    cv2.imwrite(path, img)


def sRGB2linear(img):
    # avoid gradient error.
    tmp = img.clone()
    tmp[img != 0] = torch.pow(img[img != 0], 2.2)
    return tmp

def linear2sRGB(img):
    tmp = img.clone()
    tmp[img != 0] = torch.pow(img[img != 0], 1.0 / 2.2)
    return tmp

# endregion


# region vector ================

# input: v(..., 3)
# returns: len(..., 1)
def length(v):
    v2 = v ** 2
    s = torch.sum(v2, dim=-1, keepdim=True)
    len = torch.sqrt(s)
    return len


# input: (..., 3)
def normalize3(v):
    v2 = torch.square(v)
    s = torch.sum(v2, dim=-1, keepdim=True)
    s[s == 0] = 1  # sqrt(0) is Non-differentiable
    len = torch.sqrt(s)
    return v / len


# input: (..., 3)
def dot(a, b):
    c = a * b
    c = torch.sum(c, dim=-1, keepdim=True)
    return c


# inputs: (..., 3)
def to_local(wi, s, t, n):
    x = torch.sum(wi * s, dim=len(wi.shape) - 1)
    y = torch.sum(wi * t, dim=len(wi.shape) - 1)
    z = torch.sum(wi * n, dim=len(wi.shape) - 1)
    wi = torch.stack((x, y, z), dim=len(wi.shape) - 1)
    return wi

def build_orthbasis(n):
    sn = n.clone()
    sn[n[..., 2] < -0.9999] = torch.tensor([0, 0, 1.0], device=DEVICE)
    a = 1.0 / (1.0 + sn[..., 2])
    b = -sn[..., 0] * sn[..., 1] * a
    s = torch.stack([1 - (sn[..., 0] ** 2) * a, b, -sn[..., 0]], dim=-1)
    t = torch.stack([b, 1 - (sn[..., 1] ** 2) * a, -sn[..., 1]], dim=-1)

    s[n[..., 2] < -0.9999] = torch.tensor([0, -1.0, 0], device=DEVICE)
    t[n[..., 2] < -0.9999] = torch.tensor([-1.0, 0, 0], device=DEVICE)
    return s, t


def vector2spectrum(w):
    return w * 0.5 + 0.5

def spectrum2vector(w):
    return normalize3(w * 2 - 1)


# endregion


# region other ================
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# endregion