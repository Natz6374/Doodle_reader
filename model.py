import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

# EMNIST "balanced": digits, capitals, and a few lowercase letters that differ from their capitals
LABELS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'


def blk(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU())


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(blk(1, 32), blk(32, 32), nn.MaxPool2d(2), blk(32, 64), blk(64, 64), nn.MaxPool2d(2),
                            blk(64, 128), nn.MaxPool2d(2), nn.Flatten(), nn.Dropout(.4),
                            nn.Linear(128 * 9, 256), nn.ReLU(), nn.Dropout(.3), nn.Linear(256, len(LABELS)))

    def forward(s, x):
        return s.f(x)


def aug(x):
    """Random thickness, rotation, shear, scale and shift for a batch (n,1,28,28)."""
    n, d = x.size(0), x.device
    r = torch.rand(n, 1, 1, 1, device=d)
    x = torch.where(r < .25, F.max_pool2d(x, 3, 1, 1), torch.where(r > .85, -F.max_pool2d(-x, 3, 1, 1), x))
    a, sc = (torch.rand(n, device=d) - .5) * .5, .85 + .3 * torch.rand(n, device=d)
    sh, t = (torch.rand(n, device=d) - .5) * .4, (torch.rand(n, 2, device=d) - .5) * .25
    th = torch.stack([torch.stack([sc * a.cos(), -sc * a.sin() + sh, t[:, 0]], 1),
                      torch.stack([sc * a.sin(), sc * a.cos(), t[:, 1]], 1)], 1)
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), align_corners=False)


def predict(net, g):
    """g: (n,28,28) float array -> (n,classes) probabilities, averaged over 1px shifts."""
    x = torch.from_numpy(g)[:, None]
    sh = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    with torch.no_grad():
        return (sum(F.softmax(net(torch.roll(x, s, (2, 3))), 1) for s in sh) / len(sh)).numpy()


def preprocess(img):
    """PIL 'L' image (white ink on black) -> 28x28 array. Strokes are thinned and redrawn at a
    uniform width (about 12% of the character size) so any pen size or character size looks alike."""
    a = np.asarray(img, dtype=np.float32) / 255
    ys, xs = np.where(a > 0.24)
    if len(xs) == 0:
        return None
    m = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0.24
    k = max(1, round(max(m.shape) * 0.06))
    m = ndimage.binary_dilation(np.pad(skeletonize(m), k), iterations=k)
    ys, xs = np.where(m)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    s = 20 / max(h, w)
    im = Image.fromarray((m * 255).astype(np.uint8)).resize(
        (max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    b = np.asarray(im, dtype=np.float32) / 255
    g = np.zeros((28, 28), np.float32)
    y0, x0 = (28 - b.shape[0]) // 2, (28 - b.shape[1]) // 2
    g[y0:y0 + b.shape[0], x0:x0 + b.shape[1]] = b
    yy, xx = np.indices(g.shape)
    t = g.sum()
    g = np.roll(g, (round(13.5 - (g * yy).sum() / t), round(13.5 - (g * xx).sum() / t)), (0, 1))
    g = ndimage.gaussian_filter(g, 0.5)
    return (g / g.max()).astype(np.float32)
