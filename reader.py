"""Sentence reader: split ink into characters, group into words and lines, classify each character."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from model import LABELS, preprocess, predict

SWAP = dict(zip('O0I1S5B8Z2G6', '0O1I5S8B2Z6G'))  # look-alike pairs resolved by context


def _merge(bx):
    """Join boxes that sit on top of each other (dot of i, bars of =, colon...)."""
    while True:
        for i in range(len(bx)):
            for j in range(i + 1, len(bx)):
                a, b = bx[i], bx[j]
                ov = min(a[2], b[2]) - max(a[0], b[0])
                gap = max(a[1], b[1]) - min(a[3], b[3])
                if ov > 0.5 * min(a[2] - a[0], b[2] - b[0]) and gap < 0.5 * max(a[3] - a[1], b[3] - b[1]):
                    bx[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    bx.pop(j)
                    break
            else:
                continue
            break
        else:
            return bx


def segment(a):
    """uint8 array (white ink on black) -> lines -> words -> boxes (x0, y0, x1, y1)."""
    m = a > 61
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    if n == 0:
        return []
    bx = [(s[1].start, s[0].start, s[1].stop, s[0].stop) for s in ndimage.find_objects(lab)]
    area = ndimage.sum(m, lab, range(1, n + 1))
    bx = _merge([b for b, r in zip(bx, area) if r >= 40])[:80]
    lines = []
    for b in sorted(bx, key=lambda b: b[1]):
        for ln in lines:
            y0, y1 = min(c[1] for c in ln), max(c[3] for c in ln)
            if min(y1, b[3]) - max(y0, b[1]) > 0.5 * min(y1 - y0, b[3] - b[1]):
                ln.append(b)
                break
        else:
            lines.append([b])
    out = []
    for ln in sorted(lines, key=lambda ln: min(c[1] for c in ln)):
        ln.sort(key=lambda c: c[0])
        mh = float(np.median([c[3] - c[1] for c in ln]))
        words = [[ln[0]]]
        for p, c in zip(ln, ln[1:]):
            if c[0] - p[2] > 0.55 * mh:
                words.append([c])
            else:
                words[-1].append(c)
        out.append(words)
    return out


def read(img, net):
    a = np.asarray(img.convert('L'))
    seg = segment(a)
    if not seg:
        return None
    L = []
    for ln in seg:
        hs = [b[3] - b[1] for w in ln for b in w]
        mh, top = float(np.median(hs)), max(hs)
        L.append([[dict(box=b, h=b[3] - b[1], lmax=top, dot=(b[3] - b[1]) < .3 * mh and (b[2] - b[0]) < .3 * mh)
                   for b in w] for w in ln])
    flat = []
    for ln in L:
        for w in ln:
            for it in w:
                if not it['dot']:
                    x0, y0, x1, y1 = it['box']
                    it['g'] = preprocess(Image.fromarray(a[y0:y1, x0:x1]))
                    if it['g'] is None:
                        it['dot'] = True
                    else:
                        flat.append(it)
    if flat:
        P = predict(net, np.stack([i['g'] for i in flat]))
        for it, p in zip(flat, P):
            it['p'] = p
            it['label'] = LABELS[int(p.argmax())]
    for ln in L:
        for w in ln:
            raw = [it.get('label') for it in w]
            for i, it in enumerate(w):
                if it['dot']:
                    it['label'] = '.'
                    continue
                lab = it['label']
                it['top'] = LABELS.index(lab)
                if lab in SWAP:
                    alt = SWAP[lab]
                    nb = {raw[j].isdigit() for j in (i - 1, i + 1) if 0 <= j < len(w) and raw[j]}
                    if len(nb) == 1 and nb != {lab.isdigit()} and it['p'][LABELS.index(alt)] >= .35 * it['p'][it['top']]:
                        it['label'] = alt
                if it['label'].isupper() and it['h'] < .62 * it['lmax']:
                    it['label'] = it['label'].lower()
    chars, probs = [], []
    for ln in L:
        for w in ln:
            for it in w:
                ch = dict(label=it['label'], box=[int(v) for v in it['box']], prob=1.0, preview=None, alt='')
                if not it['dot']:
                    p = it['p']
                    ch['prob'] = float(p[LABELS.index(it['label'].upper())]) if it['label'].upper() in LABELS else float(p.max())
                    ch['preview'] = (it['g'] * 255).astype(int).flatten().tolist()
                    ch['alt'] = ', '.join(f'{LABELS[i]} {p[i]:.0%}' for i in p.argsort()[::-1][:3])
                    probs.append(ch['prob'])
                chars.append(ch)
    text = '\n'.join(' '.join(''.join(it['label'] for it in w) for w in ln) for ln in L)
    return dict(text=text, conf=float(np.mean(probs)) if probs else 0.0, count=len(chars), chars=chars)
