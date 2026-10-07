"""Train on EMNIST (downloads ~500 MB once).  python train.py [epochs]   (default 12)"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from model import Net, aug

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
B = 256
dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def load(train):  # EMNIST images are stored transposed
    d = datasets.EMNIST('data', 'balanced', train=train, download=True)
    return d.data.transpose(1, 2).float().div(255).unsqueeze(1), d.targets


Xtr, Ytr = load(True)
Xte, Yte = load(False)
if os.path.exists('user_data.npz'):  # your own corrections from the Teach button, repeated 20x
    d = np.load('user_data.npz')
    ux = torch.from_numpy(d['X']).float().div(255).view(-1, 1, 28, 28)
    Xtr = torch.cat([Xtr] + [ux] * 20)
    Ytr = torch.cat([Ytr] + [torch.from_numpy(d['y'])] * 20)
    print(f'Including {len(ux)} of your own samples')

net = Net().to(dev)
opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=EPOCHS * (len(Xtr) // B))
best = 0
for ep in range(EPOCHS):
    net.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(perm) - B + 1, B):
        idx = perm[i:i + B]
        x, y = aug(Xtr[idx].to(dev)), Ytr[idx].to(dev)
        opt.zero_grad()
        F.cross_entropy(net(x), y, label_smoothing=.1).backward()
        opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        ok = sum((net(Xte[i:i + 2000].to(dev)).argmax(1).cpu() == Yte[i:i + 2000]).sum().item()
                 for i in range(0, len(Xte), 2000))
    acc = ok / len(Xte)
    print(f'epoch {ep + 1}/{EPOCHS}  test accuracy {acc:.2%}', flush=True)
    if acc > best:
        best = acc
        torch.save({k: v.cpu() for k, v in net.state_dict().items()}, 'model.pt')
if os.path.exists('model_user.pt'):
    os.remove('model_user.pt')
print(f'Saved model.pt (best {best:.2%})')
