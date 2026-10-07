"""Backend: python app.py  ->  http://localhost:5000"""
import base64, io, os
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from model import Net, LABELS, preprocess
from reader import read
from model import aug
import numpy as np

app = Flask(__name__, static_folder='static')
net = Net()
WEIGHTS = next((p for p in ('model_user.pt', 'model.pt') if os.path.exists(p)), None)
trained = WEIGHTS is not None
if trained:
    try:
        net.load_state_dict(torch.load(WEIGHTS, map_location='cpu'))
    except RuntimeError:
        print('Saved weights are from an older version. Delete model*.pt and run python train.py')
        trained = False
net.eval()


@app.get('/')
def index():
    return send_from_directory('static', 'index.html')


@app.get('/api/health')
def health():
    return jsonify(trained=trained, classes=len(LABELS), params=sum(p.numel() for p in net.parameters()))


@app.post('/api/predict')
def predict():
    if not trained:
        return jsonify(error='No trained model yet. Run: python train.py'), 503
    try:
        raw = request.get_json()['image'].split(',', 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(raw))).convert('L')
    except Exception:
        return jsonify(error='Could not read the image.'), 400
    g = preprocess(img)
    if g is None:
        return jsonify(error='The canvas is empty. Draw something first.'), 400
    with torch.no_grad():
        p = F.softmax(net(torch.from_numpy(g)[None, None]), 1)[0]
    t = p.topk(5)
    return jsonify(top=[{'label': LABELS[i], 'prob': float(v)} for v, i in zip(t.values, t.indices)],
                   preview=(g * 255).astype(int).flatten().tolist())


@app.post('/api/read')
def read_text():
    if not trained:
        return jsonify(error='No trained model yet. Run: python train.py'), 503
    try:
        raw = request.get_json()['image'].split(',', 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(raw)))
    except Exception:
        return jsonify(error='Could not read the image.'), 400
    res = read(img, net)
    if res is None:
        return jsonify(error='The canvas is empty. Write something first.'), 400
    return jsonify(res)


@app.post('/api/teach')
def teach():
    if not trained:
        return jsonify(error='No trained model yet. Run: python train.py'), 503
    xs, ys = [], []
    for smp in request.get_json().get('samples', []):
        l = smp['label'] if smp['label'] in LABELS else smp['label'].upper()
        if l in LABELS and len(smp['preview']) == 784:
            xs.append(smp['preview'])
            ys.append(LABELS.index(l))
    if not xs:
        return jsonify(error='Nothing to learn from.'), 400
    X, Y = np.array(xs, np.uint8), np.array(ys)
    if os.path.exists('user_data.npz'):
        d = np.load('user_data.npz')
        X, Y = np.vstack([d['X'], X]), np.concatenate([d['y'], Y])
    np.savez('user_data.npz', X=X, y=Y)
    x, y = torch.from_numpy(X).float().div(255).view(-1, 1, 28, 28), torch.from_numpy(Y)
    opt = torch.optim.Adam(list(net.f[-1].parameters()) + list(net.f[-4].parameters()), 5e-4)
    for _ in range(60):  # fine-tune only the last two layers on your samples
        i = torch.randint(len(x), (min(32, len(x)),))
        opt.zero_grad()
        F.cross_entropy(net(aug(x[i])), y[i]).backward()
        opt.step()
    net.eval()
    torch.save(net.state_dict(), 'model_user.pt')
    return jsonify(added=len(xs), total=len(X))


if __name__ == '__main__':
    app.run(port=5000, debug=False)
