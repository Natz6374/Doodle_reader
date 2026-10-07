# Doodle Reader

1. `pip install -r requirements.txt`
2. `python train.py`   (one time, downloads EMNIST, saves model.pt)
3. `python app.py`     then open http://localhost:5000

Draw a digit or letter, press Recognise. The canvas is sent to `/api/predict`,
preprocessed to 28x28 (crop, scale, centre by mass) and classified by a small CNN.
