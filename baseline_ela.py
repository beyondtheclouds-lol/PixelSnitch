import argparse, glob, os, numpy as np
from PIL import Image, ImageChops, ImageEnhance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def ela_vec(p, q=90, scale=10.0, size=(128,128)):
    im = Image.open(p).convert("RGB")
    tmp = p + ".ela.jpg"
    im.save(tmp, "JPEG", quality=q)
    im2 = Image.open(tmp)
    ela = ImageChops.difference(im, im2)
    ela = ImageEnhance.Brightness(ela).enhance(scale)
    ela = ela.resize(size)
    os.remove(tmp)
    return np.asarray(ela).astype(np.float32).flatten()/255.0

def load_split(root):
    X, y = [], []
    for label, sub in enumerate(["real","fake"]):
        subdir = os.path.join(root, sub)
        if not os.path.isdir(subdir): 
            continue
        for p in glob.glob(os.path.join(subdir, "*")):
            if p.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
                try:
                    X.append(ela_vec(p))
                    y.append(label)
                except Exception:
                    pass
    return np.vstack(X), np.array(y)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="data/train")
    ap.add_argument("--val_dir", default="data/val")
    args = ap.parse_args()

    Xtr, ytr = load_split(args.train_dir)
    Xva, yva = load_split(args.val_dir)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xva)
    acc = accuracy_score(yva, preds)
    print(f"ELA baseline val acc: {acc:.4f}")

if __name__ == "__main__":
    main()
