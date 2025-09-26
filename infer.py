import argparse, os, glob, torch
from PIL import Image
from torchvision import transforms, models
from utils import default_transforms

def list_images(path):
    if os.path.isdir(path):
        exts = (".jpg",".jpeg",".png",".bmp",".webp")
        return [p for p in glob.glob(os.path.join(path,"**","*"), recursive=True) if p.lower().endswith(exts)]
    return [path]

def load_model(weights_path):
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 2)
    state = torch.load(weights_path, map_location="cpu")["model"]
    m.load_state_dict(state)
    m.eval()
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="image file or folder")
    ap.add_argument("--weights", default="models/effb0.pt")
    args = ap.parse_args()

    model = load_model(args.weights)
    tfm = default_transforms(train=False)

    for p in list_images(args.path):
        img = Image.open(p).convert("RGB")
        x = tfm(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            prob_fake = torch.softmax(logits, dim=1)[0,1].item()
        label = "fake" if prob_fake >= 0.5 else "real"
        print(f"{p}: {label} (fake_prob={prob_fake:.3f})")

if __name__ == "__main__":
    main()
