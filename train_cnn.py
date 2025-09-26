import argparse, os, torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from utils import SimpleFolderDataset, default_transforms

def get_model(num_classes=2):
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m

def acc(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    tot_loss = 0.0; tot_acc = 0.0; n = 0
    for X, y, _ in tqdm(loader, desc="train", leave=False):
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward(); opt.step()
        bsz = X.size(0)
        tot_loss += loss.item()*bsz
        tot_acc  += acc(logits, y)*bsz
        n += bsz
    return tot_loss/n, tot_acc/n

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    tot_loss = 0.0; tot_acc = 0.0; n = 0
    for X, y, _ in tqdm(loader, desc="val", leave=False):
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        bsz = X.size(0)
        tot_loss += loss.item()*bsz
        tot_acc  += (logits.argmax(1)==y).float().sum().item()
        n += bsz
    return tot_loss/n, tot_acc/n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="data/train")
    ap.add_argument("--val_dir",   default="data/val")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="models/effb0.pt")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = SimpleFolderDataset(args.train_dir, transform=default_transforms(train=True))
    val_ds   = SimpleFolderDataset(args.val_dir,   transform=default_transforms(train=False))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = get_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_dl, opt, loss_fn, device)
        va_loss, va_acc = evaluate(model, val_dl, loss_fn, device)
        print(f"[epoch {ep}] train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict()}, args.out)
            print(f"  ✓ new best (val acc {best_acc:.4f}) saved -> {args.out}")

if __name__ == "__main__":
    main()
