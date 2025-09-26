import os, glob, torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

IM_SIZE = 224

def default_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IM_SIZE, IM_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1,0.1,0.1,0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IM_SIZE, IM_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

class SimpleFolderDataset(Dataset):
    """
    Expects:
      root/
        real/  *.jpg|*.png|...
        fake/  *.jpg|*.png|...
    """
    def __init__(self, root, transform=None):
        self.samples, self.labels = [], []
        for label, sub in enumerate(["real","fake"]):
            subdir = os.path.join(root, sub)
            if not os.path.isdir(subdir): 
                continue
            for p in glob.glob(os.path.join(subdir, "*")):
                if p.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
                    self.samples.append(p)
                    self.labels.append(label)
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {root}/(real|fake)")
        self.transform = transform or default_transforms(train=True)

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        y = self.labels[idx]
        return img, torch.tensor(y, dtype=torch.long), path
