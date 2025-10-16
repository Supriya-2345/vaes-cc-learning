import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

class UnconfoundedDataset(Dataset):
    def __init__(self, base_dir="/path/", transform=None):
        """
        Loads data from:
            /raid/ai23resch11004/unconfounded_2/train/images/t0/0/*.png
            /raid/ai23resch11004/unconfounded_2/train/images/t0/1/*.png
        """
        self.samples = []
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        labels = [0, 1]
        for label in labels:
            class_dir = os.path.join(base_dir, str(label))
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} does not exist.")
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    fpath = os.path.join(class_dir, fname)
                    if os.path.isfile(fpath):
                        self.samples.append((fpath, label))

        # Shuffle globally
        random.shuffle(self.samples)
        print(f"Total collected samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        try:
            image = Image.open(img_path).convert('L')
            image = self.transform(image)
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")
            image = torch.zeros((1, 64, 64))
            label = -1
        return image, label


def get_merged_unconfounded_loader(batch_size=64, num_workers=4):
    dataset = UnconfoundedDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True, drop_last=False)
    print(f"Merged unconfounded dataset loaded: {len(dataset)} samples.")
    return dataset, loader


# Example usage
if __name__ == "__main__":
    dataset, loader = get_merged_unconfounded_loader()
    for images, labels in loader:
        print(f"Batch: {images.shape}, Labels: {labels.shape}")
        break
