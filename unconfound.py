import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

class UnconfoundedDataset(Dataset):
    def __init__(self, root_dir="/u/student/2023/cs23mtech11019/uncon_small", transform=None):
        """
        Loads data from the unconfounded dataset at the given root_dir.
        Expects structure:
            root_dir/0/*.png (negative)
            root_dir/1/*.png (positive)
        """
        self.samples = []
        self.transform = transform if transform else transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        for label in [0, 1]:
            class_dir = os.path.join(root_dir, str(label))
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} does not exist.")
                continue
            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((fpath, label))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        image = self.transform(image)
        return image, label


def get_unconfounded_loader(batch_size=64, num_workers=4):
    dataset = UnconfoundedDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=False)
    print(f"Unconfounded dataset loaded: {len(dataset)} samples.")
    return dataset, loader


# Example usage
if __name__ == "__main__":
    dataset, loader = get_unconfounded_loader()

    # Iterate through the data
    for images, labels in loader:
        print(f"Batch: {images.shape}, Labels: {labels.shape}")
        break  # Remove this break if you want to iterate over all batches
