import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch


class Concon(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform if transform else transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if len(sample) != 3:
            raise ValueError(f"Bad sample at index {index}: {sample}")
        
        img_path, label, env_id = sample
        image = Image.open(img_path).convert('L')
        image = self.transform(image)
        return image, label, env_id


def collect_samples(env_id, root_dir, max_samples_per_class=None):
    samples = []
    for label in [0, 1]:
        class_dir = os.path.join(root_dir, str(label))
        if not os.path.exists(class_dir):
            continue
        fnames = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if max_samples_per_class:
            fnames = random.sample(fnames, min(len(fnames), max_samples_per_class))
        for fname in fnames:
            fpath = os.path.join(class_dir, fname)
            samples.append((fpath, label, env_id))
    return samples


def get_disjoint_datasets_and_loaders(batch_size=32, num_workers=4):
    base_path = "/path/"
    splits = ['train', 'val', 'test']

    datasets = {split: {} for split in splits}
    loaders = {split: {} for split in splits}

    for split in splits:
        for env_id in [0, 1, 2]:
            main_dir = os.path.join(base_path, split, "images", f"t{env_id}")
            samples = collect_samples(env_id, main_dir)

            # Add mixed samples only for train split
            if split == 'train':
                if env_id == 1:
                    extra_dir = os.path.join(base_path, split, "images", "t0")
                    samples += collect_samples(0, extra_dir, max_samples_per_class=100)
                elif env_id == 2:
                    samples += collect_samples(0, os.path.join(base_path, split, "images", "t0"), max_samples_per_class=40)
                    samples += collect_samples(1, os.path.join(base_path, split, "images", "t1"), max_samples_per_class=60)

            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            dataset = Concon(samples, transform=transform)  # ✅ use the correct class here
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'),
                                num_workers=num_workers, pin_memory=True, drop_last=False)

            datasets[split][env_id] = dataset
            loaders[split][env_id] = loader
            print(f"{split.capitalize()} - Env {env_id}: {len(dataset)} samples (including mixed)")

    return datasets, loaders


if __name__ == "__main__":
    datasets, loaders = get_disjoint_datasets_and_loaders()

    # Example access:
    for split in ['train', 'val', 'test']:
        for env in [0, 1, 2]:
            print(f"[{split}] Env {env}: {len(datasets[split][env])} samples")
