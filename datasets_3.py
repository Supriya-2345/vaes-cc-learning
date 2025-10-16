import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

# Dataset class for grayscale image loading
class Concon(Dataset):
    def __init__(self, env_dirs=None, transform=None):
        """
        env_dirs: A list of tuples [(env_id, path_to_env_data)]
        If None, uses default hardcoded paths.
        """
        if env_dirs is None:
            env_dirs = [
                (0, "/u/student/2023/cs23mtech11019/data_small_0"),
                (1, "/u/student/2023/cs23mtech11019/data_small_1"),
                (2, "/u/student/2023/cs23mtech11019/data_small_2")
            ]

        self.samples = []
        self.transform = transform if transform else transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        for env_id, root_dir in env_dirs:
            for label in [0, 1]:
                class_dir = os.path.join(root_dir, str(label))
                if not os.path.exists(class_dir):
                    print(f"Warning: {class_dir} does not exist.")
                    continue
                for fname in os.listdir(class_dir):
                    fpath = os.path.join(class_dir, fname)
                    if os.path.isfile(fpath) and fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((fpath, label, env_id))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label, env_id = self.samples[index]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        image = self.transform(image)
        return image, label, env_id


def get_env_datasets_and_loaders(batch_size=32, num_workers=4):
    env_dirs = [
        (0, "/u/student/2023/cs23mtech11019/data_small_0"),
        (1, "/u/student/2023/cs23mtech11019/data_small_1"),
        (2, "/u/student/2023/cs23mtech11019/data_small_2")
    ]

    env_datasets = {}
    env_loaders = {}

    for env_id, root_dir in env_dirs:
        dataset = Concon([(env_id, root_dir)])  # Use the modified Concon class
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=False)
        env_datasets[env_id] = dataset
        env_loaders[env_id] = loader
        print(f"Env {env_id} -> {len(dataset)} samples loaded from {root_dir}")

    return env_datasets, env_loaders


# Example usage in a training loop
if __name__ == "__main__":
    # Replace these with your actual args and model trainer
    class DummyArgs:
        num_samples = 100
        use_memory = 'yes'

    args = DummyArgs()

    env_datasets, env_loaders = get_env_datasets_and_loaders()

    # Dummy memory dict to simulate update_memory()
    task_memory = {0: {'x': [], 'y': [], 'tt': [], 'td': []},
                   1: {'x': [], 'y': [], 'tt': [], 'td': []},
                   2: {'x': [], 'y': [], 'tt': [], 'td': []}}

    def update_memory(task_id, loader, dataset, num_samples=100):
        sampled_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        for idx in sampled_indices:
            img_path, label, env_id = dataset.samples[idx]
            task_memory[task_id]['x'].append(img_path)
            task_memory[task_id]['y'].append(label)
            task_memory[task_id]['tt'].append(env_id)
            task_memory[task_id]['td'].append(env_id + 1)

    # Sequential training simulation
    for task_id in [0, 1, 2]:
        print(f"\n--- Training on Env {task_id} ---")
        train_loader = env_loaders[task_id]
        dataset = env_datasets[task_id]
        # Train your model here using train_loader...

        if task_id in [1, 2] and args.use_memory == 'yes':
            update_memory(task_id, train_loader, dataset, args.num_samples)
            print(f"Memory updated for Env {task_id}: {len(task_memory[task_id]['x'])} samples stored.")


# import os
# import random
# from PIL import Image
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# import torch


# # class Concon(Dataset):
# class Concon(Dataset):
#     def __init__(self, env_dirs=None, transform=None):
#         """
#         env_dirs: A list of tuples [(env_id, path_to_env_data)]
#         If None, uses default hardcoded paths.
#         """
#         if env_dirs is None:
#             env_dirs = [
#                 (0, "/u/student/2023/cs23mtech11019/data_small_0"),
#                 (1, "/u/student/2023/cs23mtech11019/data_small_1"),
#                 (2, "/u/student/2023/cs23mtech11019/data_small_2")
#             ]

#         self.samples = []
#         self.transform = transform if transform else transforms.Compose([
#             transforms.Resize((64, 64)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,))
#         ])

#         for env_id, root_dir in env_dirs:
#             for label in [0, 1]:
#                 class_dir = os.path.join(root_dir, str(label))
#                 if not os.path.exists(class_dir):
#                     print(f"Warning: {class_dir} does not exist.")
#                     continue
#                 for fname in os.listdir(class_dir):
#                     fpath = os.path.join(class_dir, fname)
#                     if os.path.isfile(fpath) and fname.lower().endswith(('.png', '.jpg', '.jpeg')):
#                         self.samples.append((fpath, label, env_id))

#         random.shuffle(self.samples)


#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, index):
#         img_path, label, env_id = self.samples[index]
#         image = Image.open(img_path).convert('RGB')
#         image = self.transform(image)
#         return image, label, env_id


# def get_env_datasets_and_loaders(batch_size=32, num_workers=4):
#     env_dirs = [
#         (0, "/u/student/2023/cs23mtech11019/data_small_0"),
#         (1, "/u/student/2023/cs23mtech11019/data_small_1"),
#         (2, "/u/student/2023/cs23mtech11019/data_small_2")
#     ]

#     env_datasets = {}
#     env_loaders = {}

#     for env_id, root_dir in env_dirs:
#         dataset = EnvConcatDataset([(env_id, root_dir)])
#         loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
#                             num_workers=num_workers, pin_memory=True, drop_last=False)
#         env_datasets[env_id] = dataset
#         env_loaders[env_id] = loader
#         print(f"Env {env_id} -> {len(dataset)} samples loaded from {root_dir}")

#     return env_datasets, env_loaders


# # Example usage in a training loop
# if __name__ == "__main__":
#     # Replace these with your actual args and model trainer
#     class DummyArgs:
#         num_samples = 100
#         use_memory = 'yes'

#     args = DummyArgs()

#     env_datasets, env_loaders = get_env_datasets_and_loaders()

#     # Dummy memory dict to simulate update_memory()
#     task_memory = {0: {'x': [], 'y': [], 'tt': [], 'td': []},
#                    1: {'x': [], 'y': [], 'tt': [], 'td': []},
#                    2: {'x': [], 'y': [], 'tt': [], 'td': []}}

#     def update_memory(task_id, loader, dataset, num_samples=100):
#         sampled_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
#         for idx in sampled_indices:
#             img_path, label, env_id = dataset.samples[idx]
#             task_memory[task_id]['x'].append(img_path)
#             task_memory[task_id]['y'].append(label)
#             task_memory[task_id]['tt'].append(env_id)
#             task_memory[task_id]['td'].append(env_id + 1)

#     # Sequential training simulation
#     for task_id in [0, 1, 2]:
#         print(f"\n--- Training on Env {task_id} ---")
#         train_loader = env_loaders[task_id]
#         dataset = env_datasets[task_id]
#         # Train your model here using train_loader...

#         if task_id in [1, 2] and args.use_memory == 'yes':
#             update_memory(task_id, train_loader, dataset, args.num_samples)
#             print(f"Memory updated for Env {task_id}: {len(task_memory[task_id]['x'])} samples stored.")

