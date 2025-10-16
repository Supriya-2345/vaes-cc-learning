import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os


class LatentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=2):
        super(LatentClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_classifier(dataloader, input_dim, device='cuda', epochs=2000, model = None):
    model = LatentClassifier(input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for zs, labels in dataloader:
            zs, labels = zs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(zs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * zs.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += zs.size(0)
            

        # print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/total_samples:.4f} - Accuracy: {total_correct/total_samples:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_path', type=str, required=True,
                        help='Path to the latent_*.pth file')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    print("Parsed arguments:", args)  # <-- added this line


    assert os.path.isfile(args.latent_path), f"File not found: {args.latent_path}"
    print(f"Loading latent representations from {args.latent_path}...")

    data = torch.load(args.latent_path)
    z = data['z']
    y = data['y']
    input_dim = z.size(1)
    num_classes = len(torch.unique(y))

    dataset = TensorDataset(z, y)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Training classifier on latent space (z_dim = {input_dim})...")
    model = train_classifier(dataloader, input_dim=input_dim, device=args.device, epochs=2000)

    # Evaluate final accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for zs, labels in dataloader:
            zs, labels = zs.to(args.device), labels.to(args.device)
            outputs = model(zs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += zs.size(0)
    print(f"\nFinal classification accuracy on training data: {correct/total:.4f}")


if __name__ == '__main__':
    main()




# # Train or fine-tune classifier on top of latent Zs
# print(f"\nTraining/Fine-tuning classifier for env {args.env_id}...")

# from classifier_module import LatentClassifier, train_classifier  # Make sure to import this properly

# # Load Zs and labels
# z_tensor = torch.cat(all_z, dim=0)
# y_tensor = torch.cat(all_y, dim=0)
# input_dim = z_tensor.size(1)
# num_classes = len(torch.unique(y_tensor))

# classifier = LatentClassifier(input_dim, num_classes=num_classes).cuda()

# # Load classifier from previous env if exists
# if args.env_id > 0:
#     prev_classifier_path = os.path.join(args.save, f'classifier_env{args.env_id - 1}.pth')
#     if os.path.isfile(prev_classifier_path):
#         classifier.load_state_dict(torch.load(prev_classifier_path))
#         print(f"Loaded classifier from {prev_classifier_path}")

# # Fine-tune on current latent data
# classifier_dataset = TensorDataset(z_tensor, y_tensor)
# classifier_loader = DataLoader(classifier_dataset, batch_size=128, shuffle=True)

# classifier = train_classifier(classifier_loader, input_dim=input_dim, device='cuda', epochs=20, model=classifier)

# # Save classifier
# classifier_path = os.path.join(args.save, f'classifier_env{args.env_id}.pth')
# torch.save(classifier.state_dict(), classifier_path)
# print(f"Saved classifier to {classifier_path}")
