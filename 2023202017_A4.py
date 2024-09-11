import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm

# Dataset class
class AgeDataset(Dataset):
    def __init__(self, data_path, annot_path, transform=None):
        self.ann = pd.read_csv(annot_path)
        self.data_path = data_path
        self.transform = transform

    def __getitem__(self, index):
        file_name = self.ann['file_id'][index]
        img = Image.open(f"{self.data_path}/{file_name}").convert('RGB')
        age = self.ann['age'][index]
        if self.transform:
            img = self.transform(img)
        return img, age

    def __len__(self):
        return len(self.ann)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()  
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # Output: 16 x 112 x 112
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # Output: 32 x 56 x 56
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # Output: 64 x 28 x 28
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 512),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 64 * 28 * 28),
            nn.ReLU(),
            nn.Unflatten(1, (64, 28, 28)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class ResNet18PlusAE(nn.Module):
    def __init__(self):
        super(ResNet18PlusAE, self).__init__() 
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Removing the final fully connected layer

        self.autoencoder = AutoEncoder()
        self.regressor = nn.Linear(512, 1)  # Using the encoded features

    def forward(self, x):
        _, encoded = self.autoencoder(x)  # Get encoded features from autoencoder
        features = self.resnet(x)  # Get features from ResNet18
        combined_features = features + encoded  # Combine features
        age = self.regressor(combined_features)
        return age

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
train_path = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/train'
train_ann = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/train.csv'


# DataLoader setup
train_dataset = AgeDataset(train_path, train_ann, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model, loss, and optimizer setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18PlusAE().to(device)
criterion_ae = nn.MSELoss()
criterion_age = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
def train(model, loader, optimizer, criterion_ae, criterion_age, device, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, ages in tqdm(loader):
            images, ages = images.to(device), ages.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            decoded_images, encoded_features = model.autoencoder(images)
            age_predictions = model(images)
            loss_ae = criterion_ae(decoded_images, images)
            loss_age = criterion_age(age_predictions, ages)
            loss = loss_ae + loss_age
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Avg Loss = {total_loss / len(loader):.4f}")

train(model, train_loader, optimizer, criterion_ae, criterion_age, device)

# Load dataset

test_path = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/test'
test_ann = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/submission.csv'
test_dataset = AgeDataset(test_path, test_ann, transform=transform)

# Define data loader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


# Evaluate the model
model.eval()
predictions = []
for images,_ in tqdm(test_loader):
    images = images.to(device)
    outputs = model(images)
    predictions.extend(outputs.flatten().detach().cpu().numpy())

# Add predictions to submission DataFrame
submit = pd.read_csv('/kaggle/input/smai-24-age-prediction/content/faces_dataset/submission.csv')
submit['age'] = predictions

# Save submission DataFrame with predictions
submit.to_csv('baseline.csv', index=False)