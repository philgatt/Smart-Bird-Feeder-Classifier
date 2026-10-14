from bird_classifier.data.dataloader import BirdDataset, create_dataloader
from bird_classifier.models.cnn import SimpleCNN
from bird_classifier.data.preprocessing import get_data_split, get_mean_std
from torch import nn
import torch 
import numpy as np
from pathlib import Path


train_paths, train_labels = get_data_split(image_dir=Path("data/interim/deduplicated_images"),
                                           labels_path=Path("data/processed/training_labels.csv"), split="train")
test_paths, test_labels = get_data_split(image_dir=Path("data/interim/deduplicated_images"), 
                                        labels_path=Path("data/processed/training_labels.csv"), split="test")
# get std and mean for normalization
train_dataset = BirdDataset(train_paths, train_labels)
train_mean, train_std = get_mean_std(train_dataset)

# initialize dataset now with calculated mean and std
train_dataset = BirdDataset(train_paths, train_labels, train_std, train_mean)
test_dataset = BirdDataset(test_paths, test_labels, train_std, train_mean)

train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)
test_loader = create_dataloader(test_dataset, batch_size=32, shuffle=False)

def evaluate(model, dataloader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)

            logits = model(images)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * images.shape[0]

            predictions = logits.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_images += images.shape[0]

    average_loss = total_loss / total_images
    accuracy = correct_predictions / total_images

    return average_loss, accuracy

num_classes = len(np.unique(train_labels))
model = SimpleCNN(in_channels=3, num_classes=num_classes)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    
    total_loss = 0
    correct_predictions = 0
    total_images = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(images)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.shape[0]

        predictions = logits.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_images += images.shape[0]

        average_loss = total_loss / total_images
        accuracy = correct_predictions / total_images

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"loss: {average_loss:.4f} | "
            f"train accuracy: {accuracy:.2%}"
        )

    validation_loss, validation_accuracy = evaluate(
        model,
        test_loader,  # temporarily acting as validation data
        loss_fn,
        device,
    )

    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"train loss: {average_loss:.4f} | "
        f"train accuracy: {accuracy:.2%} | "
        f"validation loss: {validation_loss:.4f} | "
        f"validation accuracy: {validation_accuracy:.2%}"
    )

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "mean": train_mean.tolist(),
        "std": train_std.tolist(),
        "num_classes": num_classes,
    },
    "models/bird_cnn.pt",
)