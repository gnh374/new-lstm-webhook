import torch
from torch import nn
from torch.utils.data import DataLoader

def train(model: nn.Module, loss_fn: nn.Module, optimizer: nn.Module, epochs: int, train_set: DataLoader, save_path: str) -> nn.Module:
    train_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_set:

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        # Average loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_set)
        train_losses.append(avg_train_loss)

        torch.save(model.state_dict(), save_path)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}")
          
    print("Training complete.")
    return model
