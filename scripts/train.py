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
            # print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    print("Training complete.")
    return model

# if __name__ == '__main__':
#     input_size = len(features)
#     hidden_layer_size = 128
#     output_size = len(target)

#     model = Predictor(input_size, hidden_layer_size, output_size)
#     loss_fn = RMSLELoss()
#     # loss_fn = nn.L1Loss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    
#     train