import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def train(model, train_loader, val_loader, epochs, learning_rate, weight_decay, name, device, min_rating, max_rating, lambda_cl=0.1, temperature=0.2):
    print(f"\nTraining Model: {name}")

    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    best_val_loss = float('inf')
    scaling_factor = max_rating - min_rating

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for users, pos_items, ratings in train_loader:
            users = users.to(device)
            pos_items = pos_items.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()

            predictions = model(users, pos_items)
            rating_loss = mse_criterion(predictions, ratings)

            cl_loss = (
                model.contrastive_loss(users, pos_items)
                if hasattr(model, "contrastive_loss")
                else 0.0
            )
            
            loss = rating_loss + (
                model.lambda_cl * cl_loss
                if hasattr(model, "lambda_cl")
                else 0.0
            )

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for users, items, ratings in val_loader:
                users = users.to(device)
                items = items.to(device)
                ratings = ratings.to(device)

                preds = model(users, items)

                loss = mse_criterion(preds, ratings)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        val_rmse = np.sqrt(val_loss) * scaling_factor

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val RMSE: {val_rmse:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), name + ".pth")

    model.load_state_dict(torch.load(name + ".pth", map_location=device))
    print(f"Best Val RMSE: {np.sqrt(best_val_loss) * scaling_factor:.4f}")
    return model