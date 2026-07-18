from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split


class BCDataset(Dataset):
    def __init__(self, X, Y, x_mean, x_std, y_mean, y_std):
        self.X = torch.from_numpy(((X - x_mean) / x_std).astype(np.float32))
        self.Y = torch.from_numpy(((Y - y_mean) / y_std).astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class BCMLPv2(nn.Module):
    def __init__(self, input_dim=59, output_dim=9, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, output_dim),
            # nn.Tanh(),  # bounded normalized action output
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def compute_stats(train_X, train_Y, eps=1e-6):
    x_mean = train_X.mean(axis=0)
    x_std = train_X.std(axis=0)
    x_std = np.maximum(x_std, eps)

    y_mean = train_Y.mean(axis=0)
    y_std = train_Y.std(axis=0)
    y_std = np.maximum(y_std, eps)

    return x_mean, x_std, y_mean, y_std


def denormalize_actions(y_norm, y_mean, y_std):
    return y_norm * y_std + y_mean


def evaluate(model, loader, criterion, device, y_mean, y_std):
    model.eval()

    total_loss_norm = 0.0
    total_count = 0

    preds_orig = []
    targets_orig = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            batch_size = x.shape[0]
            total_loss_norm += loss.item() * batch_size
            total_count += batch_size

            pred_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()

            pred_orig = denormalize_actions(pred_np, y_mean, y_std)
            y_orig = denormalize_actions(y_np, y_mean, y_std)

            preds_orig.append(pred_orig)
            targets_orig.append(y_orig)

    preds_orig = np.concatenate(preds_orig, axis=0)
    targets_orig = np.concatenate(targets_orig, axis=0)

    mse_orig = ((preds_orig - targets_orig) ** 2).mean()
    mae_orig = np.abs(preds_orig - targets_orig).mean()

    return total_loss_norm / total_count, mse_orig, mae_orig


def main():
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(output_dir / "bc_X.npy").astype(np.float32)
    Y = np.load(output_dir / "bc_Y.npy").astype(np.float32)

    print("Loaded data:")
    print("  X shape:", X.shape)
    print("  Y shape:", Y.shape)

    n_total = X.shape[0]
    indices = np.arange(n_total)

    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    n_train = int(0.9 * n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_val = X[val_idx]
    Y_val = Y[val_idx]

    x_mean, x_std, y_mean, y_std = compute_stats(X_train, Y_train)

    stats = {
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
    }
    with open(output_dir / "bc_v2_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    train_dataset = BCDataset(X_train, Y_train, x_mean, x_std, y_mean, y_std)
    val_dataset = BCDataset(X_val, Y_val, x_mean, x_std, y_mean, y_std)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = BCMLPv2(input_dim=X.shape[1], output_dim=Y.shape[1], hidden_dim=256).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 40
    best_val_mse_orig = float("inf")

    y_mean_np = y_mean.astype(np.float32)
    y_std_np = y_std.astype(np.float32)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            batch_size = x.shape[0]
            total_train_loss += loss.item() * batch_size
            total_train_count += batch_size

        train_loss_norm = total_train_loss / total_train_count
        val_loss_norm, val_mse_orig, val_mae_orig = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            y_mean=y_mean_np,
            y_std=y_std_np,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_norm_mse={train_loss_norm:.6f} | "
            f"val_norm_mse={val_loss_norm:.6f} | "
            f"val_orig_mse={val_mse_orig:.6f} | "
            f"val_orig_mae={val_mae_orig:.6f}"
        )

        if val_mse_orig < best_val_mse_orig:
            best_val_mse_orig = val_mse_orig
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "y_mean": y_mean,
                    "y_std": y_std,
                },
                output_dir / "bc_v2_model.pt",
            )

    print("\nTraining complete.")
    print(f"Best validation MSE in original action scale: {best_val_mse_orig:.6f}")
    print(f"Saved model to: {output_dir / 'bc_v2_model.pt'}")
    print(f"Saved stats to: {output_dir / 'bc_v2_stats.json'}")


if __name__ == "__main__":
    main()