import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import model_v1
import data
import numpy as np
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="runs/your_experiment_name")

    # Set spawn method for multiprocessing compatibility
    mp.set_start_method('spawn', force=True)

    # Initialize dataset and dataloader
    directory = "crypto_data"
    seq_len = 16  # Input sequence length
    pred_len = 1  # Prediction sequence length
    batch_size = 20

    train_loader, test_loader = data.process_csv_files(directory, seq_len, pred_len, batch_size)
    num_batches = len(train_loader)
    print(f"Total number of batches: {num_batches}")

    # Configuration class to hold model parameters
    class Configs:
        def __init__(self):
            # Input features: open, high, low, close, RSI, Bollinger Bands, volume
            self.enc_in = 9  # Number of input channels
            self.seq_len = seq_len  # Length of input sequence
            self.label_len = 48  # For forecasting tasks (not used in this example)
            self.pred_len = pred_len  # Length of prediction sequence

            # Model parameters
            self.d_model = 32  # Model dimension (adjusted to reach ~2 million parameters)
            self.d_ff = 64  # Dimension of FeedForward network
            self.e_layers = 3  # Number of encoder layers
            self.top_k = 5  # Number of top periods to consider
            self.num_kernels = 5  # Number of kernels in Inception block

            self.task_name = 'short_term_forecast'  # Task type
            self.embed = 'fixed'  # Embedding type
            self.freq = 'd'  # Frequency for temporal embedding
            self.dropout = 0.1  # Dropout rate
            self.num_class = None  # For classification tasks
            self.c_out = 1  # Number of output channels (e.g., predicting the closing price)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model and move to device
    configs = Configs()
    model = model_v1.Model_v1(configs).to(device)

    # Count total number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    # Round total parameters to nearest million
    rounded_params_millions = int(total_params / 1e6 + 0.5)

    weights_dir = 'model_weights'
    weights_filename = f'{rounded_params_millions}M_version.pth'
    weights_path = os.path.join(weights_dir, weights_filename)

    # Check if weights file exists
    if os.path.isfile(weights_path):
        print(f"Weights file {weights_filename} found. Loading weights.")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(f"Weights file {weights_filename} not found. Training from scratch.")

    # Create weights directory if it doesn't exist
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    def percentage_change_loss(predicted, target):
        """
        Custom loss function for normalized percentage change prediction.

        Args:
            predicted: Model's predicted normalized percentage changes.
            target: Ground truth normalized percentage changes.

        Returns:
            Combined loss with a penalty for incorrect direction.
        """
        # Mean Squared Error (MSE) for magnitude accuracy
        mse_loss = F.mse_loss(predicted, target)

        # Penalize incorrect sign predictions
        predicted_sign = torch.sign(predicted)
        target_sign = torch.sign(target)
        sign_mismatch = (predicted_sign != target_sign).float()  # 1 if signs differ, else 0

        penalty_factor = 2.0  # Weight for penalizing sign mismatches
        sign_penalty = sign_mismatch.mean() * penalty_factor

        # Combine MSE and sign penalty
        total_loss = mse_loss + sign_penalty
        return total_loss

    # Define loss function and optimizer
    criterion = percentage_change_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # Adjust learning rate as needed

    # Initialize lists to store metrics
    train_losses = []
    train_sign_accuracies = []

    # Training loop
    num_epochs = 50  # Adjust as needed

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_sign_accuracy = 0.0
        num_samples = 0

        for batch_idx, (x_enc, x_mark_enc, x_dec, x_mark_dec, y) in enumerate(train_loader):
            optimizer.zero_grad()

            x_enc = x_enc.to(device)
            x_mark_enc = x_mark_enc.to(device)
            x_dec = x_dec.to(device)
            x_mark_dec = x_mark_dec.to(device)
            y = y.to(device)

            outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            # Compute directional accuracy
            predicted_sign = torch.sign(outputs)
            actual_sign = torch.sign(y)
            sign_accuracy = (predicted_sign == actual_sign).float().mean().item()

            # Update metrics
            batch_size = x_enc.size(0)
            train_loss += loss.item() * batch_size
            total_sign_accuracy += sign_accuracy * batch_size
            num_samples += batch_size

            # Log metrics to TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.add_scalar("Accuracy/train_batch", sign_accuracy, global_step)

        # Average metrics for the epoch
        avg_train_loss = train_loss / num_samples
        avg_sign_accuracy = total_sign_accuracy / num_samples
        train_losses.append(avg_train_loss)
        train_sign_accuracies.append(avg_sign_accuracy)

        # Log epoch metrics to TensorBoard
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/train_epoch", avg_sign_accuracy, epoch)

        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Training Loss: {avg_train_loss:.4f}, "
            f"Directional Accuracy: {avg_sign_accuracy:.4f}"
        )

        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")

    # After training, save the model weights
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to {weights_path}")

    # Close the TensorBoard writer
    writer.close()

    # Testing loop
    model.eval()
    test_loss = 0.0
    total_test_sign_accuracy = 0.0
    num_test_samples = 0

    with torch.no_grad():
        for x_enc, x_mark_enc, x_dec, x_mark_dec, y in test_loader:
            x_enc = x_enc.to(device)
            x_mark_enc = x_mark_enc.to(device)
            x_dec = x_dec.to(device)
            x_mark_dec = x_mark_dec.to(device)
            y = y.to(device)

            outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = criterion(outputs, y)

            # Compute directional accuracy
            predicted_sign = torch.sign(outputs)
            actual_sign = torch.sign(y)
            sign_accuracy = (predicted_sign == actual_sign).float().mean().item()

            # Update metrics
            batch_size = x_enc.size(0)
            test_loss += loss.item() * batch_size
            total_test_sign_accuracy += sign_accuracy * batch_size
            num_test_samples += batch_size

    # Average test loss and accuracy
    avg_test_loss = test_loss / num_test_samples
    avg_test_sign_accuracy = total_test_sign_accuracy / num_test_samples

    print(
        f"Validation Loss: {avg_test_loss:.4f}, "
        f"Validation Directional Accuracy: {avg_test_sign_accuracy:.4f}"
    )