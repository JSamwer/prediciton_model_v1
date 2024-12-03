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
    writer = SummaryWriter(log_dir="runs/your_experiment_name")

    # Set spawn method for macOS compatibility
    mp.set_start_method('spawn', force=True)

    # Initialize dataset and dataloader
    
    directory = "crypto_data"
    seq_len = 16  # Input sequence length
    pred_len = 1  # Prediction sequence length
    batch_size = 20


    train_loader, test_loader = data.process_csv_files(directory, seq_len, pred_len, batch_size)
    num_batches = len(train_loader)
    print(f"Total number of batches: {num_batches}")

    # #Now, you can use train_loader and test_loader in your training loop
    # for x_enc, x_mark_enc, x_dec, x_mark_dec, y in train_loader:
    #     print(f"x_enc shape: {x_enc.shape}")
    #     print(f"x_mark_enc shape: {x_mark_enc.shape}")
    #     print(f"x_dec shape: {x_dec.shape}")
    #     print(f"x_mark_dec shape: {x_mark_dec.shape}")
    #     print(f"y shape: {y.shape}")
    #     x_mark_enc = x_mark_enc.long()  
    #     x_mark_dec = x_mark_dec.long()  
    #     print(y[:50])
    #     break  # Remove this line to iterate over the entire DataLoader 


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
            self.e_layers = 3 # Number of encoder layers
            self.top_k = 5  # Number of top periods to consider
            self.num_kernels = 5 # Number of kernels in Inception block

            self.task_name = 'short_term_forecast'  # Task type
            self.embed = 'fixed'  # Embedding type
            self.freq = 'd'  # Frequency for temporal embedding
            self.dropout = 0.1  # Dropout rate
            self.num_class = None  # For classification tasks
            self.c_out = 1  # Number of output channels (e.g., predicting the closing price)



    # Check if MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because PyTorch was not built with MPS enabled.")
        else:
            print("MPS not available because your macOS version is below 12.3 or your device doesn't support MPS.")
    else:
        # Set the device to MPS
        mps_device = torch.device("mps")
        print("MPS backend enabled.")

    # Initialize the model and move to device
    configs = Configs()
    model = model_v1.Model_v1(configs).to(mps_device)

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
        model.load_state_dict(torch.load(weights_path, map_location=mps_device))
    else:
        print(f"Weights file {weights_filename} not found. Training from scratch.")

    # Create weights directory if it doesn't exist
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    import torch.nn.functional as F

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) # tweak for better optimization 




    # Initialize lists to store metrics and gradients
    train_losses = []
    test_losses = []
    train_sign_accuracies = []
    test_sign_accuracies = []
    gradients = []
    all_outputs = []
    all_targets = []
    train_loss = 0.0
    total_sign_accuracy = 0.0
    num_samples = 0

    # Training loop
    num_epochs = 50  # Adjust as needed
    # train and test loss record for analysis 
   

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (x_enc, x_mark_enc, x_dec, x_mark_dec, y) in enumerate(train_loader):
            optimizer.zero_grad()

            x_enc = x_enc.to(mps_device)
            x_mark_enc = x_mark_enc.to(mps_device)
            x_dec = x_dec.to(mps_device)
            x_mark_dec = x_mark_dec.to(mps_device)
            y = y.to(mps_device)

            
            outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            # regular forward pass
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients.append(param.grad.detach().cpu().numpy().flatten())

            
            all_outputs.append(outputs.cpu())
            all_targets.append(y.cpu())


            # Compute directional accuracy
            predicted_sign = torch.sign(outputs)
            actual_sign = torch.sign(y)
            sign_accuracy = (predicted_sign == actual_sign).float().mean().item()
            avg_sign_accuracy = 0
            # Average metrics for the epoch
            if num_samples != 0 : 
                train_loss /= num_samples
                avg_sign_accuracy = total_sign_accuracy / num_samples
            # Update metrics
            train_loss += loss.item() * x_enc.size(0)
            train_losses.append(loss.item() * x_enc.size(0))
            total_sign_accuracy += sign_accuracy * x_enc.size(0)
            train_sign_accuracies.append(avg_sign_accuracy)
            num_samples += x_enc.size(0)
            #log for tensorboard 
            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + batch_idx)

      

        # Store metrics
        train_losses.append(train_loss)
        train_sign_accuracies.append(avg_sign_accuracy)

        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Training Loss: {train_loss:.4f}, "
            f"Directional Accuracy: {avg_sign_accuracy:.4f}"
        )
        # Log other metrics at the end of an epoch
        writer.add_scalar("Accuracy/train", total_sign_accuracy, epoch)
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
        #save weights after each epoch so that you can always reload to last epoch in case 
        if (epoch + 1) % 10 == 0:  # Save every 5 epochs
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")
    # After training, save the model weights
    writer.close()
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to {weights_path}")


    # Testing loop
    # Optional: Evaluate on test set at each epoch
    model.eval()
    test_loss = 0.0
    total_test_sign_accuracy = 0.0
    num_test_samples = 0

    with torch.no_grad():
        for x_enc, x_mark_enc, x_dec, x_mark_dec, y in test_loader:
            x_enc = x_enc.to(mps_device)
            x_mark_enc = x_mark_enc.to(mps_device)
            x_dec = x_dec.to(mps_device)
            x_mark_dec = x_mark_dec.to(mps_device)
            y = y.to(mps_device)

            with torch.cuda.amp.autocast():
                outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                loss = criterion(outputs, y)

            test_loss += loss.item() * x_enc.size(0)

            # Compute directional accuracy
            predicted_sign = torch.sign(outputs)
            actual_sign = torch.sign(y)
            sign_accuracy = (predicted_sign == actual_sign).float().mean().item()

            total_test_sign_accuracy += sign_accuracy * x_enc.size(0)
            num_test_samples += x_enc.size(0)

    # Average test loss and accuracy
    # Average test loss and accuracy
    test_loss /= num_test_samples
    avg_test_sign_accuracy = total_test_sign_accuracy / num_test_samples

    test_losses.append(test_loss)
    test_sign_accuracies.append(avg_test_sign_accuracy)

    print(
        f"Validation Loss: {test_loss:.4f}, "
        f"Validation Directional Accuracy: {avg_test_sign_accuracy:.4f}"
    )

    