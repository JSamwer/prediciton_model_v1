import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset

class CryptoDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, target_col_idx):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col_idx = target_col_idx
        self.num_samples = len(data) - seq_len - pred_len + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Extract sequences
        x = self.data[idx:idx + self.seq_len]  # Input sequence [seq_len, num_features]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]  # Prediction sequence [pred_len, num_features]

        # Ensure no NaN values in inputs
        x_enc = np.nan_to_num(x[:, 1:], nan=0.0).astype(np.float32)  # Exclude timestamp column

        # Extract target variable dynamically
        y_target = np.nan_to_num(y[:, self.target_col_idx], nan=0.0).astype(np.float32)

        # Generate time features for encoder and decoder
        x_mark_enc = self.generate_time_features(x[:, 0])  # Timestamp column for encoder
        x_mark_dec = self.generate_time_features(y[:, 0])  # Timestamp column for decoder

        # Decoder input (e.g., zeros or placeholders)
        x_dec = np.zeros((self.pred_len, x_enc.shape[1]), dtype=np.float32)

        return (
            torch.tensor(x_enc, dtype=torch.float32),
            torch.tensor(x_mark_enc, dtype=torch.float32),
            torch.tensor(x_dec, dtype=torch.float32),
            torch.tensor(x_mark_dec, dtype=torch.float32),
            torch.tensor(y_target[:, None], dtype=torch.float32)  # Add extra dimension
        )

    def generate_time_features(self, timestamps):
        """
        Generates time features for TimesNet from timestamps.
        """
        # Convert timestamps to datetime
        if np.issubdtype(timestamps.dtype, np.number):
            timestamps = pd.to_datetime(timestamps, unit='ms', errors='coerce')
        else:
            timestamps = pd.to_datetime(timestamps, errors='coerce')

        # Handle invalid timestamps
        if timestamps.isnull().any():
            raise ValueError("Found invalid or NaT timestamps in the input. Check your data.")

        # Convert to pandas Series to use .dt accessor
        timestamps = pd.Series(timestamps)

        time_features = np.stack([
            timestamps.dt.month.values - 1,  # Month (0-11)
            timestamps.dt.day.values - 1,    # Day (0-based for consistency)
            timestamps.dt.weekday.values     # Weekday (0=Monday to 6=Sunday)
        ], axis=1).astype(np.float32)

        return time_features

def process_csv_files(directory, seq_len, pred_len, batch_size, train_split=0.8):
    combined_train_data = []
    combined_test_data = []

    for file in os.listdir(directory):

        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            print(f"Processing {file_path}...")

            # Read CSV file
            df = pd.read_csv(file_path)

            # Calculate percentage change and normalize
            df['price_change_percentage'] = df['close'].pct_change().fillna(0) 
            pct_mean = df['price_change_percentage'].mean
            pct_sdt = df['price_change_percentage'].std

            # Find the target column index dynamically
            target_col_idx = df.columns.get_loc('price_change_percentage')

            # Drop NaN rows and convert to numpy array
            df = df.dropna()
            data = df.values

            # Create dataset
            dataset = CryptoDataset(data, seq_len, pred_len, target_col_idx)

            # Train/test split
            train_size = int(train_split * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

            combined_train_data.append(train_dataset)
            combined_test_data.append(test_dataset)

    # Combine datasets
    combined_train_dataset = ConcatDataset(combined_train_data)
    combined_test_dataset = ConcatDataset(combined_test_data)
    print(len(combined_train_dataset))
    print(len(combined_test_dataset))

    # Create DataLoaders
    train_loader = DataLoader(
        combined_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=1, pin_memory= True, persistent_workers= True
    )
    test_loader = DataLoader(
        combined_test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=12,pin_memory= True, persistent_workers= True
    )

    return train_loader, test_loader









































# this is for later and woulöd only come into play when feeding into the recurrent nerual network.
def __getitem__(self, idx):
    # Input sequence
    x_enc = self.features[idx:idx + self.seq_len]
    # Target sequence
    y = self.targets[idx + self.seq_len: idx + self.seq_len + self.pred_len]
    y = y.reshape(-1, 1)

    # Time features
    x_mark_enc = self.time_features[idx:idx + self.seq_len]
    x_mark_dec = self.time_features[idx + self.seq_len: idx + self.seq_len + self.pred_len]

    # For forecasting, decoder input can be zeros or previous targets
    x_dec = np.zeros_like(x_enc[-self.pred_len:])

    # Convert to torch tensors and add batch dimension
    x_enc = torch.tensor(x_enc, dtype=torch.float32)
    x_mark_enc = torch.tensor(x_mark_enc, dtype=torch.float32)
    x_dec = torch.tensor(x_dec, dtype=torch.float32)
    x_mark_dec = torch.tensor(x_mark_dec, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return x_enc, x_mark_enc, x_dec, x_mark_dec, y

def calculate_RSI(self, prices, period=14):
    delta = prices.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    gain = up.rolling(window=period, min_periods=1).mean()
    loss = down.rolling(window=period, min_periods=1).mean()
    RS = gain / loss
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return RSI

def calculate_Bollinger_Bands(self, prices, period=20, k=2):
    MA = prices.rolling(window=period, min_periods=1).mean()
    STD = prices.rolling(window=period, min_periods=1).std()
    upper_band = MA + k * STD
    lower_band = MA - k * STD
    return upper_band, MA, lower_band

