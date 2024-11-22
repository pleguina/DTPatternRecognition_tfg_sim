import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from newGeo.Digis import * 
from newGeo.dtGeometry import *

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def extract_features(df_combined):
    """
    Extracts numerical features from digis and segments for each event.
    
    Args:
        df_combined (pd.DataFrame): Combined DataFrame containing digis and segments.
        
    Returns:
        X (np.ndarray): Input features for the NN.
        y (np.ndarray): Output targets for the NN.
    """
    feature_list = []
    target_list = []
    
    for idx, row in df_combined.iterrows():
        # --- Digi Features ---
        digis = row['digi_superLayer']
        layers = row['digi_layer']
        wires = row['digi_wire']
        times = row['digi_time']
        
        # Example aggregate features for digis
        if len(digis) > 0:
            digi_count = len(digis)
            digi_mean_wire = np.mean(wires)
            digi_std_wire = np.std(wires)
            digi_mean_time = np.mean(times)
            digi_std_time = np.std(times)
        else:
            digi_count = 0
            digi_mean_wire = 0
            digi_std_wire = 0
            digi_mean_time = 0
            digi_std_time = 0
        
        # --- Segment Features ---
        seg_hasPhi = row['seg_hasPhi']
        seg_hasZed = row['seg_hasZed']
        seg_posLoc_x = row['seg_posLoc_x']
        seg_posLoc_y = row['seg_posLoc_y']
        seg_posLoc_z = row['seg_posLoc_z']
        
        if len(seg_hasPhi) > 0:
            seg_count = len(seg_hasPhi)
            seg_hasPhi_mean = np.mean(seg_hasPhi)
            seg_hasZed_mean = np.mean(seg_hasZed)
            seg_mean_pos_x = np.mean(seg_posLoc_x)
            seg_mean_pos_y = np.mean(seg_posLoc_y)
            seg_mean_pos_z = np.mean(seg_posLoc_z)
        else:
            seg_count = 0
            seg_hasPhi_mean = 0
            seg_hasZed_mean = 0
            seg_mean_pos_x = 0
            seg_mean_pos_y = 0
            seg_mean_pos_z = 0
        
        # Combine features into a single list
        features = [
            digi_count,
            digi_mean_wire,
            digi_std_wire,
            digi_mean_time,
            digi_std_time,
            seg_count,
            seg_hasPhi_mean,
            seg_hasZed_mean,
            seg_mean_pos_x,
            seg_mean_pos_y,
            seg_mean_pos_z
        ]
        
        feature_list.append(features)
        
        # --- Define Target ---
        # For simplicity, let's predict the number of segments
        target = seg_count
        target_list.append(target)
    
    X = np.array(feature_list)
    y = np.array(target_list)
    
    return X, y

def aggregate_chamber_features(df_combined, df_geometry):
    """
    Aggregate digi features per chamber using accurate (x, z) coordinates.

    Args:
        df_combined (pd.DataFrame): Combined DataFrame containing digis and segments.
        df_geometry (pd.DataFrame): DataFrame containing chamber geometry data.

    Returns:
        df_features (pd.DataFrame): Aggregated features per chamber.
        df_targets (pd.Series): Number of segments per chamber.
    """
    chamber_ids = ['event_number', 'wheel', 'sector', 'station']
    grouped = df_combined.groupby(chamber_ids)
    
    features = []
    targets = []
    
    for name, group in grouped:
        event_number, wheel, sector, station = name
        
        # Retrieve the corresponding chamber geometry
        try:
            rawId = get_rawId(wheel=wheel, station=station, sector=sector)
        except ValueError as ve:
            logging.error(f"Error computing rawId for chamber ({wheel}, {sector}, {station}): {ve}")
            continue
        
        chamber_df = get_chamber_data(df_geometry, rawId)
        if chamber_df is None or chamber_df.empty:
            logging.error(f"No geometry data found for Chamber rawId {rawId}. Skipping chamber.")
            continue
        chamber = create_chamber_object(chamber_df)
        
        n_digis = group['digi_superLayer'].apply(len).sum()
        
        if n_digis > 0:
            # Explode the list columns
            digi_superLayers = group['digi_superLayer'].explode().astype(int)
            digi_layers = group['digi_layer'].explode().astype(int)
            digi_wires = group['digi_wire'].explode().astype(int)  # Assuming wire numbers are integers
            digi_times = group['digi_time'].explode().astype(float)
            
            # Initialize lists to store coordinates
            x_coords = []
            z_coords = []
            
            # Iterate through digis and convert to coordinates
            for idx in digi_superLayers.index:
                sl = digi_superLayers.loc[idx]
                layer = digi_layers.loc[idx]
                wire = digi_wires.loc[idx]
                
                x, z = chamber.convert_wire_to_xy(sl, layer, wire)
                if x is not None and z is not None:
                    x_coords.append(x)
                    z_coords.append(z)
            
            # Convert lists to numpy arrays for aggregation
            x_coords = np.array(x_coords)
            z_coords = np.array(z_coords)
            
            # Handle cases where conversion failed for all digis
            if len(x_coords) == 0:
                wire_mean = wire_std = wire_min = wire_max = 0
                x_mean = x_std = x_min = x_max = 0
                z_mean = z_std = z_min = z_max = 0
            else:
                # Aggregate features
                wire_mean = digi_wires.mean()
                wire_std = digi_wires.std()
                wire_min = digi_wires.min()
                wire_max = digi_wires.max()
                
                x_mean = x_coords.mean()
                x_std = x_coords.std()
                x_min = x_coords.min()
                x_max = x_coords.max()
                
                z_mean = z_coords.mean()
                z_std = z_coords.std()
                z_min = z_coords.min()
                z_max = z_coords.max()
        else:
            # If no digis, fill with zeros
            wire_mean = wire_std = wire_min = wire_max = 0
            x_mean = x_std = x_min = x_max = 0
            z_mean = z_std = z_min = z_max = 0
        
        # Append aggregated features
        features.append([
            wire_mean, wire_std, wire_min, wire_max,
            x_mean, x_std, x_min, x_max,
            z_mean, z_std, z_min, z_max,
            n_digis
        ])
        
        # Target: number of segments
        n_segments = group['n_segments'].sum()
        targets.append(n_segments)
    
    # Define feature columns
    feature_columns = [
        'wire_mean', 'wire_std', 'wire_min', 'wire_max',
        'x_mean', 'x_std', 'x_min', 'x_max',
        'z_mean', 'z_std', 'z_min', 'z_max',
        'n_digis'
    ]
    
    df_features = pd.DataFrame(features, columns=feature_columns)
    df_targets = pd.Series(targets, name='n_segments')
    
    return df_features, df_targets

# Define Dataset class
class NNDataset(Dataset):
    def __init__(self, features, targets):
        """
        Args:
            features (np.ndarray): Feature matrix.
            targets (pd.Series or list): Target vector.
        """
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets.values, dtype=torch.float32)  # Assuming regression
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# Define Neural Network Model
class SegmentsNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64]):
        super(SegmentsNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.output = nn.Linear(hidden_sizes[1], 1)  # Output: number of segments
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.output(x)
        return x.squeeze()  # Return as 1D tensor

# Define Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=10):
    """
    Train the neural network model with early stopping.
    
    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer.
        num_epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
    
    Returns:
        model: Trained model.
        history: Dictionary containing training and validation loss history.
    """
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = np.inf
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break
    
    # Load best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, history

# Define Evaluation Function
def evaluate_model(model, test_loader, criterion):
    """
    Evaluate the model on the test set.
    
    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for test data.
        criterion: Loss function.
    
    Returns:
        predictions (list): Predicted number of segments.
        actuals (list): Actual number of segments.
    """
    model.eval()
    test_losses = []
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_losses.append(loss.item())
            predictions.extend(outputs.numpy())
            actuals.extend(y_batch.numpy())
    
    avg_test_loss = np.mean(test_losses)
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    print(f"\nTest MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R² Score: {r2:.4f}")
    
    return predictions, actuals

# Define Plotting Functions
def plot_loss(history):
    """
    Plot training and validation loss over epochs.
    
    Args:
        history (dict): Dictionary containing 'train_loss' and 'val_loss' lists.
    """
    plt.figure(figsize=(10,6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(predictions, actuals, num_samples=100):
    """
    Scatter plot of actual vs predicted number of segments.
    
    Args:
        predictions (list): Predicted values.
        actuals (list): Actual values.
        num_samples (int): Number of samples to plot.
    """
    plt.figure(figsize=(10,6))
    plt.scatter(actuals[:num_samples], predictions[:num_samples], alpha=0.6, edgecolors='w', s=100)
    plt.plot([min(actuals[:num_samples]), max(actuals[:num_samples])],
             [min(actuals[:num_samples]), max(actuals[:num_samples])],
             'r--', label='Ideal')
    plt.xlabel('Actual Number of Segments')
    plt.ylabel('Predicted Number of Segments')
    plt.title('Actual vs Predicted Number of Segments')
    plt.legend()
    plt.grid(True)
    plt.show()

# Define Prediction Function
def predict_segments(model, scaler, new_event_features):
    """
    Predict the number of segments for a new chamber.
    
    Args:
        model (nn.Module): Trained model.
        scaler (StandardScaler): Scaler used for training data.
        new_event_features (list or np.ndarray): Feature list for the new chamber.
    
    Returns:
        float: Predicted number of segments.
    """
    model.eval()
    with torch.no_grad():
        # Convert to numpy array and reshape
        features = np.array(new_event_features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        
        # Predict
        prediction = model(features_tensor)
        return prediction.item()

