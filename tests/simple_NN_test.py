import sys
import os

# Add the base directory (one level up from tests) to the system path
# This allows the test to access the modules in the base directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from newGeo.dtGeometry import *
from newGeo.Digis import *
from newGeo.nn_basic_lib import *  # Basic library for the neural network design.
from plotter.cmsDraw import *

""" Overview

    Geometry Parsing
        Parsing DTGeometry XML: Utilize your provided classes and functions to parse the geometry XML and create a structured DataFrame (df_geometry).

    Data Preparation
        Feature Extraction: Convert digis and segments into numerical features, including mapping wire numbers to physical xx and yy positions using df_geometry.
        Normalization: Scale the features for better neural network performance.
        Train-Test Split: Divide the data into training and testing sets.
        Dataset and DataLoader: Create PyTorch datasets and loaders for efficient data handling.

    Model Building
        Define a feedforward neural network architecture using PyTorch's nn.Module.

    Training
        Implement learning curves by training the model on increasing subsets of data.

    Evaluation
        Assess the model's performance on the test data.
        Visualize training progress and predictions.

"""

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# **************************************************************************************************************************************************************************************
# Loading the Data
# **************************************************************************************************************************************************************************************

geometry_xml_path = "newGeo/DTGeometry.xml"
root_file_path = 'dtTuples/DTDPGNtuple_12_4_2_Phase2Concentrator_Simulation_89.root'
tree_name = 'dtNtupleProducer/DTTREE'

# Parse DT Geometry XML
print("\nParsing DT Geometry XML...")
df_geometry = parse_dtgeometry_xml(geometry_xml_path)
if df_geometry.empty:
    logging.error("Geometry DataFrame is empty. Check the XML file.")
    sys.exit(1)

# Generate Combined DataFrame that holds all the digis and segments of all the events
print("\nGenerating Combined DataFrame...")
df_combined = generate_combined_dataframe(root_file_path, tree_name)
if df_combined is None or df_combined.empty:
    logging.error("Combined DataFrame is empty or failed to generate.")
    sys.exit(1)

# The objective is that for an input of digis from a certain chamber, we want to predict the number of segments that will be produced in that chamber.
# The segments of a chamber are the result of the digis of that chamber. The number of segments is the target variable that we want to predict.
# So basically, we have a database with all the digis and segments of all the events. Each of these chambers receives digis and produces segments.
# The algorithm has to be able to predict the number of segments that will be produced in a specific chamber, for its specific digis.

# **************************************************************************************************************************************************************************************
# Plot some random events with digis and segments
# **************************************************************************************************************************************************************************************

#Then we print random events with digis and segments to have examples to work with
# You can add a random seed to the function to get the same events every time (third argument)
selected_events = print_random_events_with_counts(df_combined, 1, 6)

#Finally we plot a specific event with the plot_specific_event function

for event in selected_events:
        event_number = event['event_number']
        wheel = event['wheel']
        sector = event['sector']
        station = event['station']

        plot_specific_event(wheel=wheel, sector=sector, station=station, event_number=event_number,
                            df_combined=df_combined, df_geometry=df_geometry)
        
        plt.show()

# Also you can print the muon chamber and sector being processed
print(f"Processing Wheel: {wheel}, Station: {station}, Sector: {sector}")

draw_cms_muon_chambers(wheel, sector, station)

# **************************************************************************************************************************************************************************************
# Feature Extraction
# **************************************************************************************************************************************************************************************

# Expand the df_combined to have one row per digi
def expand_digis(df_combined):
    records = []
    for idx, row in df_combined.iterrows():
        event_number = row['event_number']
        wheel = row['wheel']
        sector = row['sector']
        station = row['station']
        if isinstance(row['digi_superLayer'], list):
            for i in range(len(row['digi_superLayer'])):
                records.append({
                    'event_number': event_number,
                    'wheel': wheel,
                    'sector': sector,
                    'station': station,
                    'superLayer': row['digi_superLayer'][i],
                    'layer': row['digi_layer'][i],
                    'wire': row['digi_wire'][i],
                    'time': row['digi_time'][i]
                })
    df_digis_expanded = pd.DataFrame(records)
    return df_digis_expanded

df_digis_expanded = expand_digis(df_combined)

# Check the expanded digis DataFrame
print("\nExpanded Digis DataFrame:")
print(df_digis_expanded.head())

# Group digis by event_number, wheel, sector, station
grouped_digis = df_digis_expanded.groupby(['event_number', 'wheel', 'sector', 'station'])

# Initialize list to collect features
features_list = []

for group_name, group_data in grouped_digis:
    event_number, wheel, sector, station = group_name
    num_digis = len(group_data)
    num_digis_SL1 = len(group_data[group_data['superLayer'] == 1])
    num_digis_SL2 = len(group_data[group_data['superLayer'] == 2])
    num_digis_SL3 = len(group_data[group_data['superLayer'] == 3])
    
    # Digi times statistics
    digi_times = group_data['time']
    time_mean = digi_times.mean()
    time_std = digi_times.std()
    time_min = digi_times.min()
    time_max = digi_times.max()
    
    # Wire numbers statistics
    wire_numbers = group_data['wire']
    wire_mean = wire_numbers.mean()
    wire_std = wire_numbers.std()
    wire_min = wire_numbers.min()
    wire_max = wire_numbers.max()
    
    # Create feature dict
    features = {
        'event_number': event_number,
        'wheel': wheel,
        'sector': sector,
        'station': station,
        'num_digis': num_digis,
        'num_digis_SL1': num_digis_SL1,
        'num_digis_SL2': num_digis_SL2,
        'num_digis_SL3': num_digis_SL3,
        'time_mean': time_mean,
        'time_std': time_std,
        'time_min': time_min,
        'time_max': time_max,
        'wire_mean': wire_mean,
        'wire_std': wire_std,
        'wire_min': wire_min,
        'wire_max': wire_max,
        # Additional features can be added here
    }
    
    features_list.append(features)

df_features = pd.DataFrame(features_list)
# Check the features DataFrame
print("\nFeatures DataFrame:")
print(df_features.head())

# Create a mapping from (chamber_rawId, superLayerNumber, layerNumber, wire_number) to x_position
def create_wire_position_mapping(df_geometry):
    mapping = {}
    for idx, row in df_geometry.iterrows():
        chamber_rawId = row['Chamber_rawId']
        superLayerNumber = row['SuperLayerNumber']
        layerNumber = row['LayerNumber']
        channels_total = row['Channels_total']
        wireFirst = row['WirePositions_FirstWire']
        wireLast = row['WirePositions_LastWire']
        # Create Wire object to get positions
        wire_obj = Wire(wireFirst, wireLast, channels_total, row['Layer_Local_z'])
        # Map wire numbers to positions
        for wire_idx, x_pos in enumerate(wire_obj.positions):
            wire_number = wire_idx + 1  # Assuming wire numbers start from 1
            key = (chamber_rawId, superLayerNumber, layerNumber, wire_number)
            mapping[key] = x_pos
    return mapping

wire_position_mapping = create_wire_position_mapping(df_geometry)
# Check the wire position mapping
print("\nWire Position Mapping:")
for key, value in list(wire_position_mapping.items())[:5]:
    print(f"{key}: {value}")

def add_chamber_rawId_to_digis(df_digis_expanded):
    chamber_rawIds = []
    for idx, row in df_digis_expanded.iterrows():
        wheel = int(row['wheel'])
        station = int(row['station'])
        sector = int(row['sector'])
        try:
            rawId = get_rawId(wheel, station, sector)
        except ValueError as e:
            logging.error(f"Error computing rawId for digi at index {idx}: {e}")
            rawId = None
        chamber_rawIds.append(rawId)
    df_digis_expanded['chamber_rawId'] = chamber_rawIds
    return df_digis_expanded

df_digis_expanded = add_chamber_rawId_to_digis(df_digis_expanded)

# Check the updated digis DataFrame  
print("\nUpdated Digis DataFrame:")
print(df_digis_expanded.head())
# Check if 'chamber_rawId' is in columns
print("Columns in df_digis_expanded:", df_digis_expanded.columns)

def map_digi_wires_to_positions(df_digis_expanded, wire_position_mapping):
    x_positions = []
    for idx, row in df_digis_expanded.iterrows():
        key = (row['chamber_rawId'], row['superLayer'], row['layer'], row['wire'])
        x_pos = wire_position_mapping.get(key, None)
        x_positions.append(x_pos)
    df_digis_expanded['x_position'] = x_positions
    return df_digis_expanded

df_digis_expanded = map_digi_wires_to_positions(df_digis_expanded, wire_position_mapping)
# Check the updated digis DataFrame
print("\nUpdated Digis DataFrame with Wire Positions:")
print(df_digis_expanded.head())

# **************************************************************************************************************************************************************************************
# Target Variable
# **************************************************************************************************************************************************************************************

# The target variable is the number of segments produced by each chamber. We need to extract this information from the segments data.
# The segments data is already part of the combined DataFrame, so we can use that to calculate the number of segments per chamber.

# First, expand the segments data
def expand_segments(df_combined):
    records = []
    for idx, row in df_combined.iterrows():
        event_number = row['event_number']
        wheel = row['wheel']
        sector = row['sector']
        station = row['station']
        if isinstance(row['seg_hasPhi'], list):
            for i in range(len(row['seg_hasPhi'])):
                records.append({
                    'event_number': event_number,
                    'wheel': wheel,
                    'sector': sector,
                    'station': station,
                    # Include other segment fields as needed
                })
    df_segments_expanded = pd.DataFrame(records)
    return df_segments_expanded

df_segments_expanded = expand_segments(df_combined)
# Check the expanded segments DataFrame
print("\nExpanded Segments DataFrame:")
print(df_segments_expanded.head())

# Group segments by event_number, wheel, sector, station
grouped_segments = df_segments_expanded.groupby(['event_number', 'wheel', 'sector', 'station'])
segments_count = grouped_segments.size().reset_index(name='num_segments')
# Check the segments count DataFrame
print("\nSegments Count DataFrame:")
print(segments_count.head())

# Merge the segments count with the features DataFrame
df_dataset = pd.merge(df_features, segments_count, on=['event_number', 'wheel', 'sector', 'station'], how='left')
# Fill NaN in num_segments with 0 (if no segments were found for that chamber)
df_dataset['num_segments'] = df_dataset['num_segments'].fillna(0).astype(int)

df_dataset = df_dataset.fillna(0)

print("\nFinal Dataset:")
print(df_dataset.head())

# **************************************************************************************************************************************************************************************
# Data Preparation
# **************************************************************************************************************************************************************************************

# Normalize the features
# This is important for neural networks to perform well as it helps in faster convergence and better generalization.

feature_columns = [col for col in df_dataset.columns if col not in ['event_number', 'wheel', 'sector', 'station', 'num_segments']]

scaler = StandardScaler()
df_dataset[feature_columns] = scaler.fit_transform(df_dataset[feature_columns])

X = df_dataset[feature_columns]
y = df_dataset['num_segments']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to tensors
# We need tensors to train the neural network using PyTorch. Tensors are similar to NumPy arrays but can be used on GPUs for faster computation.

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# **************************************************************************************************************************************************************************************
# Model Building
# **************************************************************************************************************************************************************************************

# Define a simple feedforward neural network model using PyTorch's nn.Module class.
# The model will take the input features and output the number of segments produced by the chamber.
# You can experiment with different architectures, activation functions, and hyperparameters to improve the model's performance.

class SegmentPredictor(nn.Module):
    def __init__(self, input_size):
        super(SegmentPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1)  # Output layer for regression
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

input_size = len(feature_columns)
# No need to instantiate the model here since we'll create a new one for each subset

# **************************************************************************************************************************************************************************************
# Implementing Learning Curves
# **************************************************************************************************************************************************************************************

print("\nImplementing Learning Curves...")

# Learning curves can help in understanding how the model performs with varying amounts of training data.
# We will train the model on increasing subsets of the training data and evaluate it on the validation set to observe the learning progress.
# How to interpret the learning curves:
# - If the training loss is much lower than the validation loss, the model may be overfitting.
# - If the training and validation losses are both high, the model may be underfitting.
# - The goal is to have both training and validation losses converge to a low value.


# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the percentages of data to use
train_sizes = [0.1, 0.3, 0.5, 0.7, 1.0]
validation_losses = []
training_losses = []
num_examples = []

batch_size = 64  # Define batch size

for size in train_sizes:
    print(f"\nTraining model with {int(size * 100)}% of the training data.")
    # Determine the number of samples
    num_samples = int(len(X_train_tensor) * size)
    num_examples.append(num_samples)
    
    # Create a subset of the training data
    indices = np.random.choice(len(X_train_tensor), num_samples, replace=False)
    X_train_subset = X_train_tensor[indices]
    y_train_subset = y_train_tensor[indices]
    
    # Create DataLoader for the subset
    train_dataset_subset = TensorDataset(X_train_subset, y_train_subset)
    train_loader_subset = DataLoader(train_dataset_subset, batch_size=batch_size, shuffle=True)
    
    # Initialize a new model for each subset
    model_subset = SegmentPredictor(input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_subset.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 20  # You can adjust the number of epochs
    for epoch in range(num_epochs):
        model_subset.train()
        running_loss = 0.0
        for features, targets in train_loader_subset:
            optimizer.zero_grad()
            outputs = model_subset(features)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)
        epoch_loss = running_loss / num_samples
        # Optionally print epoch loss
        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    # Record training loss (last epoch)
    training_losses.append(epoch_loss)
    
    # Evaluate on validation set
    model_subset.eval()
    with torch.no_grad():
        outputs = model_subset(X_test_tensor)
        val_loss = criterion(outputs.squeeze(), y_test_tensor)
        val_loss_value = val_loss.item()
        validation_losses.append(val_loss_value)
        print(f'Validation Loss: {val_loss_value:.4f}')

# Plot the learning curve
plt.figure(figsize=(8, 6))
plt.plot(num_examples, training_losses, marker='o', label='Training Loss')
plt.plot(num_examples, validation_losses, marker='o', label='Validation Loss')
plt.xlabel('Number of Training Examples')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()

# **************************************************************************************************************************************************************************************
# Training the Final Model on Full Data
# **************************************************************************************************************************************************************************************

# Based on the learning curves, we can train the final model on the full training data and evaluate it on the test set.
# The final model will be used to predict the number of segments produced by each chamber based on the input features.


print("\nTraining the final model on the full training data...")

# Create datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize the model
model = SegmentPredictor(input_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#The model is trained on the full training data and evaluated on the test data. The test MSE and R2 score are metrics to evaluate the model's performance.
#The scatter plot of actual vs predicted number of segments shows how well the model is able to predict the number of segments for each chamber.
# The number of epochs represents the number of times the model goes through the entire training data during training.
# Modify the number of epochs can help in improving the model's performance.
#A nice number of epochs to start with is 50. You can experiment with different values to see how the model's performance changes.

num_epochs = 50
train_losses = []
test_losses = []
train_mses = []
test_mses = []

for epoch in range(num_epochs):
    # Training Phase
    # Here we train the model on the full training data and calculate the training loss.
    # We also calculate the training MSE to evaluate the model's performance on the training data.
    
    model.train()
    running_loss = 0.0
    for features, targets in train_loader:
        optimizer.zero_grad() # Zero the gradients, to avoid accumulation (which would lead to incorrect gradients)
        outputs = model(features)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    
    # Calculate training MSE 
    # We calculate the training MSE to evaluate the model's performance on the training data.
    # This works as a sanity check to ensure the model is learning from the data.
    #How to interpret the training MSE:
    # - A low training MSE indicates that the model is able to predict the number of segments accurately on the training data.
    # - A high training MSE may indicate that the model is not learning the patterns in the data.
    
    with torch.no_grad():
        train_predictions = model(X_train_tensor)
        train_mse = mean_squared_error(y_train_tensor.numpy(), train_predictions.squeeze().numpy())
        train_mses.append(train_mse)
    
    # Validation Phase
    model.eval()
    
    # Calculate validation loss
    # Here we evaluate the model on the test data and calculate the validation loss.
    # The validation loss helps in understanding how well the model generalizes to unseen data.
    # A low validation loss indicates that the model is able to predict the number of segments accurately on the test data.
    # A high validation loss may indicate that the model is overfitting to the training data.
    
    test_loss = 0.0
    with torch.no_grad():
        for features, targets in test_loader:
            outputs = model(features)
            loss = criterion(outputs.squeeze(), targets)
            test_loss += loss.item() * features.size(0)
    epoch_test_loss = test_loss / len(test_loader.dataset)
    test_losses.append(epoch_test_loss)
    
    # Calculate validation MSE
    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_mse = mean_squared_error(y_test_tensor.numpy(), test_predictions.squeeze().numpy())
        test_mses.append(test_mse)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Validation Loss: {epoch_test_loss:.4f}')

# Loss Curves
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.show()

# MSE Curves
plt.figure(figsize=(10, 4))
plt.plot(train_mses, label='Training MSE')
plt.plot(test_mses, label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training and Validation MSE over Epochs')
plt.legend()
plt.show()

# **************************************************************************************************************************************************************************************
# Evaluation
# **************************************************************************************************************************************************************************************

# Since before we trained the model on the full training data and evaluated it on the test data, we can now evaluate the model's performance using the test data.
# We calculate the test MSE and R2 score to evaluate the model's performance on the test data.

model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
    for features, targets in test_loader:
        outputs = model(features)
        predictions.extend(outputs.squeeze().tolist())
        actuals.extend(targets.tolist())

mse = mean_squared_error(actuals, predictions)
r2 = r2_score(actuals, predictions)
print(f'Test MSE: {mse:.4f}, R2 Score: {r2:.4f}')

plt.figure(figsize=(8, 6))
plt.scatter(actuals, predictions, alpha=0.6)
plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
plt.xlabel('Actual Number of Segments')
plt.ylabel('Predicted Number of Segments')
plt.title('Predicted vs Actual Number of Segments')
plt.show()

#**************************************************************************************************************************************************************************************
# Results
#**************************************************************************************************************************************************************************************

# Test MSE: 6.9875, R2 Score: 0.6262
# The test MSE and R2 score are metrics to evaluate the model's performance.
# The values that we obtained indicate that the model is able to predict the number of segments with reasonable accuracy.
# But the plot of actual vs predicted number of segments shows that there is still room for improvement.
# The training and validation loss curves show that the model reduces the loss over epochs, indicating that it is learning from the data, 
# but there is a slight gap between the training and validation losses, suggesting that the model may be slightly overfitting.

# **************************************************************************************************************************************************************************************
# Summary
# **************************************************************************************************************************************************************************************

# The model was trained on the full training data and evaluated on the test data. The test MSE and R2 score are metrics to evaluate the model's performance.
# The scatter plot of actual vs predicted number of segments shows how well the model is able to predict the number of segments for each chamber.
# The learning curves show the training and validation loss over epochs, which can help in understanding the model's convergence and generalization.
# The MSE curves show the training and validation MSE over epochs, which provide a quantitative measure of the model's performance.
# The model can be further improved by tuning hyperparameters, adding more features, or using more complex architectures.

# **************************************************************************************************************************************************************************************

#Now you can try to add more features to the model, tune hyperparameters, or experiment with different architectures to improve the model's performance.
#You can modify the target varibale. For the moment, we are predicting the number of segments produced by a chamber. You can try to match the segments to the digis and predict the exact segments produced by a specific number of digis.
#For example, if 12 digis in one station produce 3 segments, which digis are responsible for the segments? This can be a more complex problem to solve.

# To load more data, check load_multiple_root_files() function in digis.py

# End of simple_NN_test.py