#Digis.py

# This library contains functions to process digis and segments data from ROOT files.

import pandas as pd
import uproot
import random

# Define the filtering function

def load_root_file(root_file_path, tree_name):
    try:
        file = uproot.open(root_file_path)
        print("\nSuccessfully opened the ROOT file.")
        if tree_name in file:
            tree = file[tree_name]
            print(f"\nExpanding the '{tree_name}' structure:")
            return tree
        else:
            print(f"\nThe tree '{tree_name}' was not found in the ROOT file.")
            return None
    except FileNotFoundError:
        print(f"\nThe ROOT file '{root_file_path}' was not found.")
        return None

def load_multiple_root_files(root_folder_path, tree_name):
    """
    Load and concatenate data from multiple ROOT files in a folder.

    Parameters:
    - root_folder_path (str): Path to the folder containing the ROOT files.
    - tree_name (str): Name of the tree in each ROOT file to extract.

    Returns:
    - pd.DataFrame: Concatenated DataFrame containing data from all ROOT files.
    """
    all_dfs = []
    
    # Iterate over all files in the folder
    for root_file in os.listdir(root_folder_path):
        if root_file.endswith(".root"):  # Only process ROOT files
            root_file_path = os.path.join(root_folder_path, root_file)
            print(f"\nProcessing ROOT file: {root_file_path}")
            
            # Generate DataFrame from the current ROOT file
            df_combined = generate_combined_dataframe(root_file_path, tree_name)
            
            if df_combined is not None:
                all_dfs.append(df_combined)  # Append the DataFrame to the list
    
    if not all_dfs:
        print("No valid ROOT files were found or processed.")
        return None
    
    # Concatenate all the DataFrames
    final_df_combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nSuccessfully concatenated {len(all_dfs)} ROOT files into a single DataFrame.")
    
    return final_df_combined

def extract_data(tree, branches_to_extract):
    print("\nLoading branches from the ROOT file...")
    try:
        arrays = tree.arrays(branches_to_extract, library="np")
        print("Branches loaded successfully.")
        return arrays
    except KeyError as e:
        print(f"Error loading branches: {e}")
        return None

def build_events_dataframe(arrays, seg_numeric_branches):
    # Create a list to store each event's data
    data = []

    # Iterate over events using the length of `event_eventNumber`
    print("\nProcessing events to build the DataFrame...")
    for i, event_number in enumerate(arrays["event_eventNumber"]):
        # Extract the input data (digis) for the current event
        n_digis = arrays["digi_nDigis"][i]
        if n_digis > 0:
            digis = {
                "digi_wheel": arrays["digi_wheel"][i][:n_digis].tolist(),
                "digi_sector": arrays["digi_sector"][i][:n_digis].tolist(),
                "digi_station": arrays["digi_station"][i][:n_digis].tolist(),
                "digi_superLayer": arrays["digi_superLayer"][i][:n_digis].tolist(),
                "digi_layer": arrays["digi_layer"][i][:n_digis].tolist(),
                "digi_wire": arrays["digi_wire"][i][:n_digis].tolist(),
                "digi_time": arrays["digi_time"][i][:n_digis].tolist()
            }
        else:
            digis = {
                "digi_wheel": [],
                "digi_sector": [],
                "digi_station": [],
                "digi_superLayer": [],
                "digi_layer": [],
                "digi_wire": [],
                "digi_time": []
            }

        # Extract the output data (segments) for the current event
        n_segments = arrays["seg_nSegments"][i]
        if n_segments > 0:
            segment_data = {}
            for branch in seg_numeric_branches:
                if branch == "seg_nSegments":
                    continue
                segment_data[branch] = arrays[branch][i][:n_segments].tolist()
        else:
            segment_data = {branch: [] for branch in seg_numeric_branches if branch != "seg_nSegments"}

        # Store the data for the current event
        data.append({
            "event_number": event_number,
            "n_digis": n_digis,
            "digi_data": digis,
            "n_segments": n_segments,
            "segment_data": segment_data
        })

    # Convert the list into a pandas DataFrame
    df_events = pd.DataFrame(data)
    print("DataFrame with combined digi and segment data created successfully.")
    return df_events

def filter_digis_specific(df, event_number, wheel, sector, station, superlayers):
    """
    Filters digis for a specific event, wheel, sector, station, and superlayers.

    Args:
        df (pd.DataFrame): The long format DataFrame containing digis.
        event_number (int): The event number to filter by.
        wheel (int): The wheel number to filter by.
        sector (int): The sector number to filter by.
        station (int): The station number to filter by.
        superlayers (list): List of superlayer numbers to filter by.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    mask = (
        (df['event_number'] == event_number) &
        (df['digi_wheel'] == wheel) &
        (df['digi_sector'] == sector) &
        (df['digi_station'] == station) &
        (df['digi_superLayer'].isin(superlayers))
    )
    return df[mask]

# Define a validation function
def validate_digis(df):
    """
    Validates the integrity of digi data in the DataFrame.

    Args:
        df (pd.DataFrame): The long format DataFrame containing digis.

    Returns:
        bool: True if data is valid, False otherwise.
    """
    required_columns = ['event_number', 'digi_wheel', 'digi_sector',
                        'digi_station', 'digi_superLayer', 'digi_layer', 'digi_wire', 'digi_time']
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing required column: {col}")
            return False

    # Check for nulls
    if df[required_columns].isnull().any().any():
        print("DataFrame contains null values in required columns.")
        return False

    # Additional checks can be added here
    return True


def summarize_data_availability(df_digis_grouped, df_segments_grouped):
    print("\nData Availability Summary:")
    print("\nDigis Data:")
    if df_digis_grouped.empty:
        print("No digi data available.")
    else:
        print(df_digis_grouped[['event_number', 'digi_wheel', 'digi_sector', 'digi_station', 'digi_superLayer']].drop_duplicates())
    
    print("\nSegments Data:")
    if df_segments_grouped.empty:
        print("No segment data available.")
    else:
        print(df_segments_grouped[['event_number', 'seg_wheel', 'seg_sector', 'seg_station']].drop_duplicates())


def prepare_digi_data(df_events):
    digi_records = []
    for idx, row in df_events.iterrows():
        event_number = row['event_number']
        n_digis = row['n_digis']
        if n_digis == 0:
            continue  # Skip events with zero digis
        digis = row['digi_data']
        for i in range(n_digis):
            digi_record = {
                'event_number': event_number,
                'digi_wheel': digis['digi_wheel'][i],
                'digi_sector': digis['digi_sector'][i],
                'digi_station': digis['digi_station'][i],
                'digi_superLayer': digis['digi_superLayer'][i],
                'digi_layer': digis['digi_layer'][i],
                'digi_wire': digis['digi_wire'][i],
                'digi_time': digis['digi_time'][i],
            }
            digi_records.append(digi_record)
    df_digis_flat = pd.DataFrame(digi_records)
    return df_digis_flat

def prepare_segment_data(df_events):
    segment_records = []
    for idx, row in df_events.iterrows():
        event_number = row['event_number']
        n_segments = row['n_segments']
        if n_segments == 0:
            continue  # Skip events with zero segments
        segments = row['segment_data']
        for i in range(n_segments):
            segment_record = {
                'event_number': event_number,
                'seg_wheel': segments['seg_wheel'][i],
                'seg_sector': segments['seg_sector'][i],
                'seg_station': segments['seg_station'][i],
                # Include other segment fields as needed
                'seg_hasPhi': segments['seg_hasPhi'][i],
                'seg_hasZed': segments['seg_hasZed'][i],
                'seg_posLoc_x': segments['seg_posLoc_x'][i],
                'seg_posLoc_y': segments['seg_posLoc_y'][i],
                'seg_posLoc_z': segments['seg_posLoc_z'][i],
                'seg_dirLoc_x': segments['seg_dirLoc_x'][i],
                'seg_dirLoc_y': segments['seg_dirLoc_y'][i],
                'seg_dirLoc_z': segments['seg_dirLoc_z'][i],
                'seg_phi_normChi2': segments['seg_phi_normChi2'][i],
                'seg_phi_nHits': segments['seg_phi_nHits'][i],
                'seg_posLoc_x_SL1': segments['seg_posLoc_x_SL1'][i],
                'seg_posLoc_x_SL3': segments['seg_posLoc_x_SL3'][i],
                'seg_posLoc_x_midPlane': segments['seg_posLoc_x_midPlane'][i],
                'seg_posGlb_phi': segments['seg_posGlb_phi'][i],
                'seg_posGlb_eta': segments['seg_posGlb_eta'][i],
                'seg_dirGlb_phi': segments['seg_dirGlb_phi'][i],
                'seg_dirGlb_eta': segments['seg_dirGlb_eta'][i],
                'seg_phi_t0': segments['seg_phi_t0'][i],
                'seg_phi_vDrift': segments['seg_phi_vDrift'][i],
                'seg_phi_normChi2': segments['seg_phi_normChi2'][i],
                'seg_z_normChi2': segments['seg_z_normChi2'][i],
                'seg_z_nHits': segments['seg_z_nHits'][i]
                # Add other fields as needed
            }
            segment_records.append(segment_record)
    df_segments_flat = pd.DataFrame(segment_records)
    return df_segments_flat


def aggregate_digi_tuples(group):
    return pd.Series({
        'digi_tuples': list(zip(group['digi_layer'], group['digi_wire'], group['digi_time']))
    })
    
def group_digi_data(df_digis_flat):
    # Grouping the digis up to station level
    digi_group_keys = ['event_number', 'digi_wheel', 'digi_sector', 'digi_station']
    digi_agg_dict = {
        'digi_superLayer': list,
        'digi_layer': list,
        'digi_wire': list,
        'digi_time': list
    }
    df_digis_grouped = df_digis_flat.groupby(digi_group_keys).agg(digi_agg_dict).reset_index()
    print("\nDigi data grouped successfully.")
    return df_digis_grouped

def group_segment_data(df_segments_flat):
    # Grouping the segments up to station level
    segment_group_keys = ['event_number', 'seg_wheel', 'seg_sector', 'seg_station']
    agg_dict = {col: list for col in df_segments_flat.columns if col not in segment_group_keys}
    df_segments_grouped = df_segments_flat.groupby(segment_group_keys).agg(agg_dict).reset_index()
    print("\nSegment data grouped successfully.")
    return df_segments_grouped

def merge_digi_segment_data(df_digis_grouped, df_segments_grouped):
    # Rename columns for consistency (we need same labels for merging)
    df_digis_grouped.rename(columns={
        'digi_wheel': 'wheel',
        'digi_sector': 'sector',
        'digi_station': 'station'
    }, inplace=True)
    df_segments_grouped.rename(columns={
        'seg_wheel': 'wheel',
        'seg_sector': 'sector',
        'seg_station': 'station'
    }, inplace=True)

    # Merge the DataFrames
    df_combined = pd.merge(
        df_digis_grouped,
        df_segments_grouped,
        on=['event_number', 'wheel', 'sector', 'station'],
        how='outer',
        suffixes=('_digi', '_seg')
    )

    # Handle missing data
    list_columns = ['digi_superLayer', 'digi_layer', 'digi_wire', 'digi_time',
                    'seg_hasPhi', 'seg_hasZed', 'seg_posLoc_x', 'seg_posLoc_y',
                    'seg_posLoc_z', 'seg_dirLoc_x', 'seg_dirLoc_y', 'seg_dirLoc_z',
                    'seg_phi_normChi2', 'seg_phi_nHits', 'seg_posLoc_x_SL1','seg_posLoc_x_SL3',
                    'seg_posLoc_x_midPlane', 'seg_posGlb_phi', 'seg_posGlb_eta', 'seg_dirGlb_phi',
                    'seg_dirGlb_eta', 'seg_phi_t0', 'seg_phi_vDrift', 'seg_phi_normChi2', 'seg_z_normChi2', 'seg_z_nHits']
    for col in list_columns:
        if col in df_combined.columns:
            df_combined[col] = df_combined[col].apply(lambda x: x if isinstance(x, list) else [])
    print("\nDigi and segment data merged successfully.")
    return df_combined

def summarize_data_availability(df_combined):
    print("\nData Availability Summary:")
    print("\nCombined Data:")
    if df_combined.empty:
        print("No combined data available.")
    else:
        print(df_combined[['event_number', 'wheel', 'sector', 'station']].drop_duplicates())

def find_first_non_empty_filter(df_combined):
    for idx, row in df_combined.iterrows():
        event_number = row['event_number']
        wheel = row['wheel']
        sector = row['sector']
        station = row['station']
        # You can add additional checks if necessary
        return event_number, wheel, sector, station
    return None  # No data found

def filter_combined_data(df_combined, event_number, wheel, sector, station):
    filtered_data = df_combined[
        (df_combined['event_number'] == event_number) &
        (df_combined['wheel'] == wheel) &
        (df_combined['sector'] == sector) &
        (df_combined['station'] == station)
    ]
    return filtered_data

def process_digis_by_station_combined_dummy(df_combined, specific_event_number, specific_wheel, specific_sector):
    # Filter the combined data for the specific event, wheel, sector
    filtered_data = df_combined[
        (df_combined['event_number'] == specific_event_number) &
        (df_combined['wheel'] == specific_wheel) &
        (df_combined['sector'] == specific_sector)
    ]

    if filtered_data.empty:
        print("\nNo data found matching the specified criteria.")
        return
    else:
        print(f"\nProcessing data for Event {specific_event_number}, Wheel {specific_wheel}, Sector {specific_sector}")

    # Get the unique stations
    stations = filtered_data['station'].unique()
    print(f"Stations in this event, wheel, sector: {stations}")

    for station in stations:
        # Get the data for this station
        station_data = filtered_data[filtered_data['station'] == station]
        # Since we grouped up to station level, there should be one row per station
        for idx, row in station_data.iterrows():
            print(f"\nStation {station}:")

            # Process digis
            if 'digi_layer' in row and row['digi_layer']:
                superlayers = row['digi_superLayer']
                layers = row['digi_layer']
                wires = row['digi_wire']
                times = row['digi_time']
                # Create a list of tuples
                hits = list(zip(superlayers, layers, wires, times))
                print(f"Digis (SuperLayer, Layer, Wire, Time):")
                for hit in hits:
                    print(f"  SuperLayer: {hit[0]}, Layer: {hit[1]}, Wire: {hit[2]}, Time: {hit[3]}")
            else:
                print("No digis in this station.")

            # Process segments
            if 'seg_posLoc_x' in row and row['seg_posLoc_x']:
                seg_hasPhi = row['seg_hasPhi']
                seg_hasZed = row['seg_hasZed']
                seg_posLoc_x = row['seg_posLoc_x']
                seg_posLoc_y = row['seg_posLoc_y']
                seg_posLoc_z = row['seg_posLoc_z']
                seg_dirLoc_x = row['seg_dirLoc_x']
                seg_dirLoc_y = row['seg_dirLoc_y']
                seg_dirLoc_z = row['seg_dirLoc_z']
                seg_posGlb_phi = row['seg_posGlb_phi']
                seg_posGlb_eta = row['seg_posGlb_eta']
                seg_dirGlb_phi = row['seg_dirGlb_phi']
                seg_dirGlb_eta = row['seg_dirGlb_eta']
                seg_phi_t0 = row['seg_phi_t0']
                seg_phi_vDrift = row['seg_phi_vDrift']
                seg_phi_normChi2 = row['seg_phi_normChi2']
                seg_phi_nHits = row['seg_phi_nHits']

                # Number of segments
                n_segments = len(seg_posLoc_x)
                print(f"Number of Segments: {n_segments}")
                print("Segments:")
                for i in range(n_segments):
                    print(f"  Segment {i+1}:")
                    print(f"    hasPhi: {seg_hasPhi[i]}, hasZed: {seg_hasZed[i]}")
                    print(f"    pos_SL1: {seg_posLoc_x[i]}, pos_SL3: {seg_posLoc_y[i]}, pos_midPlane: {seg_posLoc_z[i]}")
                    print(f"    pos_x: {seg_posLoc_x[i]}, pos_y: {seg_posLoc_y[i]}, pos_z: {seg_posLoc_z[i]}")
                    print(f"    dirLoc_x: {seg_dirLoc_x[i]}, dirLoc_y: {seg_dirLoc_y[i]}, dirLoc_z: {seg_dirLoc_z[i]}")
                    print(f"    posGlb_phi: {seg_posGlb_phi[i]}, posGlb_eta: {seg_posGlb_eta[i]}")
                    print(f"    dirGlb_phi: {seg_dirGlb_phi[i]}, dirGlb_eta: {seg_dirGlb_eta[i]}")
                    print(f"    phi_normChi2: {seg_phi_normChi2[i]}, phi_nHits: {seg_phi_nHits[i]}")
                    print(f"    phi_t0: {row['seg_phi_t0'][i]}, phi_vDrift: {row['seg_phi_vDrift'][i]}")
                    print(f"    phi_normChi2: {row['seg_phi_normChi2'][i]}, z_normChi2: {row['seg_z_normChi2'][i]}, z_nHits: {row['seg_z_nHits'][i]}")
                    
            else:
                print("No segments in this station.")
                
def find_station_with_most_digis(df_combined, return_digi_counts=False):
    """
    Finds the station with the highest total number of digis across all events.

    Parameters:
        df_combined (pd.DataFrame): The combined DataFrame containing digis and segments data.
        return_digi_counts (bool): If True, returns the digi_counts DataFrame as well.

    Returns:
        max_station_info (dict): A dictionary containing the station information and total digis.
        digi_counts (pd.DataFrame): (Optional) DataFrame with total digis per station.
    """
    # Ensure that 'digi_layer' exists in the DataFrame
    if 'digi_layer' not in df_combined.columns:
        print("The DataFrame does not contain 'digi_layer' column.")
        return None

    # Create a copy to avoid modifying the original DataFrame
    df = df_combined.copy()

    # Calculate the number of digis for each row (station in an event)
    df['n_digis'] = df['digi_layer'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # For each station, find the maximum number of digis observed in any single event
    #station_max_digis = df.groupby(['wheel', 'sector', 'station'])['n_digis'].max().reset_index()

    # Find the highest maximum number of digis among all stations
    highest_max_digis = df['n_digis'].max()

    # Get the station(s) with this highest maximum number of digis
    max_stations = df[df['n_digis'] == highest_max_digis]

    print(f"\nStation(s) with the highest number of digis in a single event:")
    for idx, row in max_stations.iterrows():
        print(f"Event: {row['event_number']}, Wheel: {row['wheel']}, Sector: {row['sector']}, Station: {row['station']}, Digis: {row['n_digis']}")
    if return_digi_counts:
        return max_stations, highest_max_digis
    else:
        return max_stations


def get_digis_for_event(df_combined, event_number, wheel, sector, station):
    """
    Returns a DataFrame containing the digis for a specific event number, wheel, sector, and station.

    Args:
        df_combined (pd.DataFrame): The combined DataFrame containing digi and segment data.
        event_number (int): The event number.
        wheel (int): The wheel number.
        sector (int): The sector number.
        station (int): The station number.

    Returns:
        pd.DataFrame: A DataFrame with the digis for the specified event, wheel, sector, and station.
    """
    # Filter the combined data for the specific event, wheel, sector, and station
    filtered_data = df_combined[
        (df_combined['event_number'] == event_number) &
        (df_combined['wheel'] == wheel) &
        (df_combined['sector'] == sector) &
        (df_combined['station'] == station)
    ]

    if filtered_data.empty:
        print("\nNo digis found for the specified event, wheel, sector, and station.")
        return pd.DataFrame()  # Return an empty DataFrame
    else:
        # Create a DataFrame with the digis (digi_superLayer, digi_layer, digi_wire, digi_time)
        digis_data = []
        for idx, row in filtered_data.iterrows():
            for i in range(len(row['digi_layer'])):
                digis_data.append({
                    'superLayer': row['digi_superLayer'][i],
                    'layer': row['digi_layer'][i],
                    'wire': row['digi_wire'][i],
                    'time': row['digi_time'][i]
                })
        return pd.DataFrame(digis_data)

# Function to return the unique segments for a given event number, wheel, and station
def get_segments_for_event(df_combined, event_number, wheel, sector, station):
    """
    Returns the unique segments for a given event number, wheel, and station.

    Args:
        df_combined (pd.DataFrame): The combined DataFrame containing digi and segment data.
        event_number (int): The event number.
        wheel (int): The wheel number.
        station (int): The station number.

    Returns:
        pd.DataFrame: A DataFrame with the segment details for the specified event, wheel, and station.
    """
    # Filter the combined data for the specific event, wheel, and station
    filtered_data = df_combined[
        (df_combined['event_number'] == event_number) &
        (df_combined['wheel'] == wheel) &
        (df_combined['sector'] == sector) &
        (df_combined['station'] == station)
    ]

    if filtered_data.empty:
        print("\nNo segments found for the specified event, wheel, and station.")
        return pd.DataFrame()  # Return an empty DataFrame
    else:
        # Create a DataFrame with the segment details
        segments_data = []
        for idx, row in filtered_data.iterrows():
            for i in range(len(row['seg_posLoc_x'])):
                segments_data.append({
                    'hasPhi': row['seg_hasPhi'][i],
                    'hasZed': row['seg_hasZed'][i],
                    'posLoc_x': row['seg_posLoc_x'][i],
                    'posLoc_y': row['seg_posLoc_y'][i],
                    'posLoc_z': row['seg_posLoc_z'][i],
                    'dirLoc_x': row['seg_dirLoc_x'][i],
                    'dirLoc_y': row['seg_dirLoc_y'][i],
                    'dirLoc_z': row['seg_dirLoc_z'][i],
                    'phi_normChi2': row['seg_phi_normChi2'][i],
                    'phi_nHits': row['seg_phi_nHits'][i],
                    'posLoc_x_SL1': row['seg_posLoc_x_SL1'][i],
                    'posLoc_x_SL3': row['seg_posLoc_x_SL3'][i],
                    't0': row['seg_phi_t0'][i],
                    'vDrift': row['seg_phi_vDrift'][i]
                })
        return pd.DataFrame(segments_data)


def generate_combined_dataframe(root_file_path, tree_name):
    """
    Generate a combined DataFrame of segments and digis from a ROOT file.
    
    Parameters:
    - root_file_path (str): Path to the ROOT file containing digis and segments.
    - tree_name (str): Name of the tree in the ROOT file.
    
    Returns:
    - pd.DataFrame: Combined DataFrame containing both digis and segments.
    """
    # Load the ROOT file and tree
    tree = load_root_file(root_file_path, tree_name)
    if tree is None:
        print("Failed to load ROOT file or tree.")
        return None
    
    # Define the branches to extract
    branches_to_extract = [
        'event_eventNumber',
        'digi_nDigis',
        'digi_wheel',
        'digi_sector',
        'digi_station',
        'digi_superLayer',
        'digi_layer',
        'digi_wire',
        'digi_time',
        'seg_nSegments',
        'seg_wheel',
        'seg_sector',
        'seg_station',
        'seg_hasPhi',
        'seg_hasZed',
        'seg_posLoc_x',
        'seg_posLoc_y',
        'seg_posLoc_z',
        'seg_dirLoc_x',
        'seg_dirLoc_y',
        'seg_dirLoc_z',
        'seg_phi_normChi2',
        'seg_phi_nHits',
        'seg_posLoc_x_SL1',
        'seg_posLoc_x_SL3',
        'seg_posLoc_x_midPlane',
        'seg_posGlb_phi',
        'seg_posGlb_eta',
        'seg_dirGlb_phi',
        'seg_dirGlb_eta',
        'seg_phi_t0',
        'seg_phi_vDrift',
        'seg_z_normChi2',
        'seg_z_nHits'
    ]
    
    # Extract data from the tree
    arrays = extract_data(tree, branches_to_extract)
    if arrays is None:
        print("Failed to extract data from ROOT file.")
        return None
    
    # Define segment numeric branches
    seg_numeric_branches = [
        'seg_wheel', 'seg_sector', 'seg_station',
        'seg_hasPhi', 'seg_hasZed',
        'seg_posLoc_x', 'seg_posLoc_y', 'seg_posLoc_z',
        'seg_dirLoc_x', 'seg_dirLoc_y', 'seg_dirLoc_z',
        'seg_phi_normChi2', 'seg_phi_nHits',
        'seg_posLoc_x_SL1', 'seg_posLoc_x_SL3',
        'seg_posLoc_x_midPlane', 'seg_posGlb_phi',
        'seg_posGlb_eta', 'seg_dirGlb_phi',
        'seg_dirGlb_eta', 'seg_phi_t0',
        'seg_phi_vDrift', 'seg_phi_normChi2',
        'seg_z_normChi2', 'seg_z_nHits'
    ]
    
    # Build events DataFrame
    df_events = build_events_dataframe(arrays, seg_numeric_branches)
    if df_events.empty:
        print("Events DataFrame is empty. No data to process.")
        return None
    
    # Prepare digi and segment DataFrames
    df_digis_flat = prepare_digi_data(df_events)
    df_segments_flat = prepare_segment_data(df_events)
    
    # Group and merge digi data
    df_digis_grouped = group_digi_data(df_digis_flat)
    
    # Group and merge segment data
    df_segments_grouped = group_segment_data(df_segments_flat)
    
    # Merge digis and segments into a combined DataFrame
    df_combined = merge_digi_segment_data(df_digis_grouped, df_segments_grouped)
    
    print(f"Combined DataFrame generated with shape: {df_combined.shape}")
    return df_combined

import random

def print_random_events_with_counts(df_combined, num_events=5, seed=None):
    """
    Select and print a specified number of random events that have both digis and segments.
    
    For each selected event, prints:
    - Event Number
    - Wheel
    - Sector
    - Station
    - Number of Digis
    - Number of Segments
    
    Additionally, returns a list of dictionaries containing the key variables for each event.
    
    Parameters:
    - df_combined (pd.DataFrame): Combined DataFrame containing digis and segments data.
    - num_events (int): Number of random events to select and print (default is 5).
    
    Returns:
    - List[Dict]: A list where each dictionary contains 'event_number', 'wheel', 'sector', 'station'
                  for one of the selected events.
    """
    # Define segment indicator columns based on the merged DataFrame structure
    # Adjust these column names if your DataFrame has different segment indicators
    segment_indicator_columns = ['seg_hasPhi', 'seg_hasZed', 'seg_posLoc_x']
    
    # Check which segment indicator columns are present
    existing_segment_columns = [col for col in segment_indicator_columns if col in df_combined.columns]
    
    if not existing_segment_columns:
        print("No suitable segment indicator columns found in the DataFrame.")
        return []
    
    # Filter events that have both digis and segments
    eligible_events = df_combined[
        (df_combined['digi_superLayer'].apply(lambda x: len(x) > 0)) &
        (df_combined[existing_segment_columns].apply(lambda row: any(len(x) > 0 for x in row), axis=1))
    ]
    
    # Get unique (event_number, wheel, sector, station) combinations
    unique_combinations = eligible_events[['event_number', 'wheel', 'sector', 'station']].drop_duplicates()
    
    available = len(unique_combinations)
    if available == 0:
        print("No events found with both digis and segments.")
        return []
    
    # Determine the number of events to select
    select_count = min(num_events, available)
    
    # Randomly select events
    #add a random seed to make the selection reproducible
    if seed is None:
        seed = random
    else:
        seed = int(seed)
    #generate a random seed
    random.seed(seed)
    
    selected_events = unique_combinations.sample(n=select_count, random_state=random.randint(0, 10000))
    
    
    
    print(f"\nSelected {select_count} Random Event(s) with Both Digis and Segments:\n")
    
    # Initialize a list to store selected event details
    selected_event_info = []
    
    for idx, row in selected_events.iterrows():
        event_number = row['event_number']
        wheel = row['wheel']
        sector = row['sector']
        station = row['station']
        
        # Retrieve the corresponding row in df_combined
        event_row = eligible_events[
            (eligible_events['event_number'] == event_number) &
            (eligible_events['wheel'] == wheel) &
            (eligible_events['sector'] == sector) &
            (eligible_events['station'] == station)
        ].iloc[0]
        
        # Calculate number of digis and segments
        num_digis = len(event_row['digi_superLayer'])
        num_segments = len(event_row['seg_hasPhi'])  # Assuming 'seg_hasPhi' indicates segments
        
        # Print event details
        print(f"Event Number: {event_number}")
        print(f"Wheel: {wheel}, Sector: {sector}, Station: {station}")
        print(f"Number of Digis: {num_digis}")
        print(f"Number of Segments: {num_segments}")
        print("-" * 50)
        
        # Append the event variables to the list
        selected_event_info.append({
            'event_number': event_number,
            'wheel': wheel,
            'sector': sector,
            'station': station
        })
    
    return selected_event_info