'''
Code for doing DT Trigger
'''

from geometry.CMSDT import CMSDT
from particle_objects.Muon import Muon
from plotter.plotter import Plotter
from plotter.cmsDraw import draw_cms_muon_chambers, drawCMS_RZ
from particle_objects.Digis import *
import matplotlib.pyplot as plt
import numpy as np

import uproot
import pandas as pd


def main():


   

    # Step 2: Open the ROOT file
    root_file_path = 'dtTuples/DTDPGNtuple_12_4_2_Phase2Concentrator_Simulation_89.root'
    tree_name = 'dtNtupleProducer/DTTREE;1'  # Adjust based on your ROOT file structure

    tree = load_root_file(root_file_path, tree_name)
    if tree is None:
        return

    # Step 3: Define all seg_* numeric branches
    seg_numeric_branches = [
        "seg_nSegments",
        "seg_wheel",
        "seg_sector",
        "seg_station",
        "seg_hasPhi",
        "seg_hasZed",
        "seg_posLoc_x",
        "seg_posLoc_y",
        "seg_posLoc_z",
        "seg_dirLoc_x",
        "seg_dirLoc_y",
        "seg_dirLoc_z",
        "seg_posLoc_x_SL1",
        "seg_posLoc_x_SL3",
        "seg_posLoc_x_midPlane",
        "seg_posGlb_phi",
        "seg_posGlb_eta",
        "seg_dirGlb_phi",
        "seg_dirGlb_eta",
        "seg_phi_t0",
        "seg_phi_vDrift",
        "seg_phi_normChi2",
        "seg_phi_nHits",
        "seg_z_normChi2",
        "seg_z_nHits"
    ]

    # Step 4: Define branches to extract
    branches_to_extract = [
        "event_eventNumber",  
        "digi_nDigis", "digi_wheel", "digi_sector", "digi_station", 
        "digi_superLayer", "digi_layer", "digi_wire", "digi_time",
        *seg_numeric_branches
    ]

    # Step 5: Load the branches into a dictionary of numpy arrays
    arrays = extract_data(tree, branches_to_extract)
    if arrays is None:
        return

    # Step 6: Build the events DataFrame
    df_events = build_events_dataframe(arrays, seg_numeric_branches)

    # Step 7: Prepare and group the digi data
    df_digis_flat = prepare_digi_data(df_events)
    if df_digis_flat.empty:
        print("\nNo digi data available.")
        return
    df_digis_grouped = group_digi_data(df_digis_flat)

    # Step 8: Prepare and group the segment data
    df_segments_flat = prepare_segment_data(df_events)
    if df_segments_flat.empty:
        print("\nNo segment data available.")
        # Decide whether to proceed or exit
        df_segments_grouped = pd.DataFrame()  # Empty DataFrame
    else:
        df_segments_grouped = group_segment_data(df_segments_flat)

    # Step 9: Merge the digi and segment data
    df_combined = merge_digi_segment_data(df_digis_grouped, df_segments_grouped)

    # Step 10: Summarize data availability
    summarize_data_availability(df_combined)

    # Step 11: Find first non-empty data
    result = find_first_non_empty_filter(df_combined)
    if result:
        specific_event_number, specific_wheel, specific_sector, specific_station = result
        print(f"\nYou have available data in Event {specific_event_number}, Wheel {specific_wheel}, Sector {specific_sector}, Station {specific_station} !!!")
    else:
        print("\nNo non-empty data found.")
        
    
    process_digis_by_station_combined_dummy(df_combined, specific_event_number, specific_wheel, specific_sector)
    
    # Find the station with the most digis and get digi_counts
    max_station_info, digi_counts = find_station_with_most_digis(df_combined, return_digi_counts=True)
    if max_station_info is not None and not max_station_info.empty:
        row = max_station_info.iloc[0] # Get the first row
        event_number = row['event_number'] 
        wheel = row['wheel']
        sector = row['sector']
        station = row['station']
        n_digis = row['n_digis']
        print(f"\nStation with most digis: Wheel {wheel}, Sector {sector}, Station {station}, Total digis: {n_digis}")
    else:
        print("\nNo station with digis found.")
        
    process_digis_by_station_combined_dummy(df_combined, event_number, wheel, sector)
    
    df_digis = get_digis_for_event(df_combined, event_number, wheel, sector, 2)
    df_segments = get_segments_for_event(df_combined, event_number, wheel, sector,2)
    # Print concise summary of the DataFrame structure
    print("Digis DataFrame structure:")
    print(df_digis.info())  # Provides column names, non-null counts, and data types

    print("\nFirst 5 rows of the Digis DataFrame:")
    print(df_digis.head())  # Shows the first 5 rows of the DataFrame

    print("\nSummary statistics of the Digis DataFrame:")
    print(df_digis.describe())  # Shows statistical summary of the numeric columns

    # Repeat for the segments DataFrame
    print("\nSegments DataFrame structure:")
    print(df_segments.info())

    print("\nFirst 5 rows of the Segments DataFrame:")
    print(df_segments.head())

    print("\nSummary statistics of the Segments DataFrame:")
    print(df_segments.describe())
    MB = CMSDT(wheel, sector, station)  # Replace with actual initialization if necessary    
    #draw_cms_muon_chambers(-1*wheel, sector, station)
    p = Plotter(MB)
    p.plot_digis(df_digis)
    p.plot_segments(df_segments)
    p.save_canvas("prueba")

plt.show()

if __name__ == "__main__":
    main()