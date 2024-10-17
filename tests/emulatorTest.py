import sys
import os

# Add the base directory (one level up from tests) to the system path
# This allows the test to access the modules in the base directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

'''
Generic test to handle the data extraction and processing steps
'''

from geometry.CMSDT import CMSDT
from particle_objects.Muon import *
from plotter.plotter import Plotter
from plotter.dtPlotter import *
from plotter.cmsDraw import *
from particle_objects.Digis import *
import matplotlib.pyplot as plt
import numpy as np

import uproot
import pandas as pd


def main():

    # Step 1: Parse the DT geometry XML
    print("\nParsing DT Geometry XML...")
    df_geometry = parse_dtgeometry_xml("plotter/DTGeometry.xml")
    if df_geometry.empty:
        print("Geometry DataFrame is empty. Check the XML file.")
        return None, None

    #First we generate the combined dataframe (digis and segments) from the ROOT file
    
    df_combined = generate_combined_dataframe('dtTuples/DTDPGNtuple_12_4_2_Phase2Concentrator_Simulation_89.root','dtNtupleProducer/DTTREE')
    
    #Then we print random events with digis and segments to have examples to work with
    # You can add a random seed to the function to get the same events every time (third argument)
    selected_events = print_random_events_with_counts(df_combined, 1, 3)
    
    #Finally we plot a specific event with the plot_specific_event function
    
    for event in selected_events:
            event_number = event['event_number']
            wheel = event['wheel']
            sector = event['sector']
            station = event['station']
    
            plot_specific_event(wheel=wheel, sector=sector, station=station, event_number=event_number,
                                df_combined=df_combined, df_geometry=df_geometry)
            
            plt.show()
    

if __name__ == "__main__":
    main()