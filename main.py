'''
Code for dealing with Pseudo-Bayes patterns 
'''

# === Main imports
# -- Utilities
from utils.pattern_plotter import *
from utils.geometry import *

# -- Source files
# --- Geometry stuff
from src.MBstation import *
from src.Layer import *
from src.DriftCell import *
from src.Primitive import *

# --- Plugins 
from src.pattern_trainer import pattern_trainer

# -- Extra python libraries
import argparse 
import re


def add_parsing_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode','-m', metavar = "mode",  dest = "mode", default = None)  
    parser.add_argument('--geom','-g', metavar = "geometry",  dest = "geometry", default = CMS)
    return parser.parse_args()

# Main script 
if __name__ == "__main__":    
    opts = add_parsing_options()
    mode = opts.mode
    geom = opts.geometry
    
    if (mode == "train"):
        
        trainer = pattern_trainer(geom, (-1, 1, "MB1"))
        trainer.generate_patterns("correlated")
        trainer.generate_patterns("uncorrelated_SL1")
        trainer.generate_patterns("uncorrelated_SL3")

        pats = trainer.get_patterns()
        print(len(pats))

    elif (mode == "plot"):
        plotter = pattern_plotter()

    


  








