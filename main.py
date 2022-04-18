'''
Code for dealing with Pseudo-Bayes patterns 
'''

# === Main imports
# -- Utilities
from utils.pattern_plotter import *
from utils.geometry import *
from utils.pickle_toc import *

# -- Source files
# --- Geometry stuff
from src.MBstation import *
from src.Layer import *
from src.DriftCell import *
from src.Primitive import *
from src.Pattern import *

# --- Plugins 
from src.pattern_trainer import pattern_trainer

# -- Extra python libraries
import argparse 
import re

def add_parsing_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode','-m', metavar = "mode",  dest = "mode", default = None)  
    parser.add_argument('--geom','-g', metavar = "geometry",  dest = "geometry", default = CMS)
    parser.add_argument('--outpath', '-o', metavar = "outpath", dest = "outpath", default = "./results")
    return parser.parse_args()


def make_new_training(geom, config, train_name):
    ''' Function to perform new trainings '''
    trainer = pattern_trainer(geom, config)

    train.generate_patterns("correlated")
    train.generate_patterns("uncorrelated_SL1")
    train.generate_patterns("uncorrelated_SL3")

    trained_patterns = trainer.get_patterns()

    save_pickle(outpath, train_name, trained_patterns)  
    pickle_toc(outpath, train_name, "%s.pck"%train_name)

    return trainer

# Main script 
if __name__ == "__main__":    
    opts = add_parsing_options()
    mode = opts.mode
    geom = opts.geometry
    outpath = opts.outpath

    if (mode == "train"):
        train_configs = {"MB1_right" : (-1, 0, "MB1") # Example of an MB1 chamber with right shift between SLs
                         "MB1_left"  : (+1, 0, "MB1") # Example of an MB1 chamber with left shift between SLs
                         "MB2_right" : (+1, 0, "MB2") # Example of an MB2 chamber with right shift between SLs
                         "MB2_left"  : (-1, 0, "MB2") # Example of an MB2 chamber with left shift between SLs
                         "MB3"       : (0 , 0, "MB3") # Example of an MB3 
                         "MB4_right" : (+1, 0, "MB4") # Example of an MB4 chamber with right shift between SLs
                         "MB4"       : (0 , 3, "MB4") # Example of an MB4 chamber with no shift between SLs
                         "MB4_left"  : (-1, 0, "MB4") # Example of an MB4 chamber with left shift between SLs
                        }        

        for train in train_configs:
            print("Training patterns for config: %s"%train)
            trainer = make_new_training(geom, train_configs[train])
            

    elif (mode == "plot"):
        plotter = pattern_plotter()

    


  








