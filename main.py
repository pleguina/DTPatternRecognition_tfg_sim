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


# Main script 
if __name__ == "__main__":    
    opts = add_parsing_options()
    mode = opts.mode
    geom = opts.geometry
    outpath = opts.outpath

    if (mode == "train"):
        
        MB1_right = pattern_trainer(geom, (-1, 1, "MB1"))
        MB1_right.generate_patterns("correlated")
        MB1_right.generate_patterns("uncorrelated_SL1")
        MB1_right.generate_patterns("uncorrelated_SL3")

        pats = MB1_right.get_patterns() 

        save_pickle(outpath, "MB1_right", pats)  
        pickle_toc(outpath, "MB1_right", "MB1_right.pck")

        # == Plot some of them
        muons_to_plot = []
        muons = MB1_right.get_muons()
        seeds = MB1_right.get_seeds()
        MB = MB1_right.get_chamb_fail()
        for mu, seed in enumerate(seeds):
            ls, lf, _ = seed
            if ls == 3 and lf == 7: 
                muons_to_plot.append(muons[mu])
        p = pattern_plotter(MB)
        p.plot_muons(muons_to_plot)
        p.save_canvas(outpath, "prueba")

    elif (mode == "plot"):
        plotter = pattern_plotter()

    


  








