# ----------------------------------- #
#           IMPORT LIBRARIES          #
# ----------------------------------- #
from geometry.MBstation import MBstation
from geometry.Layer import Layer
from geometry.DriftCell import DriftCell

from particle_objects.Primitive import Primitive
from particle_objects.Pattern import Pattern

from utils.DTTrainer import DTTrainer
from utils.DTPlotter import DTPlotter
from utils.rfile_gen import *

import argparse 
import re, os

def add_parsing_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath', '-o', metavar = "outpath", dest = "outpath", default = "./results/")
    return parser.parse_args()

def dict_to_list(dict_):
    list_ = []
    for dictkey in dict_:
        # There's another dictionary here
        for dictkey_2 in dict_[dictkey]:
            list_.extend(dict_[dictkey][dictkey_2])
    return list_

def print_summary(outpath, patterns, patterns_list):
    traduce_dictkey = lambda key: "Lower: L%s, Upper: L%s"%(re.match("s(.*)f(.*)", key).groups()[0],
                                                            re.match("s(.*)f(.*)", key).groups()[1])

    #logfile = open("%s/logfile.txt"%outpath, "w")
    #log = "  ======  SUMMARY: ====== \n"
    #log += "  -- Total generated patterns: %d\n"%len(patterns_list)
    #for mode in patterns:
    #    log += "    Mode: %s\n"%mode
    #    total = 0
    #    for layer_combination in patterns[mode]:
    #        npat = len(patterns[mode][layer_combination])
    #        log += "      * %s: %d\n"%(traduce_dictkey(layer_combination), npat)
    #        total += npat
    #    log += "    total: %d\n"%total
    
    
    for mode in patterns:
        logfile = open("%s/logfile_%s.txt"%(outpath, mode), "w")
        log = " ======  COMPLETE LOG ====== : \n"
        log += "    Mode: %s\n"%mode
        for layer_combination in patterns[mode]:
            log += "      * %s \n"%traduce_dictkey(layer_combination)
            for ipat, pat in enumerate(patterns[mode][layer_combination]): 
                hitlog = "        - Pattern %d: "%ipat
                hits = ["(%d, %d, %d)"%(hit[0], hit[1], hit[2]) for hit in pat.get_hits()]
                hitlog += " | ".join(hits) + "\n"

                log += hitlog
        logfile.write(log)
    return

def launch_trainings(outpath, configname, train_config, modes):
    ''' Function to launch new pattern trainings '''

    if not os.path.exists(outpath+"/"+configname): 
        os.system("mkdir -p %s/%s/"%(outpath, configname))

    trainer = DTTrainer(train_config)

    for mode in modes:
        trainer.load_training_config(mode, modes[mode])

    trainer.train()
    
    patterns = trainer.get_patterns()

    list_patterns = dict_to_list(patterns)


    print_summary(outpath+"/"+configname, patterns, list_patterns)

    # -- Use muon trajectories to generate plots
    trainer.plot_muons(outpath+"/"+configname)

  #  # -- Use patterns to generate rootfiles

  
  #  # -- Generate pickle files
  #  save_pickle(outpath+"/"+configname, configname,  list_patterns)  

  #  # -- Generate C files
  #  pickle_toc(outpath+"/"+configname, configname)

  #  # -- Generate rootfiles
  #  os.system("root -l -b -q %s/%s.cc "%(outpath+"/"+configname, configname))
    return trainer

# Main script 
if __name__ == "__main__":    
    opts = add_parsing_options()
    outpath = opts.outpath

    modes = {
        "correlated_both"  : ( [0, 1, 2, 3], [4, 5, 6, 7], False),
        "uncorrelated_SL1" : ( [0, 1, 2, 3], [0, 1, 2, 3], True),
        "uncorrelated_SL3" : ( [4, 5, 6, 7], [4, 5, 6, 7], True)
    }

    train_configs = {
        "MB1_right"  : (-1, 0, "MB1"), # Example of an MB1 chamber with right shift between SLs
#        "MB1_left"   : (+1, 0, "MB1"), # Example of an MB1 chamber with left shift between SLs
#        "MB2_right"  : (+1, 0, "MB2"), # Example of an MB2 chamber with right shift between SLs
#        "MB2_left"   : (-1, 0, "MB2"), # Example of an MB2 chamber with left shift between SLs
#        "MB3"        : (0 , 0, "MB3"), # Example of an MB3 
#        "MB4_right"  : (+1, 0, "MB4"), # Example of an MB4 chamber with right shift between SLs
#        "MB4"        : (0 , 3, "MB4"), # Example of an MB4 chamber with no shift between SLs
#        "MB4_left"   : (-1, 0, "MB4")  # Example of an MB4 chamber with left shift between SLs
    }

    for configname in train_configs:
        launch_trainings(outpath, configname, train_configs[configname], modes)
    

            

    


  








