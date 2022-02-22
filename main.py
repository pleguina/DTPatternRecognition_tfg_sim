'''
Code for plotting pseudo-bayes/Standard patterns
in DTs
'''


from plotter.plotter import *
from objects.Primitive import *
from geometry import *
import argparse 
import re

def recta(x, x0, slope):
    y = [(xi-x0)*slope for xi in x]
    return y

def get_primitive(line):
    chamber_match = "(.*): Wh(.*) Se(.*) St(..)?"
    chamber_info = re.match(chamber_match, line).groups()
    hits = [int(hit) for hit in line.split("|")[1].split(" ") if hit != '']
    tdcs = [float(tdc) for tdc in line.split("|")[2].split(" ") if tdc != '']
    lats = [int(lat) for lat in line.split("|")[3].split(" ") if lat != '']
    fits = [fit for fit in line.split("|")[4].split(" ") if fit != '']

    prim = Primitive(chamber_info, hits, tdcs, lats, fits)
    
    return prim, chamber_info

def clasify_primitives(logfile):
    ''' Generate a dictionary containing primitive objects'''

    f = open(logfile, "r")
    
    events = {}
    evt = -1
    # == Iterate over the log file
    for line in f.readlines():
        if line[0] in [" ", "\n", "-"]: continue # Skip comments 
        if "Inspecting" in line: # == Would be nice to produce logfiles in different format
            evt = int(line.split(" ")[-2]) # Forget the \n
            events[evt] = {} # Create a new entry for this entry
            continue
        prim, chamber_info = get_primitive(line)
        chamber = "Wh%s_Sec%s_St%s"%(int(chamber_info[1]), int(chamber_info[2]), int(chamber_info[3]))
        if chamber not in events[evt]:
            events[evt][chamber] = []
        else:
            events[evt][chamber].append(prim)

    return events


def add_parsing_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--event','-e', metavar = "event",  dest = "event")  
    parser.add_argument('--logfile','-l', default="EventDumpList_StdToBayes.log", metavar = "logfile",  dest = "logfile")  
    #parser.add_argument('--station','-s', metavar = "station",  dest = "station", required = True)  
    return parser.parse_args()

# Main script 
if __name__ == "__main__":
    
    # Parse options
    opts = add_parsing_options()	
    
    #station = opts.station
    logfile = opts.logfile

    events = clasify_primitives(logfile)
    
    prims = events[12006]["Wh-1_Sec7_St3"]
    
    p = plotter(MB3)
    p.plot_pattern(prims)
    p.save_canvas("Wh-1_Sec7_St3")
    


    

  








