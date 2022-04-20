"""
----------------------------------------------------------
        Class definition for traning patterns
----------------------------------------------------------
geom       : Function (defined in geometry.py) from where the geometry
             is extract.
wh, sc     : Wheel and sector. These are used to get the correct shift
             between superLayers. Shifts are defined in src.MBstation.py.
chamb_name : Type of DT station for the generation of muons.
mode       : To run correlated or uncorrelated patterns.
----------------------------------------------------------
"""

from geometry.CMSDT import CMSDT
from particle_objects.Muon import Muon
from particle_objects.Pattern import Pattern
from utils.DTPlotter import DTPlotter

class DTTrainer(object):
    def __init__(self, chamb_args):
        ''' Constructor '''
        wh, sc, chamb_name = chamb_args

        self.wheel = wh
        self.sc = sc
        
        self.chamb = CMSDT(wh, sc)[chamb_name]
        self.chamb_fail = CMSDT(wh, sc)[chamb_name+"_tofail"]            
        # -- Configurations for the different trainings
        self.modes = {}
        # -- Output containers
        self.allMuons    = {}
        self.allSeeds    = {}
        self.allPatterns = {}
        return

    def get_chamb(self):
        ''' Get the current chamber used as seeder '''
        return self.chamb

    def get_chamb_fail(self):
        ''' Get the current chamber used as failer '''
        return self.chamb_fail

    def get_patterns(self):
        ''' Get generated patterns '''
        return self.allPatterns   

    def get_muons(self):
        ''' Get generated muon objects '''
        return self.allMuons

    def get_seeds(self):
        ''' Get generated seeds '''
        return self.allSeeds

    def load_training_config(self, modekey, mode):
        ''' Add a new training configuration '''
        self.modes[modekey] = mode
        return
    
    def train(self):
        ''' Run training for each loaded training configuration '''
        for mode in self.modes:
            seeders, failers, uncorr = self.modes[mode]
            self.allPatterns[mode] = []
            self.allMuons[mode] = []
            self.allSeeds[mode] = []
            self.generate_patterns(mode, seeders, failers, uncorr)
        return

    def generate_patterns(self, mode, seeders , failers, uncorr):   
        ''' 
        Algorithm to generate patterns
        -------------------------------------------------------
        seeder : Lower part of the chamber used to generate muons
        failer : Upper part of the chamber used to discard muons
                    that don't fall within the chamber.
        '''
        seeder = self.chamb
        failer = self.chamb_fail
        layers_seeder = [seeder.get_layers()[iLayer] for iLayer in seeders]
        layers_failer = [failer.get_layers()[iLayer] for iLayer in failers]


        # -- Some parameters
        slopes = (-1, +1) # Generate muons with negative and positive slopes
        lateralities = (0.25, 0.75) # Lateralities considered
        mmax = 0.3  

        # -- Start generating patterns
        for lat in lateralities:
            for slope in slopes:
                for layer_s in layers_seeder:
                    print("== Seeding in layer {} in seeder".format(layer_s.id()))                    
                    cell_s = layer_s.get_cell(1) 

                    x0, y0 = cell_s.get_position_at_min()
                    x0    += cell_s.get_width()*lat
                    y0    += cell_s.get_height()/2.0 

                    for layer_f in reversed(layers_failer):
                        if (layer_f.id() <= layer_s.id()) and uncorr: 
                            continue

                        print("\t - Seeding in layer {} in failer".format(layer_f.id()))                    
                        cells_f = layer_f.get_cells()
                        
                        for cell_f in cells_f:
                            if slope > 0 and cell_f.id() < 0: continue
                            if slope < 0 and cell_f.id() > 0 : continue 
                            for semicell in lateralities:
                                xf, yf = cell_f.get_position_at_min()
                                xf += semicell*cell_f.get_width()
                                yf += cell_f.get_height()/2.0

                                if abs(xf-x0) < 0.1*cell_f.get_width():
                                    m = 100000
                                else:
                                    m = (yf-y0)/(xf - x0)

                                if abs(m) < mmax: continue

                                seed = [layer_s.id(), 
                                        layer_f.id(), 
                                        cell_f.id() - cell_s.id()]
                                muon = Muon(x0, y0, m)
                                failer.check_in(muon)
                                hits = muon.getPattern()
                                pat = Pattern(seed, hits)

                                self.allMuons[mode].append(muon)
                                self.allSeeds[mode].append(seed)
                                self.allPatterns[mode].append(pat)

        return 

    def plot_muons(self, outpath):
        ''' Method to plot identified muon trajectories in a chamber '''
        chamb = self.get_chamb_fail() # Use the fail chamber to plot muon trajectories
                                        # This basically centers them... Not important

        for mode in self.modes:
            muons = self.get_muons()[mode] 
            dtplotter = DTPlotter(chamb)

            dtplotter.plot_chamber()
            dtplotter.plot_muons(muons)
            dtplotter.save_canvas("%s/"%outpath, mode)

        return
            
