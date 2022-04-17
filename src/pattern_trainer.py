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

from src.Muon import * 
from src.Pattern import *

class pattern_trainer(object):
    available_modes = [
        "correlated", 
        "uncorrelated_SL1",
        "uncorrelated_SL3"
        ]
    mmax = 0.3

    def __init__(self, geom, chamb_args):
        wh, sc, chamb_name = chamb_args
        self.wheel = wh
        self.sc = sc
        self.chamb = geom(wh, sc)[chamb_name]
        self.chamb_fail = geom(wh, sc)[chamb_name+"_tofail"]            

        # -- Output containers
        self.allMuons    = []
        self.allSeeds    = []
        self.allPatterns = []
        return

    def get_chamb(self):
        return self.chamb
    def get_chamb_fail(self):
        return self.chamb_fail

    def get_patterns(self):
        return self.allPatterns   

    def get_muons(self):
        return self.allMuons

    def get_seeds(self):
        return self.allSeeds

    def generate_patterns(self, mode):   
        ''' 
        Algorithm to generate patterns 
        -------------------------------------------------------
        mode   : This affects how we seed layers (we have to avoid
                 double counting of patterns, which can happen in uncorrelated)
        seeder : Lower part of the chamber used to generate muons
        failer : Upper part of the chamber used to discard muons
                 that don't fall within the chamber.
        '''

        if mode not in self.available_modes:
            raise RuntimeError("Mode %s not available. Choose from: %s"%
            (mode, ",".join(self.available_modes)))
        
        uncorr = True if "uncorrelated" in mode else False

        seeder = self.chamb
        failer = self.chamb_fail
        layers_seeder = []
        layers_failer = []

        # == Fetch layers used for seeding patterns 
        if mode == "correlated":
            layers_seeder = seeder.get_layers()[:4]
            layers_failer = failer.get_layers()[4:]
        elif mode == "uncorrelated_SL1":
            layers_seeder = seeder.get_layers()[:4]
            layers_failer = failer.get_layers()[:4]
        elif mode == "uncorrelated_SL3":
            layers_seeder = seeder.get_layers()[4:]
            layers_failer = failer.get_layers()[4:]

        # -- Some parameters
        slopes = (-1, +1) # Generate muons with negative and positive slopes
        lateralities = (0.25, 0.75) # Lateralities considered

        # -- Start generating patterns
        for lat in lateralities:
            for slope in slopes:
                for layer_s in layers_seeder:
                    print("Layer in SL1 = {}".format(layer_s.id()))
                    # -- First cell is used as seeder
                    seed_cell = layer_s.get_cell(1) 

                    # -- Generate position for a given laterality
                    x0, y0 = seed_cell.get_position_at_min()
                    x0 += seed_cell.get_width()*lat
                    y0 += seed_cell.get_height()/2.0 # At center of cell

                    # -- Now finish generating muon in the failer chamber
                    for layer_f in reversed(layers_failer):
                        # -- Neglect double counting in uncorrelated.
                        if (layer_f.id() <= layer_s.id()) and uncorr: 
                            continue
                        print("Layer in SL3 = {}".format(layer_f.id()))
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

                                if abs(m) < self.mmax: continue

                                muon = Muon(x0, y0, m)
                                failer.check_in(muon)
                            
                                self.allMuons.append(muon)
                                seed = [layer_s.id(), layer_f.id(), cell_f.id() - seed_cell.id()]
                                self.allSeeds.append(seed)
                                pat = Pattern(seed, muon.getPattern())
                                self.allPatterns.append(pat)
        return 
        

    

