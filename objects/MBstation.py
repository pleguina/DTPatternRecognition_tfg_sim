from objects.Layer import *

class MBstation(object):
    nLayers = 8        
    nCells = {"MB1" : 47, "MB2": 59, "MB3": 73, "MB4": 102}
    
    """ 
         ::    Depending on the wheel the sign changes 
             -half width of a Drift cell  # mm for MB1 
              full width of a Drift cell  # mm for MB2
              0                           # mm for MB3
              twice width of a Drift cell # mm for MB4
             ----> This is a given as a parameter to 
                   the constructor. Just posting here for
                   bookeeping
         :: Positive sign: SL1 displaced to the left of SL 3  
            (low cells of SL3 correlate with high cells of SL1)
        
         :: Negative sign: SL1 displaced to the right of SL3 
            (low cells of SL1 correlate with high cells of SL3)
        
         :: MB1 is negative in the positive wheels (and positive sectors of Wh0)
         :: MB2 is positive in the positive wheels (and positive sectors of Wh0)
         :: MB3 has always 0 SL shift
         :: MB4 is a mess"""

    shift_signs = {"Wh<0": {"MB1" : [+1, +1, +1,     +1, +1, +1, +1, +1, +1, +1,      +1, +1],
                            "MB2" : [-1, -1, -1,     -1, -1, -1, -1, -1, -1, -1,      -1, -1],
                            "MB3" : [ 0,  0,  0,      0,  0,  0,  0,  0,  0,  0,       0,  0],
                            "MB4" : [-1, -1, -1, (0, 0), +1, +1, +1, +1,  0, (-1, +1), 0, -1]},
                   "Wh=0": {"MB1" : [+1, -1, -1,     +1, +1, -1, -1, +1, +1, -1,      -1, +1],
                            "MB2" : [-1, +1, +1,     -1, -1, +1, +1, -1, -1, +1,      +1, -1],
                            "MB3" : [ 0,  0,  0,      0,  0,  0,  0,  0,  0,  0,       0,  0],
                            "MB4" : [-1, +1, +1, (0, 0), +1, -1, -1, +1,  0, (+1, -1), 0, -1]},
                   "Wh>0": {"MB1" : [-1, -1, -1,     -1, -1, -1, -1, -1, -1, -1,      -1, -1],
                            "MB2" : [+1, +1, +1,     +1, +1, +1, +1, +1, +1, +1,      +1, +1],
                            "MB3" : [ 0,  0,  0,      0,  0,  0,  0,  0,  0,  0,       0,  0],
                            "MB4" : [+1, +1, +1, (0, 0), -1, -1, -1, -1,  0, (+1, -1), 0, +1]},         

                  }
    def __init__(self, wheel, sector, MBtype, gap, SLShift, additional_cells, cellWidth, cellHeight): 
        # // Create a place where to store Layers
        self.Layers = []
        # // Unpack parsed arguments
        # - Chamber related parameters
        self.wheel = wheel
        self.sector = sector
        self.MBtype = MBtype
        # - Layer related parameters
        self.SLShift = SLShift
        self.nDriftCells = self.nCells[MBtype]
        self.gap   = gap
        self.additional_cells = additional_cells
        # - Cell related parameters
        self.cellWidth = cellWidth
        self.cellHeight = cellHeight

        # // Build the station
        self.build_station(additional_cells)
        return

    def build_station(self, adc):
        # // Build layers:
        # - Generate 8 generic layers
        
        for layer in range(self.nLayers):
            self.Layers.append(Layer(self.nDriftCells, self.additional_cells, self.cellWidth, self.cellHeight))
        
        # - Placed them at the correct position within 
        #   the MB station
        # :: reference for choosing the correct shift: IN2019_003

        if self.wheel>0: sign_wheel = "Wh>0" 
        elif self.wheel<0: sign_wheel = "Wh<0" 
        else: sign_wheel = "Wh0"
            #   IMPORTANT: THOUGHT OF A UNIFIED WAY OF REFERING TO SECTOR NUMBERS 
            #   ( lists in python start counting at 0, sectors start in 1)
            # The -1 below is hardcoded for the moment
        shift_sign = self.shift_signs[sign_wheel][self.MBtype][self.sector-1]
        self.shift_sign = shift_sign
        shift = shift_sign*self.SLShift
        space_SL = self.gap/self.cellHeight # this will be then multiplied by the height of a cell

        # Shifts are done in units of drift cell width and height
        self.Layers[0].shift_layer(-adc             , 0)
        self.Layers[1].shift_layer(-adc-0.5         , 1)
        self.Layers[2].shift_layer(-adc             , 2)
        self.Layers[3].shift_layer(-adc-0.5         , 3)
        self.Layers[4].shift_layer(-adc+shift       , space_SL+4)
        self.Layers[5].shift_layer(-adc-0.5+shift   , space_SL+5)
        self.Layers[6].shift_layer(-adc+shift       , space_SL+6)
        self.Layers[7].shift_layer(-adc-0.5+shift   , space_SL+7)

        self.set_center()
        return
    def set_center(self):
        # == Definition of the center of the chamber:
        # The center is define by two coordinates:
        #  - Y: which is placed at 11.75 cm from the lower part of the chamber
        #  - X: which, taking both SLs into account, should be placed at 
        #       the middle of both SLs.

        centery = 11.75 - 1.8*(self.MBtype in ["MB3", "MB4"])
        # IMPORTANT: One has to take into account
                  # that the middle is not given by SL1, SL3,
                  # but rather it is define also taking into account SL2!!!!
                  # This is way this is not GAP/2.0, that would not be taking into account SL2
        # IMPORTANT: The center in the Y axis varies 1.8 cm for MB3 and MB4
        #            because there is no RPC there

        # == Now, for X position
        # For MB3, there is no shift, so we can use whichever
        # combination of layers to compute the center
        sign = self.shift_sign
        L1 = 2
        L2 = 6 
        # If there is a shift, we have to choose the leftmost one
        # for reference.
        if sign > 0:
            # if sign > 0: # SL1 is the leftmost
            # if sign < 0: # SL3 is the leftmost
            L1 = 2*(sign > 0)+6*(sign < 0)
            L2 = 6*(sign < 0)+2*(sign > 0)


        left = self.get_Layer(L1).get_cell(0).x
        right = self.get_Layer(L2).get_cell(self.nDriftCells-1).x

        centerx = (left + right)/2.0
        self.center = (centerx, centery)
        return

    def get_center(self):
        return self.center

    def get_Layer(self, layer_id):
        return self.Layers[layer_id]    
