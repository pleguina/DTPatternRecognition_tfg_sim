import numpy as np
import geometry as geom

class Primitive(object):
    driftVel = 54e-4 #cm/ns
    
    def __init__(self, chamber_info, hits, tdcs, lats, fits):
        # == Parameters from the log
        # // Event info
        self.event = -1
        self.MuonType  = -1
        self.wheel = -1
        self.sector = -1
        self.station = -1
        self.hits = []
        self.tdcs = []     
        self.lateralities = []
        self.fits = []

        # // Fits
        self.Q = -1
        self.phi = -1
        self.phib = -1
        self.bX = -1
        self.Chi2 = -1
        self.x = -1
        self.tanPsi = -1
        self.t0 = -1
        self.id = -1


        # Processed attributes
        self.trueTimes = []   
        self.x_pos = []
        self.y_pos = []
        self.semicells = []

        self.currentMB = -1
        # Chamber attributes
        self.load_data(chamber_info, hits, tdcs, lats, fits)
        
        return

    # == Setters and Getters

    def load_data(self, chamber_info, hits, tdcs, lats, fits):
        
        self.set_attribute("MuonType", chamber_info[0])
        self.set_attribute("wheel", int(chamber_info[1]))
        self.set_attribute("sector", int(chamber_info[2]))
        self.set_attribute("station", int(chamber_info[3]))
        self.set_attribute("hits", hits)
        self.set_attribute("tdcs", tdcs)
        self.set_attribute("lateralities", lats)
        self.set_attribute("fits", fits)
        self.set_attribute("Q", fits[0])
        self.set_attribute("phi", float(fits[1]))
        self.set_attribute("phib", float(fits[2]))
        self.set_attribute("bX", float(fits[3]))
        self.set_attribute("Chi2", float(fits[4]))
        self.set_attribute("x", float(fits[5]))
        self.set_attribute("tanPsi", float(fits[6]))
        self.set_attribute("t0", float(fits[7]))
        
        # == Load the geometry for the MB station associated to this primitive
        MBobject = geom.stations["MB%d"%self.get_attribute("station")]
        self.set_attribute("currentMB", MBobject)

        self.add_to_chamber(MBobject)
        return

    def isDefined(self, attr):
        if attr not in dir(self):
            raise RuntimeError("[ERROR] Primitive object has no attribute %s"%attr)
        return 
    def set_attribute(self, attr, par):
        self.isDefined(attr)
        self.__setattr__(attr, par)
        return
    def get_attribute(self, attr):
        self.isDefined(attr)
        par = self.__getattribute__(attr)
        return par
    

    def getX(self):            
        return self.x_pos

    def getY(self):
        return self.y_pos

    def set_trueTime(self):
        self.trueTimes = [(tdc - self.t0) for tdc in self.tdcs]
        return

    
    def center_primitive(self, x0, y0):
        
        # In order to propagate properly the x position, we
        # need the quality
        q = self.Q
        hits = self.hits
        qnumber = float(q.replace("Q", ""))
        if qnumber < 6: # This is a confirmed pattern
            # For confirmed pattern, the Y coordinate
            # changes to propagate to the proper SL
            # The shift is always the same
            shifty = 11.75

            # First, we need to know which SL
            # is the one that registered the
            # confirmed pattern
            nsl1 = len(hits[:4])
            nsl3 = len(hits[4:])
            
            shifty = (1*(nsl1 < nsl3) - 1*(nsl1 > nsl3))*shifty
            y0 += shifty
            
            # The x position will be shifted a certain
            # amount of cells, following the distribution
            # of SLs in CMS. 

            L = 2*(nsl1 > nsl3)+6*(nsl1 < nsl3)
            ndriftcells = self.currentMB.nDriftCells
            w0 = self.currentMB.get_Layer(L).get_cell(0).x
            w1 = self.currentMB.get_Layer(L).get_cell(ndriftcells-1).x
            x0 = (w0+w1)/2.0
            
        #else --> Keep the middle of the chamber as it is
        self.x_center = x0
        self.y_center = y0
        return

    def add_to_chamber(self, MB):
        self.currentMB = MB
        self.set_trueTime()
        self.set_position_within_chamber()

        # == Save the reconstructed track:
        # undo the conversions that are applied to the slope
        # during the generation of muon patterns
        # --> source: https://github.com/dtp2-tpg-am/cmssw/blob/AM_11_2_1_filterBayes/L1Trigger/DTTriggerPhase2/src/MuonPathAnalyzerInChamber.cc#L746
        # the extra 1/100 comes from the fact that it is corrected to account for the local position
        # being expressed in tens of micrometers

        # == Tangent represents the direction within the chamber
        tan_psi = self.tanPsi*(-0.001)
        psi = 0.5*np.pi - np.arctan(tan_psi) # Refer to the vertical line
        tan_corrected = np.tan(psi)
        self.tan_corrected = tan_corrected

        # ==  jm_x is the LOCAL position (given in tenths of micro)
        # - But the way this x is calculated depends on
        #   the quality of the muon.
        #   - Confirmed: jm_x is given with respect to the center of 
        #                the SL that has more hits.
        #   - Correlated: jm_x is given by the center of the chamber
        # Thus, we need to center the primitive in the correspondent
        # SL
        # ==
        
        # First, get the center of the chamber
        x0, y0 = self.currentMB.get_center()
        self.center_primitive(x0, y0)
    
        # Once the primitive has been propagated to its correspondent
        # origin of coordinates, propagate it through the x axis
        jm_x = self.x/1e3 # cm

        y0_prim = self.y_center
        x0_prim = self.x_center

        # Store this in a function that can be called anytime
        self.produce_track = lambda x: [((xi-x0_prim-jm_x)*tan_corrected+y0_prim) for xi in x]
        return
 
    
    def set_position_within_chamber(self):
        # X position is computed per hit
        #   - Laterality = 1 --> Use right wall as reference
        #   - Laterality = 0 --> User left wall as 
        MB = self.currentMB
        for layer, hit in enumerate(self.hits):
            # Get important parameters to compute the position within the chamber   
            lat = self.lateralities[layer]
            t = self.trueTimes[layer]
            
            cell = MB.get_Layer(layer).get_cell(hit)

            # lateralities have been stored as booleans. Consider also the cell position
            # within the chamber
            centerx, centery =  cell.get_center()
            x0 = centerx + ((lat == True) - (lat == False))*self.driftVel*t

            self.x_pos.append(x0)
            self.y_pos.append(centery)
  
    

          


    
