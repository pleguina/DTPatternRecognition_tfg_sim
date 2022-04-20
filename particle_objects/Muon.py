"""
----------------------------------------------------------
                Class definition of a Muon
----------------------------------------------------------
x0, y0 : generated positions.
m      : slope of its track through the station.
----------------------------------------------------------
"""
class Muon(object):
    driftvel = 54e-4 # cm/ns
    def __init__(self, x0, y0, m):
        ''' Constructor '''
        self.set_position(x0, y0)
        self.set_slope(m)

        # == Containers to store hits 
        self.hits = []
        self.lateralities = []
        return

    def getPattern(self):
        hits = self.get_hits()
        lats = self.get_lateralities()
        pattern = []
        for ihit, hit in enumerate(hits):
            pattern.append([ hit.parent.idy, hit.idx, lats[ihit] ])

        # -- Save the pattern in the object
        self.pattern = pattern
        return pattern

    def get_lateralities(self):
        return self.lats

    def get_hits(self):
        return self.hits

    # == Setters and Getters
    def add_hit(self, id_cell):
        self.hits.append(id_cell)
        return

    def add_lat(self, lat):
        self.lateralities.append(lat)
        return

    def get_hits(self):
        return self.hits
    
    def get_lateralities(self):
        return self.lateralities

    def set_position(self, x0, y0):
        ''' Set position of the primitive '''
        self.x0 = x0
        self.y0 = y0
        return

    def set_slope(self, m):
        ''' Set slope of the primitive'''
        self.m = m
        return

    def get_slope(self):
        return self.m

    def get_x_seed(self):            
        ''' Get X seed position'''
        return self.x0

    def get_y_seed(self):
        ''' Get Y seed position'''
        return self.y0

    def getY(self, x, ydef):
        ''' This method returns the y position
        of the muon with respect a given X.'''
        globalDTwidth = 4.2
        globalDTheight = 1.3
        if self.m == 100000: 
            return (abs(x - self.x0) < 0.05*globalDTwidth)*ydef + (abs(x - self.x0) > 0.05*globalDTwidth)*10000000000  

        m  = self.get_slope()
        x0 = self.get_x_seed()
        y0 = self.get_y_seed()

        return m*(x-x0)+y0

    def get_position(self):
        ''' Get both position coordinates '''
        return (self.x0, self.y0)

    def get_semicells(self):
        ''' Return semicells'''
        return self.semicells
