"""
----------------------------------------------------------
        Class definition of a Drift Cell 
----------------------------------------------------------
parent: layer in which the cell is ensembled
idx   : identifier along the X axis -- from 1 to nDriftCells
          _________________________
         |                         |
         |                         |
  1.3 cm |                         |
         |                         |
         |_________________________|
          <------- 4.2 cm --------->

----------------------------------------------------------
"""
import numpy as np

class DriftCell(object):
    
    x = 0
    y = 0 
    height = 1.3
    width = 4.2

    def __init__(self, parent, idx=-1):
        ''' Constructor '''
        self.idx = idx
        self.parent = parent
        return

    # == Getters ==
    def id(self):
        return self.idx
    def get_parent(self):
        return self.parent

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_position_at_min(self):
        return (self.x, self.y)

    def get_center(self):
        halfwidth = self.get_width()*0.5
        halfheight = self.get_height()*0.5
        xmin, ymin = self.get_position_at_min()
        return (xmin + halfwidth, ymin + halfheight)
    
    # == Setters ==
    def set_width(self, width):
        self.width = width
        return 
    def set_height(self, height):
        self.height = height
        return 
    def set_position_at_min(self, x, y):
        self.x = x
        self.y = y
        return     

    def sweep_cell(self, xleft, xright, muon):
        ''' Method to sweep through a cell and see if the muon has passed through there '''
        # -- First, check in the whole cell
        xr = np.linspace(xleft, xright, 100)
        x_cell, y_cell = self.get_position_at_min()
        height = self.get_height()
        # -- Generate the line that defines the muon track inside the cell
        y_values = muon.getY(xr, y_cell + height/2.)        

        sweep = []
        for y in y_values:
            # If the y position is within the cell, store True 
            isIn = (y >= y_cell) and y <= (y_cell+height)
            sweep.append(isIn)

        sweep = np.asarray(sweep)

        # If any of the checks is TRUE, then the muon has passed through there
        isMIn = any(sweep)
        return isMIn

    def isIn(self, muon):
        ''' Method to explicitly check if a Muon is inside a cell'''
        semiCellLeft = False
        semiCellRight = False
        
        # -- Get position of the cell
        x_cell, y_cell = self.get_position_at_min()
        width = self.get_width()
        height = self.get_height()
        
        # Here you don't know the laterality of the muon,
        # so you have to check both

        # Get y position at the left of the cell
        y_position_left  = muon.getY(x_cell, y_cell + height/2. )        

        # Get y position at the right
        y_position_right = muon.getY(x_cell + width, y_cell + height/2.) 

        # If either at the left or right side of the cell, the muon
        # has a y position smaller than the y position of the cell, 
        # then the muon has not passed through the cell
        below_cell = max(y_position_left, y_position_right) < y_cell

        # Same applies for the upper part of the cell
        above_cell = min(y_position_left, y_position_right) > y_cell + height

        if below_cell or above_cell:
            return (False, False, False)

	    # If you ara here, then the muon IS inside the CELL

        # -- Check how IN is it
        isMIn_global  = self.sweep_cell(x_cell, x_cell + width, muon)
        semiCellLeft  = self.sweep_cell(x_cell, x_cell + width/2.0, muon)
        semiCellRight = self.sweep_cell(x_cell+width/2.0, x_cell + width, muon)
    
        return (isMIn_global, semiCellLeft, semiCellRight)
