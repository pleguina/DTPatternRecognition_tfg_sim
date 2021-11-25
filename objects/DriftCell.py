class DriftCell(object):
    # === Class definition of a drift cell
    
    # - Attributes
    x = 0
    y = 0 
    height = 0
    width = 0

    # 
    def __init__(self, height, width, idx=-1):
        self.height = height
        self.width = width
        self.idx = idx
        return

    def get_center(self):
        return self.x+self.width/2.0, self.y+self.height/2.0

    def set_position_at_min(self, x, y):
        self.x = x
        self.y = y
        return     
