from objects.DriftCell import DriftCell

class Layer(object):    
    def __init__(self, nDriftCells, additional_cells, cellWidth, cellHeight):
        self.nDriftCells = nDriftCells
        self.DriftCells = []
        self.additional_cells = additional_cells
        self.cellWidth = cellWidth
        self.cellHeight = cellHeight
        self.create_layer()
        return

    def create_layer(self):
        for cell in range(self.nDriftCells):
            unit = DriftCell(self.cellHeight, self.cellWidth, idx = cell - self.additional_cells) # Create a unit
            unit.set_position_at_min(self.cellWidth*cell, 0) # Place at a position
            self.DriftCells.append(unit) # Add to list
        return
    
    def get_cell(self, cell_id):
        # Wires are numbered from 1 to nDriftCells. But python indexses from 0 to 1...
        # Need to be careful with this
        return self.DriftCells[cell_id]
        
    def shift_layer(self, shiftx, shifty):
        for cellInLayer in self.DriftCells:
            x = cellInLayer.x
            y = cellInLayer.y
            w = cellInLayer.width
            h = cellInLayer.height
            cellInLayer.set_position_at_min(x+shiftx*w, y+shifty*h)
        return
