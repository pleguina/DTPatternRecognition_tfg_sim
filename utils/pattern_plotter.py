"""
----------------------------------------------------------
        Class definition for plotting patterns
----------------------------------------------------------
MB : Chamber to be plot
----------------------------------------------------------
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.Primitive import *
import os

class pattern_plotter:
    def __init__(self, MB):
        ''' Constructor '''
        # -- Save the station to be plot
        self.current_DT = MB

        # -- Containers for patterns/hits/axes
        self.patterns = []
        self.pattern_axes = []
        self.pattern_labels = []
        self.cells = []

        # -- Create the canvas
        self.create_canvas() 

        # -- Plot the chamber        
        self.plot_chamber()
        return

    def show(self):
        ''' Method to show the canvas '''
        bbox = (0.4, 0.65, 0.3, 0.2)
        axes_with_labels = [ax[0] for ax in self.pattern_axes]
        self.axes.legend(axes_with_labels, self.pattern_labels, mode = "expand", bbox_to_anchor = bbox, shadow = True)
        plt.ion()
        plt.show()
        return

    def create_canvas(self):
        ''' Method to create the canvas '''
        # -- Create the figure object
        fig = plt.figure(figsize=(8, 8))
        axes = fig.add_subplot(111)

        cellWidth = self.current_DT.get_layer(0).get_cell(1).get_width()
        nDriftCells = self.current_DT.get_nDriftCells()
        xlim = nDriftCells*cellWidth*1.05

        axes.set_xlabel("x[cm]")
        axes.set_ylabel("y[cm]")
        axes.set_ylim(-0.5, 50)
        axes.set_xlim(-xlim, xlim)

        # -- Save in attributes for later acces
        self.fig = fig
        self.axes = axes
        return

    def plot_cell(self, xmin, ymin, width, height, linewidth = 1, edgecolor = 'k', color = 'none', alpha=1):
        ''' Method to plot DT cells '''
        cell = patches.Rectangle((xmin, ymin),
                                  width, 
                                  height, 
                                  linewidth=linewidth, 
                                  edgecolor=edgecolor, 
                                  facecolor=color, 
                                  alpha = alpha)
        self.cells.append(cell)
        self.axes.add_patch(cell)
        return

    def plot_chamber(self):        
        MB = self.current_DT
        for layer in MB.get_layers():
            for cell in layer.get_cells():
                xmin, ymin = cell.get_position_at_min()
                width = cell.get_width()
                height = cell.get_height()
                self.plot_cell(xmin, ymin, width, height, edgecolor = 'k')
        return 

    def plot_muons(self, muons, color = '-r'):
        ''' Method to plot muon trajectories '''
        for muon in muons: 
            x_range = np.linspace(-600, 600) # arbitrary        
            if muon.get_slope() < 0:
                color = "-k"
            else:
                color = "-r"
            self.axes.plot(x_range, muon.getY(x_range, 0.), color)
        return

    def save_canvas(self, path, name = "plot"):
        if not os.path.exists("%s/plots"%path):
            os.system("mkdir -p %s/plots"%path)
        self.fig.savefig("%s/plots/%s"%(path, name+".pdf"))
        self.fig.savefig("%s/plots/%s"%(path, name+".png"))
        return
