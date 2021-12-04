import matplotlib.pyplot as plt
import matplotlib.patches as patches
from objects.Primitive import *

class plotter:
    def __init__(self, MB):
        self.patterns = []
        self.pattern_axes = []
        self.pattern_labels = []

        self.current_DT = MB
        self.create_canvas() 
        self.linecolors = {0 : "orangered",
                           1 : "blue"}
        self.fillcolors = {0: ["green", 1],
                           1: ["firebrick", 0.5]}   
        self.markers = {"Bayes" : "o", "Std": "s"}
        self.cells = []
        self.plot_DT()
        return

        
    def clear_patterns(self):
        [ax[i].remove() for ax in self.pattern_axes for i in [0, 1]]
        #[self.pattern_labels[label].remove() for label in range(len(self.pattern_labels))]
        #[pattern[pattern].remove() for pattern in range(len(self.patterns))]
        return

    def show(self):
        # Plot the legend  
        bbox = (0.4, 0.65, 0.3, 0.2)
        axes_with_labels = [ax[0] for ax in self.pattern_axes]
        self.axes.legend(axes_with_labels, self.pattern_labels, mode = "expand", bbox_to_anchor = bbox, shadow = True)
        plt.ion()
        plt.show()
        return

    def create_canvas(self):
        fig = plt.figure(figsize=(10, 8))
        self.fig = fig
        axes = self.fig.add_subplot(111)

        axes.set_xlabel("x[cm]")
        axes.set_ylabel("y[cm]")
        axes.set_ylim(-0.5, 50)
        axes.set_xlim(-10, self.current_DT.nDriftCells*self.current_DT.cellWidth*1.05)
        self.axes = axes
        return

    def plot_cell(self, xmin, ymin, width, height, linewidth = 1, edgecolor = 'k', color = 'none', alpha=1):
        cell = patches.Rectangle((xmin, ymin), width, height, linewidth=linewidth, edgecolor=edgecolor, facecolor=color, alpha = alpha)
        self.cells.append(cell)
        self.axes.add_patch(cell)
        return

    def plot_DT(self):        
        MB = self.current_DT
        for layer in MB.Layers:
            for cell in layer.DriftCells:
                xmin = cell.x
                ymin = cell.y
                width = cell.width
                height = cell.height
                
                self.plot_cell(xmin, ymin, width, height, edgecolor = 'k')
        return 

    
    def plot_pattern(self, prim, counter = 0):
        if isinstance(prim, list):
            patterns = []
            title = "Primitives in Wh%s - sector: %s - station: %s"
            wheel = int(prim[0].wheel)
            sector = int(prim[0].sector)
            station = int(prim[0].station)
            self.axes.set_title(title%(wheel, sector, station))
            for pri in prim: 
                patterns.append(prim)
                self.patterns.append(patterns)
                self.plot_pattern(pri, counter)
                counter+=1
            return

        x = prim.getX()
        y = prim.getY()
        # Plot the track extracted with the fit
        x_range = np.linspace(-1000, 1000, 1000)
        y_range = prim.produce_track(x_range)
        pat_ax, = self.axes.plot(x, y, marker = self.markers[prim.MuonType], markersize = 5, markeredgecolor = self.linecolors[counter], markerfacecolor = self.linecolors[counter], linewidth = 0)
        fit_ax, = self.axes.plot(x_range, y_range, color = self.linecolors[counter], linewidth = 2)
        # Now color the cells 
        hits = prim.hits
        for layer,hit in enumerate(hits):
            xmin = self.current_DT.get_Layer(layer).get_cell(hit).x
            ymin = self.current_DT.get_Layer(layer).get_cell(hit).y
            width = self.current_DT.cellWidth
            height = self.current_DT.cellHeight
            self.plot_cell(xmin, ymin, width, height, 2, 'k', self.fillcolors[counter][0], self.fillcolors[counter][1])
        self.pattern_axes.append([pat_ax, fit_ax])
        self.pattern_labels.append("Evt: %s (%s)"%(prim.id, prim.MuonType))
        return 
       

