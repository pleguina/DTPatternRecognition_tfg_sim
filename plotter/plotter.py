import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
from particle_objects.Muon import *
from geometry.MBstation import MBstation  # Ensure correct import based on your project structure

class Plotter:
    def __init__(self, MB):
        """
        Initializes the Plotter with a given MBstation instance.

        Args:
            MB (MBstation): An instance of MBstation representing a Drift Tube station.
        """
        self.patterns = []
        self.pattern_axes = []
        self.pattern_labels = []

        self.current_DT = MB
        self.linecolors = {0: "orangered",
                           1: "blue"}
        self.fillcolors = {0: ["green", 1],
                           1: ["firebrick", 0.5]}   
        self.markers = {"[Bayes]": "o", "[Std]": "s"}
        self.cells = []
        
        self.create_canvas() 
        self.plot_DT()
    
    def create_canvas(self):
        """
        Creates the plotting canvas with appropriate labels and limits based on the MBstation.
        """
        fig = plt.figure(figsize=(10, 8))
        self.fig = fig
        axes = self.fig.add_subplot(111)

        axes.set_xlabel("x [cm]")
        axes.set_ylabel("y [cm]")
        axes.set_ylim(-0.5, 50)

        # Retrieve number of drift cells and cell width using getter methods
        nDriftCells = self.current_DT.get_nDriftCells()
        try:
            # Assuming all cells have the same width, retrieve from the first cell of the first layer
            cellWidth = self.current_DT.get_layers()[0].get_cells()[0].get_width()
        except IndexError:
            raise ValueError("MBstation has no layers or cells defined.")

        axes.set_xlim(-10, nDriftCells * cellWidth * 1.05)
        self.axes = axes

    def plot_DT(self):
        """
        Plots all Drift Cells in the MBstation.
        """
        for layer in self.current_DT.get_layers():
            for cell in layer.get_cells():
                xmin, ymin = cell.get_position_at_min()
                width = cell.get_width()
                height = cell.get_height()
                self.plot_cell(xmin, ymin, width, height, edgecolor='k')

    def plot_cell(self, xmin, ymin, width, height, linewidth=1, edgecolor='k', color='none', alpha=1):
        """
        Plots a single Drift Cell as a rectangle on the canvas.

        Args:
            xmin (float): The x-coordinate of the cell's lower-left corner.
            ymin (float): The y-coordinate of the cell's lower-left corner.
            width (float): The width of the cell.
            height (float): The height of the cell.
            linewidth (int, optional): The width of the cell's edge. Defaults to 1.
            edgecolor (str, optional): The color of the cell's edge. Defaults to 'k'.
            color (str, optional): The fill color of the cell. Defaults to 'none'.
            alpha (float, optional): The transparency level of the cell. Defaults to 1.
        """
        cell_patch = patches.Rectangle(
            (xmin, ymin), width, height, linewidth=linewidth,
            edgecolor=edgecolor, facecolor=color, alpha=alpha
        )
        self.cells.append(cell_patch)
        self.axes.add_patch(cell_patch)

    def plot_pattern(self, prims):
        """
        Plots patterns (e.g., hits and tracks) on the Drift Tube canvas.

        Args:
            prims (list): A list of Primitive objects containing hit information.
        """
        for prim in prims:
            hits = prim.get_hits()
            x = [hit.get_center()[0] for hit in hits]
            y = [hit.get_center()[1] for hit in hits]
            self.axes.plot(x, y, 'gx', markersize=5)

            # Plot the muon track line
            x_range = np.linspace(0, 600, 1000)  # Adjust the range and resolution as needed
            y_values = [prim.getY(x_val, 0.) for x_val in x_range]
            self.axes.plot(x_range, y_values, '--g')

    def show(self):
        """
        Displays the plot with a legend.
        """
        if not self.pattern_labels:
            print("No patterns to display in the legend.")
        else:
            bbox = (0.4, 0.65, 0.3, 0.2)
            self.axes.legend(
                self.pattern_axes, self.pattern_labels,
                loc='upper left', bbox_to_anchor=bbox, shadow=True
            )
        plt.ion()
        plt.show()

    def save_canvas(self, name):
        """
        Saves the current canvas to both PDF and PNG formats.

        Args:
            name (str): The base name for the saved files.
        """
        os.makedirs("./plots/", exist_ok=True)
        pdf_path = os.path.join("./plots/", f"{name}.pdf")
        png_path = os.path.join("./plots/", f"{name}.png")
        self.fig.savefig(pdf_path)
        self.fig.savefig(png_path)
        print(f"Canvas saved as {pdf_path} and {png_path}")
