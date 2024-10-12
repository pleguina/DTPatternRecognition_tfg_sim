import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
import pandas as pd
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
        
        # Define layer offsets based on superlayer
        self.layer_offsets = {1: 0, 3: 4}  # SL1: no offset, SL3: +4 layers
        
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
        plt.ioff()  # Turn off interactive mode for better control
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
        
    def plot_digis(self, df_digis):
        """
        Plots the digis (hits in cells) on the canvas.

        Args:
            df_digis (pd.DataFrame): DataFrame containing digis with columns 'superLayer', 'layer', 'wire', 'time'.
        """
        for index, row in df_digis.iterrows():
            sl = int(row['superLayer'])
            #we just want to plot digis from sl 1 and 3
            if sl != 1 and sl != 3:
                continue
            layer_num = int(row['layer'])
            wire_num = int(row['wire'])
            # Adjust the layer index based on the superlayer
            layer_offset = self.layer_offsets.get(sl, 0)
            layer_index = layer_offset + layer_num - 1  # Adjust for zero-based index

            # Get the layer from the MBstation
            try:
                layer = self.current_DT.get_layers()[layer_index]
            except IndexError:
                print(f"Layer index {layer_index} out of range. Skipping this digi.")
                continue  # Skip if layer index is out of range

            # Adjust for zero-based indexing of wires
            try:
                cell = layer.get_cells()[wire_num - 1]
            except IndexError:
                print(f"Wire index {wire_num - 1} out of range in layer {layer_index}. Skipping this digi.")
                continue  # Skip if wire index is out of range

            # Plot the cell with a different color to indicate a hit
            xmin, ymin = cell.get_position_at_min()
            width = cell.get_width()
            height = cell.get_height()
            self.plot_cell(xmin, ymin, width, height, edgecolor='k', color='red', alpha=0.5)
    
    def plot_segments(self, df_segments):
        """
        Plots the segments as straight lines on the canvas using absolute positions
        and only the x-direction as the angle.

        Args:
            df_segments (pd.DataFrame): DataFrame containing segments with columns 'posLoc_x', 'dirLoc_x', etc.
        """
        cell_height = self.current_DT.get_layers()[0].get_cells()[0].get_height()

        # Get y positions of the layers to determine starting y-position
        layers = self.current_DT.get_layers()
        y_positions = []
        for layer in layers:
            _ , ymin = layer.get_cells()[0].get_position_at_min()
            height = layer.get_cells()[0].get_height()
            #y_pos = ymin + height / 2  # Center of the layer comment as we want the lower left corner
            y_pos = ymin  # Lower left corner of the layer
            y_positions.append(y_pos)

        # Compute y0 as the minimum y position
        y0 = min(y_positions)

        for index, segment in df_segments.iterrows():
            # Use absolute values for positions
            # Extract position and direction components
            pos_x = segment['posLoc_x']
            pos_y = segment['posLoc_y']
            dir_x = segment['dirLoc_x']
            dir_y = segment['dirLoc_y']
            phi_nHits = segment['phi_nHits']

            # Compute the length L of the segment
            L = phi_nHits * cell_height
            
            #for SL3 segments, we need to add the gap between SL1 and SL3
            # quick fix, as there can be segments <4 in SL3 NEED TO FIX
            if phi_nHits > 4:
                L = L + 28.7 - 4*1.3
                
  

            # Map dirLoc_x from [-1, 1] to angle between 0 and 90 degrees (in radians)
            # Since we're only using the x direction as the angle, and plotting in the positive quadrant
            angle = abs(dir_x) * (np.pi / 2)  # Map dirLoc_x from 0 to π/2 radians

            # Compute delta_x and delta_y
            delta_x = L * np.sin(angle)
            delta_y = L * np.cos(angle)
            

            # Starting point
            x0 = pos_x
            y_start = y0  # Starting y position

            # Ending point
            x1 = x0 + delta_x
            y1 = y_start + delta_y

            # Ensure positions are in the positive quadrant
            x0 = x0
            y_start = abs(y_start)
            x1 = abs(x1)
            y1 = abs(y1)
            


            # Plot the line
            self.axes.plot([x0, x1], [y_start, y1], color='blue', linewidth=2, label='Segment' if index == 0 else "")
        # Add legend if needed
        if len(df_segments) > 0:
            self.axes.legend()
