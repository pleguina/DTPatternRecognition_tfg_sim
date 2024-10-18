# driftTubes.py
# This library holds the classes for the drift tubes geometry for the CMSSW generation approach.

import logging

class Wire:
    def __init__(self, first_pos, last_pos, num_wires, layer_y, wire_width=4.2, wire_height=1.3):
        self.first_pos = first_pos
        self.last_pos = last_pos
        self.num_wires = num_wires
        self.layer_y = layer_y  # Y position based on local Z
        self.wire_width = wire_width
        self.wire_height = wire_height
        self.positions = self.calculate_wire_positions()

    def calculate_wire_positions(self):
        # Evenly space wires between first_pos and last_pos
        if self.num_wires == 1:
            return [self.first_pos]
        spacing = (self.last_pos - self.first_pos) / (self.num_wires - 1)
        return [
            self.first_pos + i * spacing
            for i in range(self.num_wires)
        ]


class Layer:
    def __init__(self, rawId, layerNumber, cellWidth, cellHeight, cellLength,
                 channels_first, channels_last, channels_total,
                 wireFirst, wireLast, global_pos, local_pos,
                 norm_vect, bounds_width, bounds_thickness, bounds_length):
        """
        Initialize a Layer object.

        Parameters:
        - rawId (int): Raw ID of the layer.
        - layerNumber (int): Layer number within the SuperLayer.
        - cellWidth (float): Width of each cell in the layer.
        - cellHeight (float): Height of each cell in the layer.
        - cellLength (float): Length of each cell in the layer.
        - channels_first (int): First channel number.
        - channels_last (int): Last channel number.
        - channels_total (int): Total number of channels.
        - wireFirst (float): Position of the first wire.
        - wireLast (float): Position of the last wire.
        - global_pos (tuple of float): Global (x, y, z) position.
        - local_pos (tuple of float): Local (x, y, z) position.
        - norm_vect (tuple of float): Normal vector (x, y, z).
        - bounds_width (float): Width from bounds.
        - bounds_thickness (float): Thickness from bounds.
        - bounds_length (float): Length from bounds.
        """
        self.rawId = rawId
        self.layerNumber = layerNumber
        self.cellWidth = cellWidth
        self.cellHeight = cellHeight
        self.cellLength = cellLength
        self.channels_first = channels_first
        self.channels_last = channels_last
        self.channels_total = channels_total
        self.wireFirst = wireFirst
        self.wireLast = wireLast
        self.global_pos = global_pos
        self.local_pos = local_pos
        self.norm_vect = norm_vect
        self.bounds_width = bounds_width
        self.bounds_thickness = bounds_thickness
        self.bounds_length = bounds_length
        self.wires = Wire(wireFirst, wireLast, channels_total, local_pos[2])

    def __repr__(self):
        return (f"Layer(rawId={self.rawId}, layerNumber={self.layerNumber}, "
                f"cellWidth={self.cellWidth}, cellHeight={self.cellHeight}, "
                f"cellLength={self.cellLength}, channels_total={self.channels_total})")
        
    def get_dimension(self, position_type='local', axis='x'):
        """
        Retrieve a specific coordinate from the local or global position.

        Parameters:
        - position_type (str): 'local' or 'global' to specify which position to use.
        - axis (str): 'x', 'y', or 'z' to specify which coordinate to retrieve.

        Returns:
        - float: The requested coordinate value.

        Raises:
        - ValueError: If position_type or axis is invalid.
        """
        if position_type not in ['local', 'global']:
            raise ValueError("position_type must be 'local' or 'global'.")
        if axis not in ['x', 'y', 'z']:
            raise ValueError("axis must be 'x', 'y', or 'z'.")

        axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]
        if position_type == 'local':
            return self.local_pos[axis_index]
        else:
            return self.global_pos[axis_index]
        
    def get_wire_x_position(self, wire_number):
        """
        Get the X position of a specific wire in this layer.

        Parameters:
        - wire_number (int): The wire number (1-based index).

        Returns:
        - float: X position of the wire.
        """
        if 1 <= wire_number <= self.wires.num_wires:
            return self.wires.positions[wire_number - 1]
        else:
            raise ValueError(f"Invalid wire number {wire_number} for Layer {self.layerNumber}")


class SuperLayer:
    def __init__(self, rawId, superLayerNumber, global_pos, local_pos,
                 norm_vect, bounds_width, bounds_thickness, bounds_length,
                 layers):
        """
        Initialize a SuperLayer object.

        Parameters:
        - rawId (int): Raw ID of the SuperLayer.
        - superLayerNumber (int): SuperLayer number within the Chamber.
        - global_pos (tuple of float): Global (x, y, z) position.
        - local_pos (tuple of float): Local (x, y, z) position.
        - norm_vect (tuple of float): Normal vector (x, y, z).
        - bounds_width (float): Width from bounds.
        - bounds_thickness (float): Thickness from bounds.
        - bounds_length (float): Length from bounds.
        - layers (list of Layer): List of Layer objects within the SuperLayer.
        """
        self.rawId = rawId
        self.superLayerNumber = superLayerNumber
        self.global_pos = global_pos
        self.local_pos = local_pos
        self.norm_vect = norm_vect
        self.bounds_width = bounds_width
        self.bounds_thickness = bounds_thickness
        self.bounds_length = bounds_length
        self.layers = layers  # List of Layer objects

    def __repr__(self):
        return (f"SuperLayer(rawId={self.rawId}, superLayerNumber={self.superLayerNumber}, "
                f"bounds_width={self.bounds_width}, bounds_length={self.bounds_length}, "
                f"layers={len(self.layers)})")
        
    def get_dimension(self, position_type='local', axis='x'):
        """
        Retrieve a specific coordinate from the local or global position.

        Parameters:
        - position_type (str): 'local' or 'global' to specify which position to use.
        - axis (str): 'x', 'y', or 'z' to specify which coordinate to retrieve.

        Returns:
        - float: The requested coordinate value.

        Raises:
        - ValueError: If position_type or axis is invalid.
        """
        if position_type not in ['local', 'global']:
            raise ValueError("position_type must be 'local' or 'global'.")
        if axis not in ['x', 'y', 'z']:
            raise ValueError("axis must be 'x', 'y', or 'z'.")

        axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]
        if position_type == 'local':
            return self.local_pos[axis_index]
        else:
            return self.global_pos[axis_index]

    def get_layer(self, layer_number):
        """
        Retrieve a Layer object by its number.

        Parameters:
        - layer_number (int): The layer number within the SuperLayer.

        Returns:
        - Layer: The corresponding Layer object.
        """
        for layer in self.layers:
            if layer.layerNumber == layer_number:
                return layer
        raise ValueError(f"Layer number {layer_number} not found in SuperLayer {self.superLayerNumber}")


class Chamber:
    def __init__(self, rawId, chamberId, global_pos, local_pos,
                 norm_vect, bounds_width, bounds_thickness, bounds_length,
                 superlayers):
        """
        Initialize a Chamber object.

        Parameters:
        - rawId (int): Raw ID of the Chamber.
        - chamberId (str or int): Chamber identifier.
        - global_pos (tuple of float): Global (x, y, z) position.
        - local_pos (tuple of float): Local (x, y, z) position.
        - norm_vect (tuple of float): Normal vector (x, y, z).
        - bounds_width (float): Width from bounds.
        - bounds_thickness (float): Thickness from bounds.
        - bounds_length (float): Length from bounds.
        - superlayers (list of SuperLayer): List of SuperLayer objects within the Chamber.
        """
        self.rawId = rawId
        self.chamberId = chamberId
        self.global_pos = global_pos
        self.local_pos = local_pos
        self.norm_vect = norm_vect
        self.bounds_width = bounds_width
        self.bounds_thickness = bounds_thickness
        self.bounds_length = bounds_length
        self.superlayers = superlayers  # List of SuperLayer objects

    def __repr__(self):
        return (f"Chamber(rawId={self.rawId}, chamberId={self.chamberId}, "
                f"bounds_width={self.bounds_width}, bounds_length={self.bounds_length}, "
                f"superlayers={len(self.superlayers)})")
    
    def convert_wire_to_xy(self, superLayer_number, layer_number, wire_number):
        """
        Convert digi_wire to (x, z) coordinates using chamber geometry.

        Args:
            superLayer_number (int): SuperLayer number.
            layer_number (int): Layer number.
            wire_number (int): Wire number.

        Returns:
            tuple: (x, z) coordinates of the digi, or (None, None) if conversion fails.
        """
        print(f"\n--- Converting Wire to (x, z) ---")
        print(f"Chamber ID: {self.rawId}")
        print(f"SuperLayer Number: {superLayer_number} (Type: {type(superLayer_number)})")
        print(f"Layer Number: {layer_number} (Type: {type(layer_number)})")
        print(f"Wire Number: {wire_number} (Type: {type(wire_number)})")
        print(f"Chamber has {len(self.superlayers)} superlayers.")
        
        for sl in self.superlayers:
            print(f"  Checking SuperLayerNumber: {sl.superLayerNumber} (Type: {type(sl.superLayerNumber)})")
            comparison = sl.superLayerNumber == superLayer_number
            print(f"    Comparison Result: {comparison} (Type: {type(comparison)})")
        superlayer_obj = self.superlayers[superLayer_number]
            
        if not superlayer_obj:
            logging.error(f"SuperLayer {superLayer_number} not found in Chamber {self.rawId}.")
            return None, None

        # Retrieve the corresponding Layer object
        try:
            layer_obj = superlayer_obj.get_layer(layer_number)
        except ValueError as ve:
            logging.error(f"Layer {layer_number} not found in SuperLayer {superLayer_number}: {ve}")
            return None, None

        # Get the actual X position of the wire
        try:
            x_pos = layer_obj.get_wire_x_position(wire_number)
        except (AttributeError, ValueError) as e:
            logging.error(f"Error retrieving X position for wire {wire_number} in Layer {layer_number}: {e}")
            return None, None

        # Get the Z position of the layer
        try:
            z_pos = layer_obj.get_dimension('local', 'z')
        except AttributeError as e:
            logging.error(f"Error retrieving Z dimension for Layer {layer_number}: {e}")
            return None, None

        return x_pos, z_pos
    def create_chamber_object(chamber_df):
        """
        Create a Chamber object from a chamber DataFrame.
        
        Args:
            chamber_df (pd.DataFrame): DataFrame containing chamber geometry data.
        
        Returns:
            Chamber: The constructed Chamber object.
        """
        # Extract chamber-level information
        rawId = chamber_df['Chamber_rawId'].iloc[0]
        chamberId = chamber_df['Chamber_ID'].iloc[0]
        global_pos = (
            chamber_df['Chamber_global_x'].iloc[0],
            chamber_df['Chamber_global_y'].iloc[0],
            chamber_df['Chamber_global_z'].iloc[0]
        )
        local_pos = (
            chamber_df['Chamber_local_x'].iloc[0],
            chamber_df['Chamber_local_y'].iloc[0],
            chamber_df['Chamber_local_z'].iloc[0]
        )
        norm_vect = (
            chamber_df['Chamber_norm_x'].iloc[0],
            chamber_df['Chamber_norm_y'].iloc[0],
            chamber_df['Chamber_norm_z'].iloc[0]
        )
        bounds_width = chamber_df['Chamber_bounds_width'].iloc[0]
        bounds_thickness = chamber_df['Chamber_bounds_thickness'].iloc[0]
        bounds_length = chamber_df['Chamber_bounds_length'].iloc[0]

        # Extract SuperLayers
        superlayers = []
        for sl_num in chamber_df['SuperLayer_number'].unique():
            sl_df = chamber_df[chamber_df['SuperLayer_number'] == sl_num]
            superlayer_rawId = sl_df['SuperLayer_rawId'].iloc[0]
            sl_global_pos = (
                sl_df['SuperLayer_global_x'].iloc[0],
                sl_df['SuperLayer_global_y'].iloc[0],
                sl_df['SuperLayer_global_z'].iloc[0]
            )
            sl_local_pos = (
                sl_df['SuperLayer_local_x'].iloc[0],
                sl_df['SuperLayer_local_y'].iloc[0],
                sl_df['SuperLayer_local_z'].iloc[0]
            )
            sl_norm_vect = (
                sl_df['SuperLayer_norm_x'].iloc[0],
                sl_df['SuperLayer_norm_y'].iloc[0],
                sl_df['SuperLayer_norm_z'].iloc[0]
            )
            sl_bounds_width = sl_df['SuperLayer_bounds_width'].iloc[0]
            sl_bounds_thickness = sl_df['SuperLayer_bounds_thickness'].iloc[0]
            sl_bounds_length = sl_df['SuperLayer_bounds_length'].iloc[0]

            # Extract Layers within this SuperLayer
            layers = []
            for layer_num in sl_df['Layer_number'].unique():
                layer_df = sl_df[sl_df['Layer_number'] == layer_num]
                layer_rawId = layer_df['Layer_rawId'].iloc[0]
                cellWidth = layer_df['Layer_cellWidth'].iloc[0]
                cellHeight = layer_df['Layer_cellHeight'].iloc[0]
                cellLength = layer_df['Layer_cellLength'].iloc[0]
                channels_first = layer_df['Layer_channels_first'].iloc[0]
                channels_last = layer_df['Layer_channels_last'].iloc[0]
                channels_total = layer_df['Layer_channels_total'].iloc[0]
                wireFirst = layer_df['Layer_wireFirst'].iloc[0]
                wireLast = layer_df['Layer_wireLast'].iloc[0]
                layer_global_pos = (
                    layer_df['Layer_global_x'].iloc[0],
                    layer_df['Layer_global_y'].iloc[0],
                    layer_df['Layer_global_z'].iloc[0]
                )
                layer_local_pos = (
                    layer_df['Layer_local_x'].iloc[0],
                    layer_df['Layer_local_y'].iloc[0],
                    layer_df['Layer_local_z'].iloc[0]
                )
                layer_norm_vect = (
                    layer_df['Layer_norm_x'].iloc[0],
                    layer_df['Layer_norm_y'].iloc[0],
                    layer_df['Layer_norm_z'].iloc[0]
                )
                layer_bounds_width = layer_df['Layer_bounds_width'].iloc[0]
                layer_bounds_thickness = layer_df['Layer_bounds_thickness'].iloc[0]
                layer_bounds_length = layer_df['Layer_bounds_length'].iloc[0]

                layer = Layer(
                    rawId=layer_rawId,
                    layerNumber=int(layer_num),  # Ensure it's an integer
                    cellWidth=cellWidth,
                    cellHeight=cellHeight,
                    cellLength=cellLength,
                    channels_first=int(channels_first),
                    channels_last=int(channels_last),
                    channels_total=int(channels_total),
                    wireFirst=float(wireFirst),
                    wireLast=float(wireLast),
                    global_pos=layer_global_pos,
                    local_pos=layer_local_pos,
                    norm_vect=layer_norm_vect,
                    bounds_width=layer_bounds_width,
                    bounds_thickness=layer_bounds_thickness,
                    bounds_length=layer_bounds_length
                )
                layers.append(layer)

            superlayer = SuperLayer(
                rawId=superlayer_rawId,
                superLayerNumber=int(sl_num),  # Ensure it's an integer
                global_pos=sl_global_pos,
                local_pos=sl_local_pos,
                norm_vect=sl_norm_vect,
                bounds_width=sl_bounds_width,
                bounds_thickness=sl_bounds_thickness,
                bounds_length=sl_bounds_length,
                layers=layers
            )
            superlayers.append(superlayer)

        chamber = Chamber(
            rawId=rawId,
            chamberId=chamberId,
            global_pos=global_pos,
            local_pos=local_pos,
            norm_vect=norm_vect,
            bounds_width=bounds_width,
            bounds_thickness=bounds_thickness,
            bounds_length=bounds_length,
            superlayers=superlayers
        )

        print(f"Created Chamber: {chamber}")
        for sl in chamber.superlayers:
            print(f"  SuperLayer: {sl}")
            for layer in sl.layers:
                print(f"    Layer: {layer}")

        return chamber

