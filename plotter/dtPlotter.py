import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define Wire class
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

# Function to parse XML and create DataFrame
def parse_dtgeometry_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []

    for chamber in root.findall('.//Chamber'):
        chamber_rawId = chamber.get('rawId')
        chamber_chamberId = chamber.get('chamberId')  # May be None

        # Extract Chamber Positions
        chamber_global_pos_elem = chamber.find('GlobalPosition')
        if chamber_global_pos_elem is not None and chamber_global_pos_elem.text:
            chamber_global_pos = chamber_global_pos_elem.text.strip('()').split(',')
            if len(chamber_global_pos) != 3:
                print(f"Warning: Chamber rawId {chamber_rawId} has invalid GlobalPosition format.")
                continue
            chamber_global_x, chamber_global_y, chamber_global_z = map(float, chamber_global_pos)
        else:
            print(f"Warning: Chamber rawId {chamber_rawId} missing GlobalPosition.")
            continue

        chamber_local_pos_elem = chamber.find('LocalPosition')
        if chamber_local_pos_elem is not None and chamber_local_pos_elem.text:
            chamber_local_pos = chamber_local_pos_elem.text.strip('()').split(',')
            if len(chamber_local_pos) != 3:
                print(f"Warning: Chamber rawId {chamber_rawId} has invalid LocalPosition format.")
                continue
            chamber_local_x, chamber_local_y, chamber_local_z = map(float, chamber_local_pos)
        else:
            print(f"Warning: Chamber rawId {chamber_rawId} missing LocalPosition.")
            continue

        chamber_norm_vect_elem = chamber.find('NormalVector')
        if chamber_norm_vect_elem is not None and chamber_norm_vect_elem.text:
            chamber_norm_vect = chamber_norm_vect_elem.text.strip('()').split(',')
            if len(chamber_norm_vect) != 3:
                print(f"Warning: Chamber rawId {chamber_rawId} has invalid NormalVector format.")
                continue
            chamber_norm_x, chamber_norm_y, chamber_norm_z = map(float, chamber_norm_vect)
        else:
            print(f"Warning: Chamber rawId {chamber_rawId} missing NormalVector.")
            continue

        # Extract Bounds
        chamber_bounds = chamber.find('Bounds')
        if chamber_bounds is not None:
            try:
                chamber_width = float(chamber_bounds.get('width'))
                chamber_thickness = float(chamber_bounds.get('thickness'))
                chamber_length = float(chamber_bounds.get('length'))
            except (TypeError, ValueError):
                print(f"Warning: Chamber rawId {chamber_rawId} has invalid Bounds values.")
                continue
        else:
            print(f"Warning: Chamber rawId {chamber_rawId} missing Bounds.")
            continue

        for superlayer in chamber.findall('.//SuperLayer'):
            superlayer_rawId = superlayer.get('rawId')
            superLayerNumber = superlayer.get('superLayerNumber')

            # Extract SuperLayer Positions
            super_global_pos_elem = superlayer.find('GlobalPosition')
            if super_global_pos_elem is not None and super_global_pos_elem.text:
                super_global_pos = super_global_pos_elem.text.strip('()').split(',')
                if len(super_global_pos) != 3:
                    print(f"Warning: SuperLayer rawId {superlayer_rawId} has invalid GlobalPosition format.")
                    continue
                super_global_x, super_global_y, super_global_z = map(float, super_global_pos)
            else:
                print(f"Warning: SuperLayer rawId {superlayer_rawId} missing GlobalPosition.")
                continue

            super_local_pos_elem = superlayer.find('LocalPosition')
            if super_local_pos_elem is not None and super_local_pos_elem.text:
                super_local_pos = super_local_pos_elem.text.strip('()').split(',')
                if len(super_local_pos) != 3:
                    print(f"Warning: SuperLayer rawId {superlayer_rawId} has invalid LocalPosition format.")
                    continue
                super_local_x, super_local_y, super_local_z = map(float, super_local_pos)
            else:
                print(f"Warning: SuperLayer rawId {superlayer_rawId} missing LocalPosition.")
                continue

            super_norm_vect_elem = superlayer.find('NormalVector')
            if super_norm_vect_elem is not None and super_norm_vect_elem.text:
                super_norm_vect = super_norm_vect_elem.text.strip('()').split(',')
                if len(super_norm_vect) != 3:
                    print(f"Warning: SuperLayer rawId {superlayer_rawId} has invalid NormalVector format.")
                    continue
                super_norm_x, super_norm_y, super_norm_z = map(float, super_norm_vect)
            else:
                print(f"Warning: SuperLayer rawId {superlayer_rawId} missing NormalVector.")
                continue

            # Extract SuperLayer Bounds
            super_bounds = superlayer.find('Bounds')
            if super_bounds is not None:
                try:
                    super_width = float(super_bounds.get('width'))
                    super_thickness = float(super_bounds.get('thickness'))
                    super_length = float(super_bounds.get('length'))
                except (TypeError, ValueError):
                    print(f"Warning: SuperLayer rawId {superlayer_rawId} has invalid Bounds values.")
                    continue
            else:
                print(f"Warning: SuperLayer rawId {superlayer_rawId} missing Bounds.")
                continue

            for layer in superlayer.findall('.//Layer'):
                layer_rawId = layer.get('rawId')
                layerNumber = layer.get('layerNumber')

                # Extract Layer Topology
                topology = layer.find('Topology')
                if topology is not None:
                    cellWidth_elem = topology.find('cellWidth')
                    cellHeight_elem = topology.find('cellHeight')
                    cellLength_elem = topology.find('cellLength')

                    if cellWidth_elem is not None and cellWidth_elem.text:
                        try:
                            topo_cellWidth = float(cellWidth_elem.text)
                        except ValueError:
                            print(f"Warning: Layer rawId {layer_rawId} has invalid cellWidth.")
                            continue
                    else:
                        print(f"Warning: Layer rawId {layer_rawId} missing cellWidth.")
                        continue

                    if cellHeight_elem is not None and cellHeight_elem.text:
                        try:
                            topo_cellHeight = float(cellHeight_elem.text)
                        except ValueError:
                            print(f"Warning: Layer rawId {layer_rawId} has invalid cellHeight.")
                            continue
                    else:
                        print(f"Warning: Layer rawId {layer_rawId} missing cellHeight.")
                        continue

                    if cellLength_elem is not None and cellLength_elem.text:
                        try:
                            topo_cellLength = float(cellLength_elem.text)
                        except ValueError:
                            print(f"Warning: Layer rawId {layer_rawId} has invalid cellLength.")
                            continue
                    else:
                        print(f"Warning: Layer rawId {layer_rawId} missing cellLength.")
                        continue
                else:
                    print(f"Warning: Layer rawId {layer_rawId} missing Topology.")
                    continue

                # Extract Channels
                channels = layer.find('Channels')
                if channels is not None:
                    channels_first_elem = channels.find('first')
                    channels_last_elem = channels.find('last')
                    channels_total_elem = channels.find('total')

                    if channels_first_elem is not None and channels_first_elem.text:
                        try:
                            channels_first = int(channels_first_elem.text)
                        except ValueError:
                            print(f"Warning: Layer rawId {layer_rawId} has invalid Channels.first.")
                            continue
                    else:
                        print(f"Warning: Layer rawId {layer_rawId} missing Channels.first.")
                        continue

                    if channels_last_elem is not None and channels_last_elem.text:
                        try:
                            channels_last = int(channels_last_elem.text)
                        except ValueError:
                            print(f"Warning: Layer rawId {layer_rawId} has invalid Channels.last.")
                            continue
                    else:
                        print(f"Warning: Layer rawId {layer_rawId} missing Channels.last.")
                        continue

                    if channels_total_elem is not None and channels_total_elem.text:
                        try:
                            channels_total = int(channels_total_elem.text)
                        except ValueError:
                            print(f"Warning: Layer rawId {layer_rawId} has invalid Channels.total.")
                            continue
                    else:
                        print(f"Warning: Layer rawId {layer_rawId} missing Channels.total.")
                        continue
                else:
                    print(f"Warning: Layer rawId {layer_rawId} missing Channels.")
                    continue

                # Extract Wire Positions
                wire_positions = layer.find('WirePositions')
                if wire_positions is not None:
                    first_wire_elem = wire_positions.find('FirstWire_ref_to_chamber') #Add the wire referenced to the chamber, not the layer
                    last_wire_elem = wire_positions.find('LastWire_ref_to_chamber')

                    if first_wire_elem is not None and first_wire_elem.text:
                        try:
                            first_wire = float(first_wire_elem.text)
                        except ValueError:
                            print(f"Warning: Layer rawId {layer_rawId} has invalid WirePositions.FirstWire.")
                            continue
                    else:
                        print(f"Warning: Layer rawId {layer_rawId} missing WirePositions.FirstWire.")
                        continue

                    if last_wire_elem is not None and last_wire_elem.text:
                        try:
                            last_wire = float(last_wire_elem.text)
                        except ValueError:
                            print(f"Warning: Layer rawId {layer_rawId} has invalid WirePositions.LastWire.")
                            continue
                    else:
                        print(f"Warning: Layer rawId {layer_rawId} missing WirePositions.LastWire.")
                        continue
                else:
                    print(f"Warning: Layer rawId {layer_rawId} missing WirePositions.")
                    continue

                # Extract Layer Positions
                layer_global_pos_elem = layer.find('GlobalPosition')
                if layer_global_pos_elem is not None and layer_global_pos_elem.text:
                    layer_global_pos = layer_global_pos_elem.text.strip('()').split(',')
                    if len(layer_global_pos) != 3:
                        print(f"Warning: Layer rawId {layer_rawId} has invalid GlobalPosition format.")
                        continue
                    layer_global_x, layer_global_y, layer_global_z = map(float, layer_global_pos)
                else:
                    print(f"Warning: Layer rawId {layer_rawId} missing GlobalPosition.")
                    continue

                layer_local_pos_elem = layer.find('LocalPosition')
                if layer_local_pos_elem is not None and layer_local_pos_elem.text:
                    layer_local_pos = layer_local_pos_elem.text.strip('()').split(',')
                    if len(layer_local_pos) != 3:
                        print(f"Warning: Layer rawId {layer_rawId} has invalid LocalPosition format.")
                        continue
                    layer_local_x, layer_local_y, layer_local_z = map(float, layer_local_pos)
                else:
                    print(f"Warning: Layer rawId {layer_rawId} missing LocalPosition.")
                    continue

                layer_norm_vect_elem = layer.find('NormalVector')
                if layer_norm_vect_elem is not None and layer_norm_vect_elem.text:
                    layer_norm_vect = layer_norm_vect_elem.text.strip('()').split(',')
                    if len(layer_norm_vect) != 3:
                        print(f"Warning: Layer rawId {layer_rawId} has invalid NormalVector format.")
                        continue
                    layer_norm_x, layer_norm_y, layer_norm_z = map(float, layer_norm_vect)
                else:
                    print(f"Warning: Layer rawId {layer_rawId} missing NormalVector.")
                    continue

                # Extract Layer Bounds
                layer_bounds = layer.find('Bounds')
                if layer_bounds is not None:
                    try:
                        layer_width = float(layer_bounds.get('width'))
                        layer_thickness = float(layer_bounds.get('thickness'))
                        layer_length = float(layer_bounds.get('length'))
                    except (TypeError, ValueError):
                        print(f"Warning: Layer rawId {layer_rawId} has invalid Bounds values.")
                        continue
                else:
                    print(f"Warning: Layer rawId {layer_rawId} missing Bounds.")
                    continue

                # Append to data
                try:
                    data.append({
                        'Chamber_rawId': int(chamber_rawId),
                        'Chamber_chamberId': chamber_chamberId,
                        'Chamber_Global_x': chamber_global_x,
                        'Chamber_Global_y': chamber_global_y,
                        'Chamber_Global_z': chamber_global_z,
                        'Chamber_Local_x': chamber_local_x,
                        'Chamber_Local_y': chamber_local_y,
                        'Chamber_Local_z': chamber_local_z,
                        'Chamber_Norm_x': chamber_norm_x,
                        'Chamber_Norm_y': chamber_norm_y,
                        'Chamber_Norm_z': chamber_norm_z,
                        'Chamber_Bounds_width': chamber_width,
                        'Chamber_Bounds_thickness': chamber_thickness,
                        'Chamber_Bounds_length': chamber_length,
                        'SuperLayer_rawId': int(superlayer_rawId),
                        'SuperLayerNumber': int(superLayerNumber) if superLayerNumber else None,
                        'SuperLayer_Global_x': super_global_x,
                        'SuperLayer_Global_y': super_global_y,
                        'SuperLayer_Global_z': super_global_z,
                        'SuperLayer_Local_x': super_local_x,
                        'SuperLayer_Local_y': super_local_y,
                        'SuperLayer_Local_z': super_local_z,
                        'SuperLayer_Norm_x': super_norm_x,
                        'SuperLayer_Norm_y': super_norm_y,
                        'SuperLayer_Norm_z': super_norm_z,
                        'SuperLayer_Bounds_width': super_width,
                        'SuperLayer_Bounds_thickness': super_thickness,
                        'SuperLayer_Bounds_length': super_length,
                        'Layer_rawId': int(layer_rawId),
                        'LayerNumber': int(layerNumber) if layerNumber else None,
                        'Topology_cellWidth': topo_cellWidth,
                        'Topology_cellHeight': topo_cellHeight,
                        'Topology_cellLength': topo_cellLength,
                        'Channels_first': channels_first,
                        'Channels_last': channels_last,
                        'Channels_total': channels_total,
                        'WirePositions_FirstWire': first_wire,
                        'WirePositions_LastWire': last_wire,
                        'Layer_Global_x': layer_global_x,
                        'Layer_Global_y': layer_global_y,
                        'Layer_Global_z': layer_global_z,
                        'Layer_Local_x': layer_local_x,
                        'Layer_Local_y': layer_local_y,
                        'Layer_Local_z': layer_local_z,
                        'Layer_Norm_x': layer_norm_x,
                        'Layer_Norm_y': layer_norm_y,
                        'Layer_Norm_z': layer_norm_z,
                        'Layer_Bounds_width': layer_width,
                        'Layer_Bounds_thickness': layer_thickness,
                        'Layer_Bounds_length': layer_length,
                    })
                except Exception as e:
                    print(f"Error processing Layer rawId {layer_rawId}: {e}")
                    continue

    df = pd.DataFrame(data)
    return df



# Function to compute rawId based on wheel, station, sector
def get_rawId(wheel, station, sector):
    """
    Generate the rawId based on wheel, station, sector numbers following the DTChamberId C++ class logic.

    Parameters:
    - wheel (int): Wheel number (-2 to 2)
    - station (int): Station number (1 to 4)
    - sector (int): Sector number (0 to 14)

    Returns:
    - int: Computed rawId
    """
    # Constants from C++ DetId class
    kDetMask = 0xF
    kSubdetMask = 0x7
    kDetOffset = 28
    kSubdetOffset = 25

    # Constants from C++ DTChamberId class
    minWheelId = -2
    maxWheelId = 2
    minStationId = 1
    maxStationId = 4
    minSectorId = 0
    maxSectorId = 14

    # Bit definitions for DTChamberId
    wheelMask = 0x7          # 3 bits
    wheelStartBit = 15
    stationMask = 0x7        # 3 bits
    stationStartBit = 22
    sectorMask = 0xF         # 4 bits
    sectorStartBit = 18

    # Detector and Subdetector IDs
    detector = 2  # Muon (as per DetId::Detector enum)
    subdetector = 1  # DT (Assuming DT corresponds to subdetector ID 1)

    # Validate inputs
    if not (minWheelId <= wheel <= maxWheelId):
        raise ValueError(f"Invalid wheel number: {wheel}. Must be between {minWheelId} and {maxWheelId}.")
    if not (minStationId <= station <= maxStationId):
        raise ValueError(f"Invalid station number: {station}. Must be between {minStationId} and {maxStationId}.")
    if not (minSectorId <= sector <= maxSectorId):
        raise ValueError(f"Invalid sector number: {sector}. Must be between {minSectorId} and {maxSectorId}.")

    # Compute tmpwheelid as in C++: tmpwheelid = wheel - minWheelId + 1
    tmpwheelid = wheel - minWheelId + 1

    # Compute rawId
    rawId = (
        ((detector & kDetMask) << kDetOffset) |
        ((subdetector & kSubdetMask) << kSubdetOffset) |
        ((tmpwheelid & wheelMask) << wheelStartBit) |
        ((station & stationMask) << stationStartBit) |
        ((sector & sectorMask) << sectorStartBit)
    )

    return rawId

# Function to filter DataFrame by rawId
def get_chamber_data(df, rawId):
    """
    Retrieve chamber data from DataFrame based on rawId.

    Parameters:
    - df (DataFrame): The DataFrame containing geometry data
    - rawId (int): The rawId of the Chamber

    Returns:
    - DataFrame: Filtered DataFrame containing data for the specified Chamber
    """
    chamber_df = df[df['Chamber_rawId'] == rawId]
    if chamber_df.empty:
        print(f"No data found for Chamber rawId: {rawId}")
        return None
    return chamber_df

# Function to create Chamber object from DataFrame
def create_chamber_object(chamber_df):
    """
    Create a Chamber object from the filtered DataFrame.

    Parameters:
    - chamber_df (DataFrame): DataFrame containing data for a specific Chamber

    Returns:
    - Chamber: Chamber object with SuperLayers and Layers
    """
    # Extract Chamber-level information (assuming all rows have the same Chamber data)
    chamber_rawId = chamber_df['Chamber_rawId'].iloc[0]
    chamber_chamberId = chamber_df['Chamber_chamberId'].iloc[0]
    chamber_global_pos = (
        chamber_df['Chamber_Global_x'].iloc[0],
        chamber_df['Chamber_Global_y'].iloc[0],
        chamber_df['Chamber_Global_z'].iloc[0]
    )
    chamber_local_pos = (
        chamber_df['Chamber_Local_x'].iloc[0],
        chamber_df['Chamber_Local_y'].iloc[0],
        chamber_df['Chamber_Local_z'].iloc[0]
    )
    chamber_norm_vect = (
        chamber_df['Chamber_Norm_x'].iloc[0],
        chamber_df['Chamber_Norm_y'].iloc[0],
        chamber_df['Chamber_Norm_z'].iloc[0]
    )
    chamber_bounds_width = chamber_df['Chamber_Bounds_width'].iloc[0]
    chamber_bounds_thickness = chamber_df['Chamber_Bounds_thickness'].iloc[0]
    chamber_bounds_length = chamber_df['Chamber_Bounds_length'].iloc[0]

    # Group by SuperLayer_rawId and SuperLayerNumber to handle multiple superlayers correctly
    superlayer_groups = chamber_df.groupby(['SuperLayer_rawId', 'SuperLayerNumber'])

    superlayers = []

    for (sl_rawId, sl_number), sl_group in superlayer_groups:
        # Extract SuperLayer-level information
        superLayerNumber = sl_group['SuperLayerNumber'].iloc[0]
        super_global_pos = (
            sl_group['SuperLayer_Global_x'].iloc[0],
            sl_group['SuperLayer_Global_y'].iloc[0],
            sl_group['SuperLayer_Global_z'].iloc[0]
        )
        super_local_pos = (
            sl_group['SuperLayer_Local_x'].iloc[0],
            sl_group['SuperLayer_Local_y'].iloc[0],
            sl_group['SuperLayer_Local_z'].iloc[0]
        )
        super_norm_vect = (
            sl_group['SuperLayer_Norm_x'].iloc[0],
            sl_group['SuperLayer_Norm_y'].iloc[0],
            sl_group['SuperLayer_Norm_z'].iloc[0]
        )
        super_bounds_width = sl_group['SuperLayer_Bounds_width'].iloc[0]
        super_bounds_thickness = sl_group['SuperLayer_Bounds_thickness'].iloc[0]
        super_bounds_length = sl_group['SuperLayer_Bounds_length'].iloc[0]

        # Group by Layer_rawId and LayerNumber within each SuperLayer
        layer_groups = sl_group.groupby(['Layer_rawId', 'LayerNumber'])

        layers = []

        for (layer_rawId, layer_number), layer_group in layer_groups:
            # Extract Layer-level information
            layer_cellWidth = layer_group['Topology_cellWidth'].iloc[0]
            layer_cellHeight = layer_group['Topology_cellHeight'].iloc[0]
            layer_cellLength = layer_group['Topology_cellLength'].iloc[0]
            layer_channels_first = layer_group['Channels_first'].iloc[0]
            layer_channels_last = layer_group['Channels_last'].iloc[0]
            layer_channels_total = layer_group['Channels_total'].iloc[0]
            layer_wireFirst = layer_group['WirePositions_FirstWire'].iloc[0]
            layer_wireLast = layer_group['WirePositions_LastWire'].iloc[0]
            layer_global_pos = (
                layer_group['Layer_Global_x'].iloc[0],
                layer_group['Layer_Global_y'].iloc[0],
                layer_group['Layer_Global_z'].iloc[0]
            )
            layer_local_pos = (
                layer_group['Layer_Local_x'].iloc[0],
                layer_group['Layer_Local_y'].iloc[0],
                layer_group['Layer_Local_z'].iloc[0]
            )
            layer_norm_vect = (
                layer_group['Layer_Norm_x'].iloc[0],
                layer_group['Layer_Norm_y'].iloc[0],
                layer_group['Layer_Norm_z'].iloc[0]
            )
            layer_bounds_width = layer_group['Layer_Bounds_width'].iloc[0]
            layer_bounds_thickness = layer_group['Layer_Bounds_thickness'].iloc[0]
            layer_bounds_length = layer_group['Layer_Bounds_length'].iloc[0]

            # Create Layer object
            layer = Layer(
                rawId=int(layer_rawId),
                layerNumber=int(layer_number) if layer_number else None,
                cellWidth=layer_cellWidth,
                cellHeight=layer_cellHeight,
                cellLength=layer_cellLength,
                channels_first=layer_channels_first,
                channels_last=layer_channels_last,
                channels_total=layer_channels_total,
                wireFirst=layer_wireFirst,
                wireLast=layer_wireLast,
                global_pos=layer_global_pos,
                local_pos=layer_local_pos,
                norm_vect=layer_norm_vect,
                bounds_width=layer_bounds_width,
                bounds_thickness=layer_bounds_thickness,
                bounds_length=layer_bounds_length
            )
            layers.append(layer)

        # Create SuperLayer object
        superlayer = SuperLayer(
            rawId=int(sl_rawId),
            superLayerNumber=int(sl_number) if sl_number else None,
            global_pos=super_global_pos,
            local_pos=super_local_pos,
            norm_vect=super_norm_vect,
            bounds_width=super_bounds_width,
            bounds_thickness=super_bounds_thickness,
            bounds_length=super_bounds_length,
            layers=layers
        )
        superlayers.append(superlayer)

    # Create Chamber object
    chamber = Chamber(
        rawId=int(chamber_rawId),
        chamberId=chamber_chamberId,
        global_pos=chamber_global_pos,
        local_pos=chamber_local_pos,
        norm_vect=chamber_norm_vect,
        bounds_width=chamber_bounds_width,
        bounds_thickness=chamber_bounds_thickness,
        bounds_length=chamber_bounds_length,
        superlayers=superlayers
    )

    return chamber


# Function to plot Chamber
def plot_chamber(chamber, chamber_rawId):
    """
    Plot the Chamber with its SuperLayers, Layers, and Wires.

    Parameters:
    - chamber (Chamber): Chamber object to plot
    - chamber_rawId (int): rawId of the Chamber (used for labeling)
    """
    fig, ax = plt.subplots(figsize=(15, 10))

    # Define colors for different superlayers for better visualization
    colors = ['skyblue', 'lightgreen', 'salmon', 'violet', 'orange', 'yellow']
    
    width_ch = chamber.bounds_width

    for sl_index, superlayer in enumerate(chamber.superlayers):
        color = colors[sl_index % len(colors)]  # Cycle through colors if more superlayers

        if superlayer.superLayerNumber == 2:
            # SL2 is the theta layer; draw 4 large rectangles representing layers
            for layer_index, layer in enumerate(superlayer.layers):
                wire_x = layer.wires.positions[0]  # Get the first wire position
                # Draw each wire as a rectangle
                rect = patches.Rectangle(
                    (wire_x - layer.wires.wire_width / 2, layer.get_dimension('local','z') - layer.wires.wire_height / 2),  # Center the rectangle
                    layer.bounds_length,
                    layer.wires.wire_height,
                    linewidth=0.5,
                    edgecolor='black',
                    facecolor=color,
                    alpha=0.2  # 70% transparency
                )
                ax.add_patch(rect)

                # Label the theta layer
                ax.text(
                    0,
                    layer.get_dimension('local','z'),
                    f'Theta La{layer_index+1}',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=9,
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1)
                )
        else:
            # For other SuperLayers, plot individual wires as rectangles
            for layer_index, layer in enumerate(superlayer.layers):
                for wire_x in layer.wires.positions:
                    # Draw each wire as a rectangle
                    rect = patches.Rectangle(
                        (wire_x - layer.wires.wire_width / 2, layer.get_dimension('local','z') - layer.wires.wire_height / 2),  # Center the rectangle
                        layer.wires.wire_width,
                        layer.wires.wire_height,
                        linewidth=0.5,
                        edgecolor='black',
                        facecolor=color,
                        alpha=0.7  # 70% transparency
                    )
                    ax.add_patch(rect)

                # Label the layer
                ax.text(
                    0,  # X position for label (centered)
                    layer.get_dimension('local','z'),
                    f'SL{superlayer.superLayerNumber}-La{layer_index+1}',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8,
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1)
                )

    # Setting plot limits based on the maximum wire positions and SuperLayer bounds
    ax.set_xlim(-width_ch/2, width_ch/2)
    # Determine Z limits based on layers' positions and SL2 bounds
    all_z = [layer.get_dimension('local','z') for sl in chamber.superlayers if sl.superLayerNumber != 2 for layer in sl.layers]
    sl2_z_positions = [
        (theta_layer + 1) * sl.bounds_length / (4 + 1) - sl.bounds_length / 2
        for sl in chamber.superlayers if sl.superLayerNumber == 2
        for theta_layer in range(4)
    ]
    all_z.extend(sl2_z_positions)
    min_z = min(all_z) - 10
    max_z = max(all_z) + 10
    ax.set_ylim(min_z, max_z)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Z Position (Y in plot)')
    ax.set_title(f'2D Visualization of Chamber {chamber_rawId} with SuperLayers, Layers, and Wires')
    ax.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()



# Function to compute rawId based on wheel, station, sector
def get_rawId(wheel, station, sector):
    """
    Generate the rawId based on wheel, station, sector numbers following the DTChamberId C++ class logic.

    Parameters:
    - wheel (int): Wheel number (-2 to 2)
    - station (int): Station number (1 to 4)
    - sector (int): Sector number (0 to 14)

    Returns:
    - int: Computed rawId
    """
    # Constants from C++ DetId class
    kDetMask = 0xF
    kSubdetMask = 0x7
    kDetOffset = 28
    kSubdetOffset = 25

    # Constants from C++ DTChamberId class
    minWheelId = -2
    maxWheelId = 2
    minStationId = 1
    maxStationId = 4
    minSectorId = 0
    maxSectorId = 14

    # Bit definitions for DTChamberId
    wheelMask = 0x7          # 3 bits
    wheelStartBit = 15
    stationMask = 0x7        # 3 bits
    stationStartBit = 22
    sectorMask = 0xF         # 4 bits
    sectorStartBit = 18

    # Detector and Subdetector IDs
    detector = 2  # Muon (as per DetId::Detector enum)
    subdetector = 1  # DT (Assuming DT corresponds to subdetector ID 1)

    # Validate inputs
    if not (minWheelId <= wheel <= maxWheelId):
        raise ValueError(f"Invalid wheel number: {wheel}. Must be between {minWheelId} and {maxWheelId}.")
    if not (minStationId <= station <= maxStationId):
        raise ValueError(f"Invalid station number: {station}. Must be between {minStationId} and {maxStationId}.")
    if not (minSectorId <= sector <= maxSectorId):
        raise ValueError(f"Invalid sector number: {sector}. Must be between {minSectorId} and {maxSectorId}.")

    # Compute tmpwheelid as in C++: tmpwheelid = wheel - minWheelId + 1
    tmpwheelid = wheel - minWheelId + 1

    # Compute rawId
    rawId = (
        ((detector & kDetMask) << kDetOffset) |
        ((subdetector & kSubdetMask) << kSubdetOffset) |
        ((tmpwheelid & wheelMask) << wheelStartBit) |
        ((station & stationMask) << stationStartBit) |
        ((sector & sectorMask) << sectorStartBit)
    )

    return rawId




