# Pattern recognition for DTs
This repo contains a set of tools that can be used to implement pattern recognition algorithms
considering the geometrical features of the CMS DT system.

# Usage
The user has to provide a python file with the implementation of the pattern recognition
algorithm. The geometry of the DTs can be accessed by just importing the following libraries:

```
from geometry.MBstation import MBstation
from geometry.Layer import Layer
from geometry.DriftCell import DriftCell
from geometry.CMSDT import CMSDT

chambers = CMSDT(wheel, sector, station)
```

The `CMSDT` function defined in `geometry/CMSDT.py` takes the value for the wheel, the sector and the station (**as integers numbers**) and returns a `MBstation` object. 
An starting point of a simple simulator can be found in `testRun.py`.

## How to play with the geometry objects
The `MBstation` contains a list of 8 `Layer` objects. Each `Layer` object contains a certain amount of DriftCell objects, which simulate the different drift cells of a real DT. All the MBstations, Layers and DriftCell objects are created at runtime when the CMSDT function is called. 

Here's a small list of things someone can do in order to work with the geometry.

 * **Get all the 7 layers of an MBstation simultaneously**: 
   * `MBstation.get_layers()`
   * This method returns a list of Layer objects.
 * **Get one specific layer of an MBstation **: 
   * `MBstation.get_layer(layer_id)`, with layer_id between 0 and 7. 
   * This method returns a `Layer` object.
 * **Get all the cells from a layer **: 
   * `MBstation.get_layer(layer_id).get_cells()`.
   * This method returns the complete list of cells inside a `Layer` object. Each element of the
     list is a DriftCell object.
 * **Get one specific cell from a layer**:
   * `MBstation.get_layer(layer_id).get_cell(cell_id)`, with cell_id between 1 and nDriftCells+1.
   * The number of DriftCells is specified within `geometry/CMSDT.py`.
   * This method returns a `DriftCell` object.

 * **How to access the local position of a DriftCell**
   * `DriftCell.get_position_at_min()`: returns the position of the lower-leftmost corner of the cell
   * `DriftCell.get_center()`: returns the center position of the cell.
   * DriftCell positions are defined in the frame of reference that sits in the lower-leftmost corner of the first cell in the first layer of SL1 (SL1-L1-Wire0). 

   
    




