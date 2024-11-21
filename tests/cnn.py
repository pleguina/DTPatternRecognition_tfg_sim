import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ------------------------------
# 1. Data Preparation
# ------------------------------

from pathlib import Path
import uproot
import pandas as pd
import numpy as np

# %%
# =============================================
# 3. Definir la ruta del archivo ROOT y el nombre del árbol
# =============================================

# Ruta al archivo ROOT 
#DTDPGNtuple_12_4_2_Phase2Concentrator_Simulation_89
ruta_archivo_root = 'dtTuples/DTDPGNtuple_12_4_2_Phase2Concentrator_Simulation_89.root'

#Fijate que pongo ../* para que busque en el directorio anterior, ya que los datos están en el directorio dtTuples, el cual esta un nivel por encima de este script

# Nombre del árbol dentro del archivo ROOT 
nombre_arbol = 'dtNtupleProducer/DTTREE;1'

# %%
# =============================================
# 4. Definir las ramas a extraer
# =============================================

# Lista de ramas numéricas relacionadas con 'seg'
ramas_seg_numericas = [
    "seg_nSegments",
    "seg_wheel",
    "seg_sector",
    "seg_station",
    "seg_hasPhi",
    "seg_hasZed",
    "seg_posLoc_x",
    "seg_posLoc_y",
    "seg_posLoc_z",
    "seg_dirLoc_x",
    "seg_dirLoc_y",
    "seg_dirLoc_z",
    "seg_posLoc_x_SL1",
    "seg_posLoc_x_SL3",
    "seg_posLoc_x_midPlane",
    "seg_posGlb_phi",
    "seg_posGlb_eta",
    "seg_dirGlb_phi",
    "seg_dirGlb_eta",
    "seg_phi_t0",
    "seg_phi_vDrift",
    "seg_phi_normChi2",
    "seg_phi_nHits",
    "seg_z_normChi2",
    "seg_z_nHits"
]

# Lista completa de ramas a extraer
ramas_a_extraer = [
    "event_eventNumber",  
    "digi_nDigis", "digi_wheel", "digi_sector", "digi_station", 
    "digi_superLayer", "digi_layer", "digi_wire", "digi_time",
    *ramas_seg_numericas  # Desempaqueta las ramas de 'seg'
]

# %%
# =============================================
# 6. Función para cargar el archivo ROOT y obtener el árbol
# =============================================

def cargar_archivo_root(ruta, arbol):
    """
    Abre un archivo ROOT y obtiene el árbol especificado.
    
    Parámetros:
        ruta (str o Path): Ruta al archivo ROOT.
        arbol (str): Nombre del árbol dentro del archivo ROOT.
        
    Retorna:
        uproot.reading.ReadOnlyTree: El árbol ROOT si se encuentra, de lo contrario None.
    """
    try:
        archivo = uproot.open(ruta)
        arbol_root = archivo[arbol]
        print(f"Árbol '{arbol}' cargado exitosamente.")
        return arbol_root
    except Exception as e:
        print(f"Error al abrir el archivo ROOT o al acceder al árbol: {e}")
        return None

# %% [markdown]
# ![](imgs/2024-11-15-12-53-31.png)
# 
# COn la extension Root_file_viewer se puede ver el contenido de los archivos .root
# 
# Así pòdemos ver el nombre del arbol y de las variables que contiene
# 
# ![](imgs/2024-11-15-12-54-57.png)

# %%
# =============================================
# 7. Cargar el árbol ROOT
# =============================================

# Cargar el árbol ROOT
arbol_root = cargar_archivo_root(ruta_archivo_root, nombre_arbol)

# Verificar si el árbol se cargó correctamente
if arbol_root is None:
    raise SystemExit("No se pudo cargar el árbol ROOT. Deteniendo la ejecución.")

# %%
# =============================================
# 8. Verificar las ramas disponibles en el árbol
# =============================================

# Obtener todas las ramas disponibles en el árbol
ramas_disponibles = arbol_root.keys()
print(f"Ramas disponibles en el árbol: {ramas_disponibles}")

# Identificar las ramas faltantes
ramas_faltantes = [rama for rama in ramas_a_extraer if rama not in ramas_disponibles]
if ramas_faltantes:
    print(f"Advertencia: Las siguientes ramas no se encontraron y serán omitidas: {ramas_faltantes}")
else:
    print("Todas las ramas especificadas están disponibles en el árbol.")

# Filtrar solo las ramas que existen
ramas_existentes = [rama for rama in ramas_a_extraer if rama in ramas_disponibles]
print(f"Ramas que se extraerán: {ramas_existentes}")

# %%
# =============================================
# 9. Extraer las ramas y convertir a DataFrame
# =============================================

try:
    # Extraer las ramas en un DataFrame de pandas
    df = arbol_root.arrays(ramas_existentes, library="pd")
    print("Datos extraídos exitosamente en un DataFrame de pandas.")
except Exception as e:
    print(f"Error al extraer las ramas: {e}")
    raise SystemExit("No se pudo extraer los datos. Deteniendo la ejecución.")

# Mostrar las primeras filas del DataFrame
print("Vista previa del DataFrame:")

# Debido al tamaño, vamos a coger solo la mitad de las filas
df = df.sample(frac=0.5)


# %% [markdown]
# ![](imgs/2024-11-15-12-44-54.png)
# 
# *Podemos descargarnos una extension de vscode que permita ver los dataframes con mayor claridad, para asi poder entender mejor los datos con los que estamos trabajando*
# - **Microsoft data wrangler**
# 
# Para usarlo simplemente clicamos en el archivo que queremos ver y le damos a la opcion de "Open in data wrangler"
# 
# ![](imgs/2024-11-15-12-56-36.png)

# %%
df.head()

# %% [markdown]
# ![](imgs/2024-11-15-12-57-49.png)
# 
# Aquí podemos ver como para cada evento, tenemos un numero determinado de digis y de los segmentos que producen, que vienen ordenados por wheel, sector, station. Dentro de listas.
# 
# 138	52387	21	[2, 1, 1, 2, 2, 2, 2, -2, -1, -1, -1, 2, 2, 1, 1, 1, 1, 1, 1, -2, -2]	[3, 4, 4, 6, 8, 8, 8, 9, 3, 3, 3, 8, 8, 5, 5, 5, 5, 5, 5, 13, 13]	[1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4]	[1, 2, 2, 1, 2, 2, 2, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]	[3, 1, 2, 2, 1, 2, 3, 3, 4, 2, 3, 1, 1, 1, 1, 2, 3, 4, 4, 3, 4]	[45, 9, 10, 11, 17, 17, 16, 26, 29, 23, ..., 43, 53, 71, 71, 70, 60, 61, 10, 11]	[905, 640, 549, 604, 623, 585, 605, 708, ..., 700, 714, 740, 852, 883, 663, 852]	2	[2, 1]	[8, 5]	[1, 3]	[0, 1]	[1, 0]	[-0.5, 166]	[44.3, 0]	[0, 0]	[-1.68e-08, -0.86]	[0.837, 0]	[-0.547, -0.51]	[-999, 183]	[-999, 143]	[-999, 163]	[-2.53, 1.87]	[1.1, 0.411]	[-2.62, 3.13]	[1.21, -0.000316]	[-999, -999]	[-999, 0]	[-1, 1.43]	[0, 3]	[2.6, -1]	[3, 0]
# 
# Este es un ejemplo de una fila del df.
# 
# - 138 es el numero de fila dentro del df
# - 52387 es el numero de evento
# - 21 es el numero de digis que tiene el evento
# - [2, 1, 1, 2, 2, 2, 2, -2, -1, -1, -1, 2, 2, 1, 1, 1, 1, 1, 1, -2, -2] es la wheel a la que pertenece cada digi
# - [3, 4, 4, 6, 8, 8, 8, 9, 3, 3, 3, 8, 8, 5, 5, 5, 5, 5, 5, 13, 13] es el sector al que pertenece cada digi
# - [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4] es la estacion a la que pertenece cada digi
# etc.
# 
# esto nos da la idea de como estan estructurados los datos y como podemos acceder a ellos

# %%
# Ramas relacionadas con 'digis'
ramas_digis = [
    "event_eventNumber",
    "digi_nDigis", "digi_wheel", "digi_sector", "digi_station", 
    "digi_superLayer", "digi_layer", "digi_wire", "digi_time"
]

# Ramas relacionadas con 'segments' (seg)
ramas_segments = [
    "event_eventNumber",
    "seg_nSegments",
    "seg_wheel",
    "seg_sector",
    "seg_station",
    "seg_hasPhi",
    "seg_hasZed",
    "seg_posLoc_x",
    "seg_posLoc_y",
    "seg_posLoc_z",
    "seg_dirLoc_x",
    "seg_dirLoc_y",
    "seg_dirLoc_z",
    "seg_posLoc_x_SL1",
    "seg_posLoc_x_SL3",
    "seg_posLoc_x_midPlane",
    "seg_posGlb_phi",
    "seg_posGlb_eta",
    "seg_dirGlb_phi",
    "seg_dirGlb_eta",
    "seg_phi_t0",
    "seg_phi_vDrift",
    "seg_phi_normChi2",
    "seg_phi_nHits",
    "seg_z_normChi2",
    "seg_z_nHits"
]

# Combinar todas las ramas a extraer
ramas_a_extraer = ramas_digis + ramas_segments


# %%

# Extraer las ramas relacionadas con 'digis' en un DataFrame de pandas
df_digis = arbol_root.arrays(ramas_digis, library="pd")

# Extraer las ramas relacionadas con 'segments' en otro DataFrame de pandas
df_segments = arbol_root.arrays(ramas_segments, library="pd")

# Mostrar una vista previa de los DataFrames
print("Vista previa del DataFrame de 'digis':")

print("\nVista previa del DataFrame de 'segments':")


# %% [markdown]
# Las columnas extraídas pueden contener Awkward Arrays, que son estructuras de datos que permiten listas de diferentes longitudes en cada fila. Para manipular estos datos con pandas de manera efectiva, es recomendable convertirlos a listas de Python estándar. 
# 
# convertir_a_lista: Verifica si el elemento es una lista, tupla o arreglo de numpy. Si lo es, lo convierte en una lista de Python. Si es un valor escalar, lo envuelve en una lista para mantener la consistencia.
# Aplicación: Se aplica esta función a todas las columnas de ambos DataFrames para asegurar que todas las entradas sean listas de Python.

# %%
# Función para convertir Awkward Arrays a listas de Python
def convertir_a_lista(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x]  # En caso de que sea un valor escalar, lo convertimos en lista

# Aplicar la conversión a todas las columnas de 'digis' que son listas ( excepto 'event_eventNumber' )  

for col in ramas_digis:
        df_digis[col] = df_digis[col].apply(convertir_a_lista)

# Aplicar la conversión a todas las columnas de 'segments'
for col in ramas_segments:
    df_segments[col] = df_segments[col].apply(convertir_a_lista)

df_digis.head()


# %% [markdown]
# Al aplanar correctamente todas las columnas relevantes en los DataFrames de digis y segments, aseguras que cada fila represente una única instancia (digi o segment) con atributos escalares, lo cual es esencial para un análisis preciso y la construcción de modelos de aprendizaje automático efectivos. Primero nos aseguramos de que todas las columnas relevantes sean listas de Python, y luego comprobamos que todas las listas tengan la misma longitud. Si no es así, se rellenan con valores nulos para mantener la consistencia.

# %% [markdown]
# 

# %%
# Definir las columnas relacionadas con 'digis' que serán aplanadas
columns_to_explode_digis = ['digi_superLayer', 'digi_layer', 'digi_wire', 'digi_time', 'digi_wheel', 'digi_sector', 'digi_station']

# Función para verificar si todas las listas tienen la misma longitud en una fila
def verificar_longitudes(row, cols):
    lengths = [len(row[col]) for col in cols]
    return len(set(lengths)) == 1  # True si todas las longitudes son iguales

# Aplicar la función a cada fila
df_digis['same_length'] = df_digis.apply(lambda row: verificar_longitudes(row, columns_to_explode_digis), axis=1)

# Verificar cuántas filas cumplen la condición
print("Número de filas con listas de la misma longitud en 'digis':")
print(df_digis['same_length'].value_counts())

# Filtrar solo las filas que cumplen la condición
df_digis = df_digis[df_digis['same_length']]

# Eliminar la columna auxiliar
df_digis = df_digis.drop(columns=['same_length'])

# Verificar nuevamente
print("\nDespués de filtrar, número de filas restantes en 'digis':")
print(len(df_digis))
df_digis.head()


# %%
# ahora vamos a aplanar las columnas de 'digis' que contienen listas
columns_to_explode_digis = [
    'digi_superLayer', 'digi_layer', 'digi_wire', 
    'digi_time', 'digi_wheel', 'digi_sector', 'digi_station'
]


df_digis_exploded = df_digis.explode(columns_to_explode_digis)

# Renombrar las columnas a nombres más cortos
df_digis_exploded = df_digis_exploded.rename(columns={
    "event_eventNumber": "eventNumber",
    "digi_superLayer": "superLayer",
    "digi_layer": "layer",
    "digi_wire": "wire",
    "digi_time": "time",
    "digi_wheel": "wheel",
    "digi_sector": "sector",
    "digi_station": "station"
})

# Convertir 'event_eventNumber' a entero, ya que antes era una lista
df_digis_exploded['eventNumber'] = df_digis_exploded['eventNumber'].apply(lambda x: x if isinstance(x, (int, str)) else x[0])

# Eliminar la columna 'digi_nDigis' ya que no es necesaria, puesto que cuenta el numero de digis en cada evento, no de cada estación
df_digis_exploded = df_digis_exploded.drop(columns=['digi_nDigis'])
# ELiminamos los digis de la superLayer 2, ya que no son necesarios y pueden influir en los resultados
df_digis_exploded = df_digis_exploded[df_digis_exploded['superLayer'] != 2]
print("Preview of 'df_digis_exploded':")

# %%
# Ahora vamos a agrupar los digis por evento y estación
df_digis_grouped = df_digis_exploded.groupby(
    ['eventNumber', 'wheel', 'sector', 'station']
).agg({
    'superLayer': list,
    'layer': list,
    'wire': list,
    'time': list
}).reset_index()

df_digis_grouped['n_digis'] = df_digis_grouped['wire'].apply(len)

# Mostrar una vista previa después de aplanar
print("Vista previa del DataFrame de 'digis' después de aplanar:")
df_digis_grouped.head()
tamaño_bytes = df_digis_grouped.memory_usage(deep=True).sum()
print(f"Tamaño del DataFrame en bytes: {tamaño_bytes}")
df_digis_grouped.head()

# %%
# Eliminar columnas de segmentos con las que no vamos a trabajar (todas menos 'event_eventNumber', 'seg_wheel', 'seg_sector', 'seg_station')
#Esto es porque de momento queremos solo predecir si hay o no segmentos en una estación, no cuantos ni su posición.


columnas_a_eliminar = [col for col in df_segments.columns if col not in ['event_eventNumber', 'seg_wheel', 'seg_sector', 'seg_station', 'seg_hasPhi']]

df_segments_filtered = df_segments.drop(columns=columnas_a_eliminar)

# Nos aseguramos de que 'event_eventNumber' sea un escalar, ya que antes convertimos todas las columnas a listas
df_segments_filtered['event_eventNumber'] = df_segments['event_eventNumber'].str[0]

# Explotamos para crear una fila por estación y ver si es un segmento en phi (SL1 + SL3)
df_exploded = df_segments_filtered.explode(['seg_wheel', 'seg_sector', 'seg_station', 'seg_hasPhi'])
df_segments_phi = df_exploded[df_exploded['seg_hasPhi'] == 1]

# Agrupamos y añadir una columna con el número de segmentos. Esta columna calcula el número de segmentos por evento, estación, sector y rueda. 
# Esto lo hace contando el número de filas que comparten el mismo conjunto de valores en las columnas 'event_eventNumber', 'seg_wheel', 'seg_sector' y 'seg_station'.
df_counts = (
    df_segments_phi
    .groupby(['event_eventNumber', 'seg_wheel', 'seg_sector', 'seg_station'])
    .size()
    .reset_index(name='n_segments')
)

#Renombramos las columnas para mayor claridad

df_counts = df_counts.rename(columns={
    'event_eventNumber': 'eventNumber',
    'seg_wheel': 'wheel',
    'seg_sector': 'sector',
    'seg_station': 'station'
})


# The resulting DataFrame
print(df_counts)

# %%
# Hacemos lo mismo que con los digis, agrupamos los segmentos por evento y estación
df_merged = pd.merge(
    df_digis_grouped,
    df_counts,
    how='left',
    on=['eventNumber', 'wheel', 'sector', 'station']
)

# Fill NaN values in 'n_segments' with 0 (indicating no segments)
df_merged['n_segments'] = df_merged['n_segments'].fillna(0).astype(int)

# Define classification target: 1 if at least one segment, else 0
df_merged['has_segment'] = (df_merged['n_segments'] > 0).astype(int)

# Display a preview of the merged DataFrame
print("Preview of 'df_merged':")
df_merged.head()

df_filtered_segments = df_merged[df_merged['n_segments'] <= 15].reset_index(drop=True)
# Ensure all labels are within the range [0, 15]
assert df_filtered_segments['n_segments'].max() <= 15, "Labels exceed the maximum class range!"
assert df_filtered_segments['n_segments'].min() >= 0, "Labels are below the minimum class range!"

# ------------------------------
# 2. Filtering and Exploding the DataFrame
# ------------------------------

# Keep only superlayers 1 and 3
df_filtered = df_filtered_segments[df_filtered_segments['superLayer'].apply(lambda x: all(sl in [1, 3] for sl in x))].reset_index(drop=True)

# Explode the list-like columns to have one row per digi
df_exploded = df_filtered.explode(['superLayer', 'layer', 'wire', 'time']).reset_index(drop=True)

# Display the exploded DataFrame
print("\nExploded DataFrame:")
print(df_exploded)

# ------------------------------
# 3. Grid Mapping
# ------------------------------

# Detector Parameters
globalDTheight = 1.3  # cm (height per layer)
layers_per_superlayer = 4
number_of_superlayers = 2  # Superlayer 1 and 3
total_layers_height = globalDTheight * layers_per_superlayer * number_of_superlayers
SLgap = 28.7 - total_layers_height  # cm
gap_rows = int(round(SLgap / globalDTheight))  # Number of grid rows representing the gap

print(f"\nTotal Layers Height: {total_layers_height} cm")
print(f"Gap Height (SLgap): {SLgap} cm")
print(f"Number of Gap Rows: {gap_rows}")

# Mapping from Superlayer number to index
superlayer_to_index = {1: 0, 3: 1}

# Function to determine grid dimensions based on all events
def get_grid_dimensions(df):
    """Determine grid width based on maximum wire number across all events."""
    max_wire = df['wire'].max()
    grid_width = max_wire * 2  # Times 2 for staggered cells
    return grid_width

grid_width = get_grid_dimensions(df_exploded)
grid_height = (layers_per_superlayer * number_of_superlayers) + gap_rows  # e.g., 4*2 + 15 = 23

print(f"\nGrid Dimensions: Height = {grid_height}, Width = {grid_width}")

# Function to map superlayer and layer to grid row
def get_grid_row(superlayer, layer):
    superlayer_idx = superlayer_to_index.get(superlayer)
    if superlayer_idx is None:
        return None  # Invalid superlayer
    layer_idx = layer - 1  # Zero-based indexing
    if superlayer_idx == 0:
        # Superlayer 1 (bottom)
        row = layer_idx
    elif superlayer_idx == 1:
        # Superlayer 3 (top)
        row = layers_per_superlayer + gap_rows + layer_idx
    return row

# Function to map wire and layer to grid column
def get_grid_col(wire, layer):
    wire_idx = wire - 1  # Adjust if wire numbering starts at 1
    layer_idx = layer - 1  # Zero-based indexing
    if layer_idx % 2 == 0:
        # Even layers: cells start at even columns
        col = wire_idx * 2
    else:
        # Odd layers: cells start at odd columns (staggered)
        col = wire_idx * 2 + 1
    return col

# Initialize Grids and Labels
# Each grid will have 2 channels:
# - Channel 0: Hit presence (binary)
# - Channel 1: Hit count (integer)
grids = []
labels = []

# Group the exploded DataFrame by event
grouped = df_exploded.groupby('eventNumber')

for event_id, group in grouped:
    # Initialize a grid for the event
    grid = np.zeros((grid_height, grid_width, 2), dtype=np.float32)
    
    # Iterate over each digi in the event
    for _, hit in group.iterrows():
        superlayer = hit['superLayer']
        layer = hit['layer']
        wire = hit['wire']
        # time = hit['time']  # Not used in grid, but can be added as another channel if needed
    
        row = get_grid_row(superlayer, layer)
        if row is None:
            print(f"Invalid superlayer {superlayer} in event {event_id}; skipping hit.")
            continue
    
        col = get_grid_col(wire, layer)
    
        if 0 <= row < grid_height and 0 <= col < grid_width:
            grid[row, col, 0] = 1  # Hit presence
            grid[row, col, 1] += 1  # Hit count
        else:
            print(f"Hit out of bounds: Event {event_id}, SL{superlayer} L{layer} W{wire}")
    
    grids.append(grid)
    # Extract label for the event
    label = group['n_segments'].iloc[0]
    labels.append(label)

# Convert lists to numpy arrays
grids = np.array(grids)  # Shape: (num_events, grid_height, grid_width, 2)
labels = np.array(labels)  # Shape: (num_events,)

print(f"\nNumber of Events: {grids.shape[0]}")
print(f"Grid Shape: {grids.shape}")

# ------------------------------
# 4. Dataset and DataLoader
# ------------------------------

class DetectorDataset(Dataset):
    def __init__(self, grids, labels, transform=None):
        """
        Args:
            grids (numpy array): Array of detector grids.
            labels (numpy array): Labels for the number of segments.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.grids = grids
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        grid = self.grids[idx]  # Shape: (grid_height, grid_width, 2)
        label = self.labels[idx]  # Integer label

        # Convert to torch tensor and rearrange dimensions
        grid = torch.from_numpy(grid).permute(2, 0, 1)  # Shape: (2, grid_height, grid_width)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            grid = self.transform(grid)

        return grid, label

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    grids, labels, test_size=0.2, random_state=42
)

# Create Dataset instances
train_dataset = DetectorDataset(X_train, y_train)
val_dataset = DetectorDataset(X_val, y_val)

# Create DataLoader instances
batch_size = 2  # Adjust based on your dataset size and GPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ------------------------------
# 5. CNN Model Definition
# ------------------------------

class DetectorCNN(nn.Module):
    def __init__(self, num_channels, num_segments_classes):
        super(DetectorCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)  # Reduces each dimension by a factor of 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2)
        self.flatten = nn.Flatten()
        # Calculate the input size for the first fully connected layer
        self.fc1 = nn.Linear(self._get_conv_output_shape(), 128)
        self.fc2 = nn.Linear(128, num_segments_classes)
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_output_shape(self):
        # Create a dummy input to calculate the output shape after conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_channels, grid_height, grid_width)
            x = self.conv1(dummy_input)
            x = self.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = self.relu(x)
            x = self.flatten(x)
            output_shape = x.shape[1]
        return output_shape

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)

        logits = self.fc2(x)
        output = self.softmax(logits)

        return output

# Define the number of segment classes
num_segments_classes = 15  # e.g., 1, 2, 3...

# Define the number of channels in the grid (hit presence and hit count)
num_channels = grids.shape[3]  # 2

# Instantiate the model
model = DetectorCNN(num_channels=num_channels, num_segments_classes=num_segments_classes)

# ------------------------------
# 6. Training Setup
# ------------------------------

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------------
# 7. Training Loop
# ------------------------------

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for grids_batch, labels_batch in train_loader:
            grids_batch = grids_batch.to(device)
            labels_batch = labels_batch.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(grids_batch)
            loss = criterion(outputs, labels_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for grids_batch, labels_batch in val_loader:
                grids_batch = grids_batch.to(device)
                labels_batch = labels_batch.to(device)

                outputs = model(grids_batch)
                loss = criterion(outputs, labels_batch)

                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels_batch.size(0)
                val_correct += (predicted == labels_batch).sum().item()

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = 100 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f} "
              f"Train Acc: {epoch_acc:.2f}% "
              f"Val Loss: {val_epoch_loss:.4f} "
              f"Val Acc: {val_epoch_acc:.2f}%")

# Train the model
num_epochs = 20
train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs)

# ------------------------------
# 8. Evaluation
# ------------------------------

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for grids_batch, labels_batch in data_loader:
            grids_batch = grids_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(grids_batch)
            loss = criterion(outputs, labels_batch)

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    print(f"Evaluation - Loss: {avg_loss:.4f} Accuracy: {accuracy:.2f}%")

# Evaluate on validation set
evaluate_model(model, val_loader, criterion, device)

# ------------------------------
# 9. Visualization
# ------------------------------

def visualize_grid(grid, title="Detector Grid"):
    """
    Visualize a single detector grid.
    Args:
        grid (numpy array): Grid with shape (grid_height, grid_width, 2)
        title (str): Title of the plot
    """
    plt.figure(figsize=(10, 6))
    # Combine channels for visualization: presence and count
    presence = grid[:, :, 0]
    count = grid[:, :, 1]
    
    # Create a combined image
    combined = presence + count * 0.1  # Adjust scaling as needed
    
    plt.imshow(combined, cmap='viridis', interpolation='none', aspect='auto')
    plt.xlabel('Horizontal Position (Cells)')
    plt.ylabel('Layer (from bottom to top)')
    plt.title(title)
    plt.colorbar(label='Hit Presence + Count')
    plt.show()

# Visualize the first training grid
sample_grid = X_train[0]  # Shape: (grid_height, grid_width, 2)
visualize_grid(sample_grid, title="Sample Detector Grid - Event 1")

# ------------------------------
# 10. Saving the Model
# ------------------------------

# Save the trained model
torch.save(model.state_dict(), 'detector_cnn.pth')
print("\nModel saved to detector_cnn.pth")

# ------------------------------
# 11. Loading the Model (Optional)
# ------------------------------

# To load the model later:
# model = DetectorCNN(num_channels=num_channels, num_segments_classes=num_segments_classes)
# model.load_state_dict(torch.load('detector_cnn.pth'))
# model.to(device)
# model.eval()

# ------------------------------
# End of Script
# ------------------------------
