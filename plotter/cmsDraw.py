import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import numpy as np

def draw_cms_muon_chambers(wheel, sector, station):
    """
    Draws the RZ and Phi views of the CMS muon chambers,
    highlighting a specific station given by wheel, sector, and station.
    """
    # Validate inputs
    if wheel not in [-2, -1, 0, 1, 2]:
        raise ValueError("Wheel must be between -2 and +2.")
    if sector not in range(1, 13):
        raise ValueError("Sector must be between 1 and 12.")
    if station not in [1, 2, 3, 4]:
        raise ValueError("Station must be between 1 and 4.")

    # Adjust Z-axis based on wheel sign
    z_sign = -1 if wheel < 0 else 1

    # Define constants for wheel sizes and gaps
    wheel_size = 3.0  # Size of wheels (except Wheel 0)
    wheel0_size = 1.5  # Size of Wheel 0 (halved)
    gap_z = 0.5        # Gap between wheels along Z-axis
    gap_r = 0.05        # Gap between DT chambers and iron yokes in R
    iron_thickness = 0.5  # Thickness of iron yokes

    # Calculate Z positions for wheels
    # Starting from Z = 0
    wheel_positions = {}
    current_z = 0.0

    # Wheels to draw based on the wheel's sign
    if wheel >= 0:
        wheels_to_draw = [0, 1, 2]
    else:
        wheels_to_draw = [0, -1, -2]

    # Adjust Z positions
    for w in wheels_to_draw:
        if w == 0:
            z_start = current_z
            z_end = z_start + wheel0_size * z_sign
            current_z = z_end + gap_z * z_sign
        else:
            z_start = current_z
            z_end = z_start + wheel_size * z_sign
            current_z = z_end + gap_z * z_sign
        wheel_positions[w] = {'Z_start': z_start, 'Z_end': z_end}

    # Define radial positions for DT stations and iron yokes
    # Gaps between DT stations and iron yokes
    dt_stations = [
        {'name': 'MB1', 'R_start': 4.0, 'R_end': 4.5, 'station_num': 1},
        {'name': 'MB2', 'R_start': 5.0, 'R_end': 5.5, 'station_num': 2},
        {'name': 'MB3', 'R_start': 6.0, 'R_end': 6.5, 'station_num': 3},
        {'name': 'MB4', 'R_start': 7.0, 'R_end': 7.5, 'station_num': 4},
    ]

    iron_yokes = [
        {'R_start': 4.5 + gap_r, 'R_end': 5.0 - gap_r},
        {'R_start': 5.5 + gap_r, 'R_end': 6.0 - gap_r},
        {'R_start': 6.5 + gap_r, 'R_end': 7.0 - gap_r},
    ]

    # Create RZ view plot
    fig_rz, ax_rz = plt.subplots(figsize=(10, 6))
    ax_rz.set_xlabel('Z (meters)')
    ax_rz.set_ylabel('R (meters)')
    ax_rz.set_title('RZ View of CMS Muon Chambers (DT and Iron Yokes)')

    # Plot DT stations
    for idx, station_info in enumerate(dt_stations):
        station_num = station_info['station_num']
        for w in wheels_to_draw:
            wheel_info = wheel_positions[w]
            z_start = wheel_info['Z_start']
            z_end = wheel_info['Z_end']

            # Highlight the specified station
            highlight = (w == wheel and station_num == station)

            # DT chamber rectangle
            dt_rect = patches.Rectangle(
                (z_start, station_info['R_start']),
                z_end - z_start,
                station_info['R_end'] - station_info['R_start'],
                linewidth=1,
                edgecolor='blue' if not highlight else 'yellow',
                facecolor='lightblue' if not highlight else 'yellow',
                label='DT Chamber' if (idx == 0 and w == wheels_to_draw[0]) else ""
            )
            ax_rz.add_patch(dt_rect)

            # Label the station
            ax_rz.text(
                (z_start + z_end) / 2,
                (station_info['R_start'] + station_info['R_end']) / 2,
                f"MB{station_num}\nWheel {w}",
                color='black' if not highlight else 'white',
                fontsize=8,
                ha='center',
                va='center'
            )

    # Plot iron yokes between stations
    for idx, iron_info in enumerate(iron_yokes):
        for w in wheels_to_draw:
            wheel_info = wheel_positions[w]
            z_start = wheel_info['Z_start']
            z_end = wheel_info['Z_end']

            # Iron yoke rectangle
            iron_rect = patches.Rectangle(
                (z_start, iron_info['R_start']),
                z_end - z_start,
                iron_info['R_end'] - iron_info['R_start'],
                linewidth=1,
                edgecolor='grey',
                facecolor='lightgrey',
                label='Iron Yoke' if (idx == 0 and w == wheels_to_draw[0]) else ""
            )
            ax_rz.add_patch(iron_rect)

    # Display the wheel number, sector, and station in the plot area
    info_text = f"Highlighted:\nWheel {wheel}, Sector {sector}, Station MB{station}"
    bbox_props = dict(boxstyle="round,pad=0.5", fc="yellow", ec="black", lw=1)
    ax_rz.text(
        0.98, 0.2, info_text,
        transform=ax_rz.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=bbox_props
    )

    # Set axis limits to cover the entire quarter
    max_z = 12 * z_sign
    ax_rz.set_xlim(0, max_z)
    ax_rz.set_ylim(0, 8.0)  # Adjusted R limits to include gaps

    # Add a legend
    handles, labels = ax_rz.get_legend_handles_labels()
    ax_rz.legend(handles, labels, loc='upper right')

    # Add grid lines
    ax_rz.grid(True, which='both', linestyle='--', linewidth=0.5)

   # Create Phi view plot
    fig_phi, ax_phi = plt.subplots(figsize=(6, 6))
    ax_phi.set_xlabel('X (meters)')
    ax_phi.set_ylabel('Y (meters)')
    ax_phi.set_title('Phi View of CMS Muon Chambers (DT Sectors and Stations)')

    # Draw the sectors (12 sectors)
    sectors = 12
    sector_angles = np.linspace(0, 360, sectors + 1)
    r_inner = 3.0  # Inner radius (beam pipe outer edge)
    station_width = 0.5  # Radial thickness of each station

    for i in range(sectors):
        sector_num = i + 1
        theta1 = sector_angles[i]
        theta2 = sector_angles[i + 1]

        for j, station_info in enumerate(dt_stations):
            station_num = station_info['station_num']
            radius = station_info['R_start'] - station_width / 2

            # Draw rectangle (approximated as a wedge for perspective)
            wedge = patches.Wedge(
                center=(0, 0),
                r=radius + station_width,
                theta1=theta1,
                theta2=theta2,
                width=station_width,
                facecolor='lightblue' if not (sector_num == sector and station_num == station) else 'yellow',
                edgecolor='blue',
                linewidth=1,
                label='Station' if (i == 0 and j == 0) else ""
            )
            ax_phi.add_patch(wedge)

            # Label the sector and station
            angle = np.radians((theta1 + theta2) / 2)
            ax_phi.text(
                (radius + station_width / 2) * np.cos(angle),
                (radius + station_width / 2) * np.sin(angle),
                f"S{sector_num}\nMB{station_num}",
                color='black',
                fontsize=6,
                ha='center',
                va='center'
            )

    # Draw the inner circle (beam pipe)
    inner_circle = patches.Circle(
        (0, 0),
        r_inner,
        color='grey',
        alpha=0.3,
        label='stuff'
    )
    ax_phi.add_patch(inner_circle)

    # Set aspect ratio and limits
    ax_phi.set_aspect('equal', 'box')
    ax_phi.set_xlim(-8, 8)
    ax_phi.set_ylim(-8, 8)

    # Add a legend
    handles, labels = ax_phi.get_legend_handles_labels()
    ax_phi.legend(handles, labels, loc='upper right')

    # Add grid lines
    ax_phi.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Show both plots
    plt.show()


def drawCMS_RZ():


    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set axis labels
    ax.set_xlabel('Z (meters)')
    ax.set_ylabel('R (meters)')
    ax.set_title('RZ View of CMS Muon Chambers (DT and CSC)')

    # Define DT stations (MB1 to MB4) in the barrel region
    # MB1 to MB4 at increasing radial positions from R = 4m to R = 7m
    dt_stations = [
        {'name': 'MB1', 'R': 4.0, 'Z_start': 0, 'Z_end': 7.0},
        {'name': 'MB2', 'R': 5.0, 'Z_start': 0, 'Z_end': 7.0},
        {'name': 'MB3', 'R': 6.0, 'Z_start': 0, 'Z_end': 7.0},
        {'name': 'MB4', 'R': 7.0, 'Z_start': 0, 'Z_end': 7.0},
    ]

    # Define CSC stations (ME1 to ME4) in the endcap region
    csc_stations = [
        {'name': 'ME1', 'R_inner': 1.5, 'R_outer': 2.5, 'Z_start': 7.0, 'Z_end': 8.0},
        {'name': 'ME2', 'R_inner': 2.0, 'R_outer': 3.5, 'Z_start': 8.0, 'Z_end': 9.0},
        {'name': 'ME3', 'R_inner': 2.5, 'R_outer': 4.0, 'Z_start': 9.0, 'Z_end': 10.0},
        {'name': 'ME4', 'R_inner': 3.0, 'R_outer': 4.5, 'Z_start': 10.0, 'Z_end': 10.7},
    ]

    # Plot DT stations
    for dt in dt_stations:
        rect = patches.Rectangle(
            (dt['Z_start'], dt['R'] - 0.2),  # (x,y) position
            dt['Z_end'] - dt['Z_start'],     # width
            0.4,                             # height
            linewidth=1,
            edgecolor='blue',
            facecolor='lightblue',
            label='DT' if dt['name'] == 'MB1' else ""
        )
        ax.add_patch(rect)
        # Label the station
        ax.text(
            (dt['Z_start'] + dt['Z_end']) / 2,
            dt['R'],
            dt['name'],
            color='blue',
            fontsize=9,
            ha='center',
            va='center'
        )

    # Plot CSC stations
    for csc in csc_stations:
        # Create a polygon for each CSC station
        polygon = patches.Polygon(
            [
                (csc['Z_start'], csc['R_inner']),
                (csc['Z_start'], csc['R_outer']),
                (csc['Z_end'], csc['R_outer']),
                (csc['Z_end'], csc['R_inner']),
            ],
            closed=True,
            linewidth=1,
            edgecolor='red',
            facecolor='salmon',
            label='CSC' if csc['name'] == 'ME1' else ""
        )
        ax.add_patch(polygon)
        # Label the station
        ax.text(
            (csc['Z_start'] + csc['Z_end']) / 2,
            (csc['R_inner'] + csc['R_outer']) / 2,
            csc['name'],
            color='red',
            fontsize=9,
            ha='center',
            va='center'
        )

    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right')

    # Set axis limits
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)

    # Add grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Show the plot
    plt.show()
    
def drawCoordinates_test():
    

    # Define chamber 3D positions (X, Y, Z) and labels
    """
    Plots the chamber positions and a segment with given global phi/eta coordinates.

    Parameters:
    posGlb_phi (float): Global position phi in radians.
    posGlb_eta (float): Global position eta (pseudorapidity).
    dirGlb_phi (float): Global direction phi in radians.
    dirGlb_eta (float): Global direction eta (pseudorapidity).
    r (float): Radial distance (same as z-coordinate of the chamber).

    Returns:
    None
    """
    
    posGlb_phi=1.4642149209976196, 
    posGlb_eta=0.7911032438278198, 
    dirGlb_phi=1.4652460813522339, 
    dirGlb_eta=0.6734017133712769, 
    r=533.35
    # Define chamber 3D positions (X, Y, Z) and labels
    positions_3d = [
        (720.2, -94.895, 533.35, 'Wh:2 St:4 Se:1'),
        (671.159, 277.919, 533.35, 'Wh:2 St:4 Se:2'),
        (442.281, 576.264, 533.35, 'Wh:2 St:4 Se:3'),
        (160.75, 720.2, 533.35, 'Wh:2 St:4 Se:4'),
        (-160.75, 720.2, 533.35, 'Wh:2 St:4 Se:13'),
        (-442.281, 576.264, 533.35, 'Wh:2 St:4 Se:5'),
        (-671.159, 277.919, 533.35, 'Wh:2 St:4 Se:6'),
        (-720.2, -94.895, 533.35, 'Wh:2 St:4 Se:7'),
        (-580.461, -435.011, 533.35, 'Wh:2 St:4 Se:8'),
        (-356.143, -628.824, 533.35, 'Wh:2 St:4 Se:9'),
        (-136.77, -720.2, 533.35, 'Wh:2 St:4 Se:10'),
        (136.77, -720.2, 533.35, 'Wh:2 St:4 Se:14'),
        (356.143, -628.824, 533.35, 'Wh:2 St:4 Se:11'),
        (580.461, -435.011, 533.35, 'Wh:2 St:4 Se:12')
    ]

    # Convert from phi/eta to Cartesian coordinates for position
    segment_pos_x = r * np.cos(posGlb_phi)
    print(segment_pos_x)    
    segment_pos_y = r * np.sin(posGlb_phi)
    print(segment_pos_y)
    segment_pos_z = r * np.sinh(posGlb_eta)
    print(segment_pos_z)
    segment_pos = (segment_pos_x, segment_pos_y, segment_pos_z)
    print("--------------------")
    # Convert from phi/eta to Cartesian coordinates for direction vector (assuming unit direction)
    segment_dir_x = np.cos(dirGlb_phi)
    print(segment_dir_x)
    segment_dir_y = np.sin(dirGlb_phi)
    print(segment_dir_y)
    segment_dir_z = np.sinh(dirGlb_eta)
    print(segment_dir_z)
    segment_dir = (segment_dir_x, segment_dir_y, segment_dir_z)

    # Print the segment information
    print("Segment:")
    print("  Global Position (x, y, z):", segment_pos)
    print("  Global Direction (dir_x, dir_y, dir_z):", segment_dir)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the chambers
    for x, y, z, label in positions_3d:
        ax.scatter(x, y, z, color='blue')
        ax.text(x, y, z, label, fontsize=9, ha='right')

    # Plot the segment position
    ax.scatter(*segment_pos, color='red', label='Segment Position')

    # Plot the direction vector of the segment
    ax.quiver(segment_pos[0], segment_pos[1], segment_pos[2], 
              segment_dir[0], segment_dir[1], segment_dir[2], 
              color='green', length=100, normalize=True, label='Segment Direction')

    # Set axis labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('3D Chamber Positions and Segment')

    plt.legend()
    plt.show()