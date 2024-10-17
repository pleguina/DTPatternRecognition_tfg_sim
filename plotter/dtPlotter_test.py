from dtPlotter import *

import sys
import os

# Add the base directory (one level up from tests) to the system path
# This allows the test to access the modules in the base directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Main function to execute the workflow
def main():
    # Path to the XML file
    xml_file = 'plotter/DTGeometry.xml'  # Update with the correct path if necessary

    # Parse the XML and create DataFrame
    df = parse_dtgeometry_xml(xml_file)
    print("XML data has been parsed and DataFrame created.")
    print(df.head())

    # Example inputs: wheel, station, sector
    # Replace these values with the desired chamber identifiers
    try:
        wheel = int(input("Enter Wheel number (-2 to 2): "))
        station = int(input("Enter Station number (1 to 4): "))
        sector = int(input("Enter Sector number (0 to 14): "))
    except ValueError:
        print("Invalid input. Please enter integer values for wheel, station, and sector.")
        return

    try:
        rawId = get_rawId(wheel, station, sector)
    except ValueError as ve:
        print(f"Error: {ve}")
        return

    print(f"Computed rawId: {rawId}")

    # Retrieve Chamber data
    chamber_df = get_chamber_data(df, rawId)
    if chamber_df is None:
        return

    # Create Chamber object from DataFrame
    chamber = create_chamber_object(chamber_df)
    print(f"Chamber {rawId} object has been created.")

    # Plot the Chamber
    plot_chamber(chamber, rawId)

if __name__ == "__main__":
    main()
