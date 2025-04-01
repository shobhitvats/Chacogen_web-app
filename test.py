import streamlit as st
import tempfile
import os
import sys
import io
from Bio.PDB import PDBParser
import numpy as np

# Define constants
VDW_S = 1.80  # Van der Waals radius of sulfur (S) in Å
VDW_O = 1.52  # Van der Waals radius of oxygen (O) in Å
VDW_N = 1.55  # Van der Waals radius of nitrogen (N) in Å
DISTANCE_THRESHOLD_SO = VDW_S + VDW_O  # 3.32 Å
DISTANCE_THRESHOLD_SN = VDW_S + VDW_N  # 3.35 Å
ANGLE_THETA_MIN = 115  # Minimum angle θ for Ch-bond
ANGLE_THETA_MAX = 155  # Maximum angle θ for Ch-bond
ANGLE_DELTA_MIN = -50  # Minimum torsion angle δ for Ch-bond
ANGLE_DELTA_MAX = 50   # Maximum torsion angle δ for Ch-bond

# Function to calculate distance between two atoms
def calculate_distance(atom1, atom2):
    return np.linalg.norm(atom1.coord - atom2.coord)

# Function to calculate angle θ between centroid, S, and O/N
def calculate_theta(centroid, sulfur, atom):
    vector1 = sulfur.coord - centroid
    vector2 = atom.coord - sulfur.coord
    cos_theta = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return np.degrees(np.arccos(cos_theta))

# Function to calculate torsion angle δ
def calculate_delta(atom1, centroid, sulfur, atom2):
    # Calculate torsion angle using the four atoms
    b1 = centroid - atom1.coord
    b2 = sulfur.coord - centroid
    b3 = atom2.coord - sulfur.coord

    # Normalize vectors
    b1 /= np.linalg.norm(b1)
    b2 /= np.linalg.norm(b2)
    b3 /= np.linalg.norm(b3)

    # Calculate normals to the planes formed by the vectors
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Normalize normals
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)

    # Calculate the angle between the normals
    cos_delta = np.dot(n1, n2)
    delta = np.degrees(np.arccos(cos_delta))

    # Determine the sign of the torsion angle
    if np.dot(np.cross(n1, n2), b2) < 0:
        delta = -delta

    return delta

def calculate_centroid(residue):
    coords = [atom.coord for atom in residue]
    return np.mean(coords, axis=0)

def detect_chalcogen_bonds(file_path):
    # Parse the PDB file
    parser = PDBParser()
    structure = parser.get_structure('protein', file_path)

    # Initialize list to store Ch-bond interactions
    ch_bonds = []

    # Iterate through all sulfur atoms in the structure
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element == 'S':  # Check if the atom is sulfur
                        sulfur = atom
                        centroid = calculate_centroid(residue)  # Calculate centroid of the residue

                        # Iterate through all O and N atoms in the structure
                        for other_model in structure:
                            for other_chain in other_model:
                                for other_residue in other_chain:
                                    for other_atom in other_residue:
                                        if other_atom.element in ['O', 'N']:
                                            # Calculate distance between S and O/N
                                            distance = calculate_distance(sulfur, other_atom)

                                            # Check distance criteria
                                            if (other_atom.element == 'O' and distance <= DISTANCE_THRESHOLD_SO) or \
                                               (other_atom.element == 'N' and distance <= DISTANCE_THRESHOLD_SN):

                                                # Calculate angle θ and torsion angle δ
                                                theta = calculate_theta(centroid, sulfur, other_atom)
                                                delta = calculate_delta(other_atom, centroid, sulfur, other_atom)

                                                # Check angular criteria for Ch-bond
                                                if ANGLE_THETA_MIN <= theta <= ANGLE_THETA_MAX and \
                                                   ANGLE_DELTA_MIN <= delta <= ANGLE_DELTA_MAX:

                                                    # Add the interaction to the list of Ch-bonds
                                                    ch_bonds.append((sulfur, other_atom, distance, theta, delta))

    # Output the results
    if len(ch_bonds) > 0:
        print(f"Chalcogen bonds detected: {len(ch_bonds)}")
        for i, bond in enumerate(ch_bonds):
            sulfur, other_atom, distance, theta, delta = bond
            print(f"Ch-bond {i+1}:")
            print(f"  Sulfur position: {sulfur.coord}")
            print(f"  O/N position: {other_atom.coord}")
            print(f"  Distance: {distance:.2f} Å")
            print(f"  Angle θ: {theta:.2f}°")
            print(f"  Torsion angle δ: {delta:.2f}°")
    else:
        print("No chalcogen bonds detected.")

# Streamlit app title and description
st.title("Sulfur-Mediated Chalcogen Bond Detection")
st.markdown("""
Upload a PDB file to detect sulfur-mediated chalcogen interactions (S···O/N).
The results will be displayed below in the output box.
""")

# File uploader for PDB files
uploaded_file = st.file_uploader("Choose a PDB file", type="pdb")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Redirect stdout to capture print statements
    output_buffer = io.StringIO()
    sys.stdout = output_buffer

    # Run the `detect_chalcogen_bonds` function
    st.info("Processing the uploaded PDB file...")
    try:
        with st.spinner("Detecting chalcogen bonds..."):
            detect_chalcogen_bonds(temp_file_path)  # Call the function

        # Reset stdout and get the captured output
        sys.stdout = sys.__stdout__
        output = output_buffer.getvalue()

        # Display the output in a text box
        st.success("Chalcogen bond detection completed!")
        st.subheader("Results:")
        st.text_area("Output", value=output, height=300)

    except Exception as e:
        # Reset stdout in case of an error
        sys.stdout = sys.__stdout__
        st.error(f"An error occurred while processing the file: {e}")

    # Clean up the temporary file
    os.remove(temp_file_path)
else:
    st.warning("Please upload a PDB file to proceed.")
