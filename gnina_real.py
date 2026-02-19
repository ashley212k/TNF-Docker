#!/usr/bin/env python3
import os
import subprocess
from rdkit import Chem
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------
# Base directories
# -----------------------------
base_dir = r"C:\Users\ashle\Downloads\LigandModifications"
inputs_dir = os.path.join(base_dir, "inputs")  # All ligand subfolders
outputs_dir = os.path.join(base_dir, "docking_outputs")
os.makedirs(outputs_dir, exist_ok=True)

# -----------------------------
# GNINA parameters
# -----------------------------
receptor_path = os.path.join(base_dir, "receptor_cleaned.pdb")
center_x, center_y, center_z = -13.88, 71.43, 26.90
size_x, size_y, size_z = 18, 18, 18
exhaustiveness = 16
cnn_model = "fast"  # use "rescore" to just rescore instead of docking
max_workers = 4     # Adjust based on CPU cores

# -----------------------------
# Dock a single ligand
# -----------------------------
def dock_ligand(ligand_file, input_subfolder):
    ligand_path = f"/data/inputs/{input_subfolder}/{ligand_file}"  # Docker path
    output_file = f"/data/docking_outputs/{input_subfolder}_docked_{ligand_file}"
    local_output_file = os.path.join(outputs_dir, f"{input_subfolder}_docked_{ligand_file}")

    os.makedirs(os.path.join(outputs_dir), exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{base_dir}:/data",
        "gnina/gnina:latest",
        "gnina",
        "-r", "/data/receptor_cleaned.pdb",
        "-l", ligand_path,
        "--center_x", str(center_x),
        "--center_y", str(center_y),
        "--center_z", str(center_z),
        "--size_x", str(size_x),
        "--size_y", str(size_y),
        "--size_z", str(size_z),
        "--exhaustiveness", str(exhaustiveness),
        "--cnn", cnn_model,
        "-o", output_file
    ]

    subprocess.run(cmd, check=True)
    return local_output_file

# -----------------------------
# Extract minimized affinity from a docked SDF
# -----------------------------
def extract_affinity(sdf_file):
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
    best_affinity = None

    for mol in suppl:
        if mol is None:
            continue
        if mol.HasProp("minimizedAffinity"):
            affinity = float(mol.GetProp("minimizedAffinity"))
            if best_affinity is None or affinity < best_affinity:
                best_affinity = affinity

    return best_affinity

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Find all ligand subfolders
    subfolders = [f for f in os.listdir(inputs_dir) if os.path.isdir(os.path.join(inputs_dir, f))]
    print(f"Found subfolders: {subfolders}")

    # Gather all ligands
    all_ligands = []
    for sub in subfolders:
        folder_path = os.path.join(inputs_dir, sub)
        ligands = [f for f in os.listdir(folder_path) if f.endswith(".sdf")]
        all_ligands.extend([(f, sub) for f in ligands])

    print(f"Total ligands to dock: {len(all_ligands)}")

    # Parallel docking
    docked_files = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ligand = {executor.submit(dock_ligand, ligand, sub): (ligand, sub) for ligand, sub in all_ligands}
        for future in as_completed(future_to_ligand):
            ligand, sub = future_to_ligand[future]
            try:
                result_file = future.result()
                docked_files.append(result_file)
                print(f"Docked {ligand}")
            except Exception as e:
                print(f"Error docking {ligand}: {e}")

    # -----------------------------
    # Extract affinities and rank ligands
    # -----------------------------
    ranking_file = os.path.join(outputs_dir, "ligand_affinity_ranking.csv")
    results = []

    for docked_file in docked_files:
        affinity = extract_affinity(docked_file)
        if affinity is not None:
            results.append({
                "Ligand": os.path.basename(docked_file),
                "minimizedAffinity (kcal/mol)": affinity
            })

    # Sort by affinity (most negative = best)
    results.sort(key=lambda x: x["minimizedAffinity (kcal/mol)"])

    # Write CSV
    with open(ranking_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Ligand", "minimizedAffinity (kcal/mol)"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Docking complete. Ranking saved to {ranking_file}")
