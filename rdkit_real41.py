#!/usr/bin/env python3

import sys
import random
from rdkit import Chem
from rdkit.Chem import AllChem
import os

# ---------------------------
# 3D Optimization
# ---------------------------
def optimize_3d(mol):
    mol = Chem.AddHs(mol)
    mol.RemoveAllConformers()
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        AllChem.UFFOptimizeMolecule(mol)
    return mol

# ---------------------------
# Add Functional Group
# ---------------------------
def add_group(mol, atom_idx, group):
    mol = Chem.RWMol(Chem.AddHs(mol))
    target = mol.GetAtomWithIdx(atom_idx)

    # Remove one hydrogen
    hydrogen_idx = None
    for nbr in target.GetNeighbors():
        if nbr.GetSymbol() == "H":
            hydrogen_idx = nbr.GetIdx()
            break
    if hydrogen_idx is None:
        return None
    mol.RemoveAtom(hydrogen_idx)

    # Add the functional group
    if group == "CH3":
        new_c_idx = mol.AddAtom(Chem.Atom("C"))
        mol.AddBond(atom_idx, new_c_idx, Chem.BondType.SINGLE)
        for _ in range(3):
            h_idx = mol.AddAtom(Chem.Atom("H"))
            mol.AddBond(new_c_idx, h_idx, Chem.BondType.SINGLE)

    elif group == "OH":
        o_idx = mol.AddAtom(Chem.Atom("O"))
        mol.AddBond(atom_idx, o_idx, Chem.BondType.SINGLE)
        h_idx = mol.AddAtom(Chem.Atom("H"))
        mol.AddBond(o_idx, h_idx, Chem.BondType.SINGLE)

    elif group == "OCH3":
        o_idx = mol.AddAtom(Chem.Atom("O"))
        mol.AddBond(atom_idx, o_idx, Chem.BondType.SINGLE)
        c_idx = mol.AddAtom(Chem.Atom("C"))
        mol.AddBond(o_idx, c_idx, Chem.BondType.SINGLE)
        for _ in range(3):
            h_idx = mol.AddAtom(Chem.Atom("H"))
            mol.AddBond(c_idx, h_idx, Chem.BondType.SINGLE)

    elif group == "F":
        f_idx = mol.AddAtom(Chem.Atom("F"))
        mol.AddBond(atom_idx, f_idx, Chem.BondType.SINGLE)

    elif group == "CF3":
        c_idx = mol.AddAtom(Chem.Atom("C"))
        mol.AddBond(atom_idx, c_idx, Chem.BondType.SINGLE)
        for _ in range(3):
            f_idx = mol.AddAtom(Chem.Atom("F"))
            mol.AddBond(c_idx, f_idx, Chem.BondType.SINGLE)

    elif group == "NH2":
        n_idx = mol.AddAtom(Chem.Atom("N"))
        mol.AddBond(atom_idx, n_idx, Chem.BondType.SINGLE)
        for _ in range(2):
            h_idx = mol.AddAtom(Chem.Atom("H"))
            mol.AddBond(n_idx, h_idx, Chem.BondType.SINGLE)

    elif group == "CONH2":
        c_idx = mol.AddAtom(Chem.Atom("C"))
        mol.AddBond(atom_idx, c_idx, Chem.BondType.SINGLE)
        o_idx = mol.AddAtom(Chem.Atom("O"))
        mol.AddBond(c_idx, o_idx, Chem.BondType.DOUBLE)
        n_idx = mol.AddAtom(Chem.Atom("N"))
        mol.AddBond(c_idx, n_idx, Chem.BondType.SINGLE)
        for _ in range(2):
            h_idx = mol.AddAtom(Chem.Atom("H"))
            mol.AddBond(n_idx, h_idx, Chem.BondType.SINGLE)

    elif group == "CN":
        c_idx = mol.AddAtom(Chem.Atom("C"))
        mol.AddBond(atom_idx, c_idx, Chem.BondType.SINGLE)
        n_idx = mol.AddAtom(Chem.Atom("N"))
        mol.AddBond(c_idx, n_idx, Chem.BondType.TRIPLE)

    elif group == "SO2CH3":
        s_idx = mol.AddAtom(Chem.Atom("S"))
        mol.AddBond(atom_idx, s_idx, Chem.BondType.SINGLE)
        o1 = mol.AddAtom(Chem.Atom("O"))
        o2 = mol.AddAtom(Chem.Atom("O"))
        mol.AddBond(s_idx, o1, Chem.BondType.DOUBLE)
        mol.AddBond(s_idx, o2, Chem.BondType.DOUBLE)
        c_idx = mol.AddAtom(Chem.Atom("C"))
        mol.AddBond(s_idx, c_idx, Chem.BondType.SINGLE)
        for _ in range(3):
            h_idx = mol.AddAtom(Chem.Atom("H"))
            mol.AddBond(c_idx, h_idx, Chem.BondType.SINGLE)

    elif group == "CH2OH":
        c_idx = mol.AddAtom(Chem.Atom("C"))
        mol.AddBond(atom_idx, c_idx, Chem.BondType.SINGLE)
        o_idx = mol.AddAtom(Chem.Atom("O"))
        mol.AddBond(c_idx, o_idx, Chem.BondType.SINGLE)
        h_idx = mol.AddAtom(Chem.Atom("H"))
        mol.AddBond(o_idx, h_idx, Chem.BondType.SINGLE)
        for _ in range(2):
            h_idx = mol.AddAtom(Chem.Atom("H"))
            mol.AddBond(c_idx, h_idx, Chem.BondType.SINGLE)

    else:
        return None

    try:
        Chem.SanitizeMol(mol)
    except:
        return None

    return mol.GetMol()

# ---------------------------
# Detect Indole Ring
# ---------------------------
def find_indole_carbons(mol):
    mol = Chem.AddHs(mol)
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    indole_atoms = set()
    for ring1 in atom_rings:
        if len(ring1) == 5 and any(mol.GetAtomWithIdx(i).GetSymbol() == "N" for i in ring1):
            for ring2 in atom_rings:
                if len(ring2) == 6:
                    shared = set(ring1).intersection(ring2)
                    if len(shared) >= 2:
                        indole_atoms.update(ring1)
                        indole_atoms.update(ring2)
    candidates = []
    for idx in indole_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "C" and atom.GetIsAromatic() and atom.GetFormalCharge() == 0:
            h_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == "H"]
            if len(h_neighbors) == 1:
                candidates.append(idx)
    return sorted(candidates)

# ---------------------------
# Main
# ---------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python combined_groups.py repaired_structure.sdf")
        sys.exit(1)

    infile = sys.argv[1]
    suppl = Chem.SDMolSupplier(infile, removeHs=False)
    original_mol = next((m for m in suppl if m is not None), None)
    if original_mol is None:
        raise ValueError("Could not load molecule.")

    positions = find_indole_carbons(original_mol)
    print(f"Indole positions: {positions}")
    if not positions:
        return

    groups = ["NH2","CONH2","CN","SO2CH3","CH2OH","F","OH","CH3","CF3","OCH3"]
    num_derivatives = 100
    max_subs = 3  # max groups per derivative

    os.makedirs("Combined groups", exist_ok=True)

    for i in range(num_derivatives):
        mol = Chem.Mol(original_mol)
        available_positions = {idx:1 for idx in positions}  # track available substitutions
        k = random.randint(1, min(max_subs, len(positions)))
        chosen_positions = random.sample(positions, k)
        filename_parts = []

        for idx in chosen_positions:
            # pick a group randomly
            random.shuffle(groups)
            for grp in groups:
                if available_positions[idx] > 0:
                    new_mol = add_group(mol, idx, grp)
                    if new_mol:
                        mol = new_mol
                        filename_parts.append(f"{grp}{idx}")
                        available_positions[idx] -= 1
                        break  # only one group per selected position

        mol = optimize_3d(mol)
        outfile = f"Combined groups/random_indole_combined_{'_'.join(filename_parts)}_{i+1}.sdf"
        writer = Chem.SDWriter(outfile)
        writer.write(mol)
        writer.close()
        print(f"Saved {outfile}")

    print("Done.")

if __name__ == "__main__":
    main()
