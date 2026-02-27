#!/usr/bin/env python3

import sys
from rdkit import Chem
from rdkit.Chem import AllChem

def optimize_3d(mol):
    mol = Chem.AddHs(mol)

    # Remove old conformers
    mol.RemoveAllConformers()

    # Embed new 3D coordinates
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    # Optimize geometry
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        AllChem.UFFOptimizeMolecule(mol)

    return mol


def methylate_position(mol, atom_idx):
    mol = Chem.RWMol(Chem.AddHs(mol))

    target = mol.GetAtomWithIdx(atom_idx)

    hydrogen_idx = None
    for nbr in target.GetNeighbors():
        if nbr.GetSymbol() == "H":
            hydrogen_idx = nbr.GetIdx()
            break

    if hydrogen_idx is None:
        return None

    mol.RemoveAtom(hydrogen_idx)

    new_c = Chem.Atom("C")
    new_c_idx = mol.AddAtom(new_c)
    mol.AddBond(atom_idx, new_c_idx, Chem.BondType.SINGLE)

    for _ in range(3):
        h = Chem.Atom("H")
        h_idx = mol.AddAtom(h)
        mol.AddBond(new_c_idx, h_idx, Chem.BondType.SINGLE)

    new_mol = mol.GetMol()

    try:
        Chem.SanitizeMol(new_mol)
    except:
        return None

    # 🔥 THIS IS THE IMPORTANT PART
    new_mol = optimize_3d(new_mol)

    return new_mol


def find_methylatable_carbons(mol):
    mol = Chem.AddHs(mol)
    candidates = []

    for atom in mol.GetAtoms():
        if (
            atom.GetSymbol() == "C"
            and atom.GetIsAromatic()
            and atom.GetFormalCharge() == 0
        ):
            h_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == "H"]
            if len(h_neighbors) == 1:
                candidates.append(atom.GetIdx())

    return candidates


def main():
    if len(sys.argv) != 2:
        print("Usage: python methylate_with_optimization.py repaired_structure.sdf")
        sys.exit(1)

    infile = sys.argv[1]
    suppl = Chem.SDMolSupplier(infile, removeHs=False)
    mol = next((m for m in suppl if m is not None), None)

    if mol is None:
        raise ValueError("Could not load molecule.")

    positions = find_methylatable_carbons(mol)

    print(f"Found {len(positions)} methylatable positions: {positions}")

    for idx in positions:
        new_mol = methylate_position(mol, idx)

        if new_mol:
            outfile = f"methylated_optimized_{idx}.sdf"
            writer = Chem.SDWriter(outfile)
            writer.write(new_mol)
            writer.close()
            print(f"Saved {outfile}")

    print("Done.")


if __name__ == "__main__":
    main()
