import os
import sys
import math

# script, which reads two geometry files (in xyz format) and reorders the atoms in the second file to match the order of the atoms in the first file
# the script is called with two arguments: the first argument is the path to the first geometry file, the second argument is the path to the second geometry file
first_file = os.path.abspath(os.path.expanduser(sys.argv[1]))
second_file = os.path.abspath(os.path.expanduser(sys.argv[2]))

# read the first geometry file
with open(first_file, 'r') as f:
    lines = f.readlines()
    f.close()
n_atoms = int(lines[0])
atoms = [line.split()[0] for line in lines[2:]]
atom_positions = [line.split()[1:4] for line in lines[2:]]
atom_positions = [[float(x) for x in pos] for pos in atom_positions]

# read the second geometry file
with open(second_file, 'r') as f:
    lines = f.readlines()
    n_atoms2 = int(lines[0])
    atoms2 = [line.split()[0] for line in lines[2:]]
    atom_positions2 = [line.split()[1:4] for line in lines[2:]]
    atom_positions2 = [[float(x) for x in pos] for pos in atom_positions2]
    f.close()

# reorder the atoms in the second geometry file
# only reoreder based on the positions of the atoms
# the atom types are not considered
reordered_atoms_2 = []
reordered_atom_positions_2 = []
for atom, pos in zip(atoms, atom_positions):
    for atom2, pos2 in zip(atoms2, atom_positions2):
        if math.isclose(pos[0], pos2[0], rel_tol=1e-3) and\
            math.isclose(pos[1], pos2[1], rel_tol=1e-3) and\
             math.isclose(pos[2], pos2[2], rel_tol=1e-3):
            reordered_atom_positions_2.append(pos2)
            reordered_atoms_2.append(atom2)
            #print(atom2, pos2)
    # the number of atoms between the two files does not necessarily have to be the same, any extra atoms are added to the end of the list
if len(reordered_atoms_2) != n_atoms2:
    # add the missing atoms to the end of the list
    for atom2, pos2 in zip(atoms2, atom_positions2):
        if pos2 not in reordered_atom_positions_2:
            reordered_atoms_2.append(atom2)
            reordered_atom_positions_2.append(pos2)

# write the reordered geometry file
with open('reordered_geometry.xyz' , 'w') as f:
    f.write(str(len(reordered_atoms_2)) + '\n')
    f.write('\n')
    for atom, pos in zip(reordered_atoms_2, reordered_atom_positions_2):
        f.write(atom + ' ' + ' '.join([str(x) for x in pos]) + '\n')
os.rename('reordered_geometry.xyz',second_file)
