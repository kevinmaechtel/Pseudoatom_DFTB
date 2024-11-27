# Needed: conda environment with dftbplus installed
import os
import math
import shutil
from typing import List, Tuple
from time import time
from scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

# define function which reads the skf files
def read_skf_file(file: str) -> Tuple[List[List[float]], List[List[str]], List[str]]:
    """
    Reads an SKF file and extracts the Hamiltonian and Overlap matrix (H_S), header, and footer.
    Args:
        file (str): The path to the SKF file to be read.
    Returns:
        Tuple[List[List[float]], List[List[str]], List[str]]:
            - H_S (List[List[float]]): The Hamiltonian and Overlap matrix extracted from the file.
            - header (List[List[str]]): The header lines of the file.
            - footer (List[str]): The footer lines of the file.
    The function reads the SKF file line by line. It first extracts the header until it encounters
    a line with 6, 12, 16, or 20 elements. The first line of the header contains the number of elements
    in the H_S matrix. The H_S matrix is then extracted from the subsequent lines, and the remaining lines
    are considered as the footer.
    """
    with open(file) as f:
        lines = f.readlines()
        # as long as the length of the line array is not 6, 12 or 20, add the lines to the header
        # if the length of the line array is 6, 12 or 20, stop adding lines to the header
        header = []
        for i in range(len(lines)):
            if len(lines[i].split()) == 6 or\
                len(lines[i].split()) == 12 or\
                len(lines[i].split()) == 20 or\
                len(lines[i].split()) == 16:
                break
            header.append(lines[i].split())
        # the first line of the header contains the number of elements in the H_S matrix
        len_HS = int(header[0][1].split(",")[0])
        #print(len_HS)

        # the H_S matrix is the lines between the header and the footer and it is split into a list of lists and the list of the lists is len_HS long
        H_S = [lines[i].split() for i in range(len(header),
                                               len(header) + len_HS)]

        # the footer is the lines after the H_S matrix
        footer = lines[len(header) + len_HS:]
    return H_S, header, footer

# define a function, which manipulates only the hubbard of the s and p orbitals
def manipulate_spl_Hubbard(original_dir: str, new_dir: str, Hubbard_s: float, Hubbard_p: float, Energy_p: float, Energy_s: float, el: str, elements: List[str]) -> None:
    """
    Function manipulates the .spl files in the original directory and copies the manipulated files to the new directory.

    Args:
        original_dir (str): Directory containing the original .spl files.
        new_dir (str): Directory where the manipulated files are copied to.
        Hubbard_s (float): Hubbard U parameter of the s Orbital, it should be changed to.
        Hubbard_p (float): Hubbard U parameter of the p Orbital, it should be changed to.
        Energy_p (float): Energy of the p Orbital, it should be changed to.
        Energy_s (float): Energy of the s Orbital, it should be changed to.
        el (str): Element to be replaced in the file names.
        elements (List[str]): List of elements in the geometry file.

    Returns:
        None: Copies the manipulated files to the new directory.
    """
    # if the directory exits, remove it
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    # create a new directory
    os.makedirs(new_dir)
    # change to the new directory
    os.chdir(new_dir)
    # copy the files from the original directory, which end with .spl and have one of the elements in the first two letters
    for file in os.listdir(original_dir):
        if file.endswith('.spl') and file[1] in elements and file[0] in elements:
            shutil.copyfile(f'{original_dir}/{file}', f'{file}')

    # create a list of all of the files in the target directory ending with .spl
    files = [f for f in os.listdir() if f.endswith('.spl')]
    # create a list of all of the uncompressed files containing a c in the first two letters
    uncompressed_files = [f for f in files if f.endswith('-uncomp-c.spl')]
    # create a list of all of the compressed files containing a c in the first two letters
    compressed_files = [f for f in files if f.endswith('-c.spl') and not f.endswith('-uncomp-c.spl')]

    for file in uncompressed_files:
        if 'cc' in file[:2]:
            H_S, header, footer = read_skf_file(file)
            header[1][1] = str(Energy_p)
            header[1][2] = str(Energy_s)
            header[1][5] = str(Hubbard_p)
            header[1][6] = str(Hubbard_s)
            header[1][-1] = '2.000000000000E+00'
            header[1][-2] = '3.000000000000E+00'
            with open(file[:2].replace('c',el)+file[2:], 'w') as f:
                for line in header:
                    f.write(' '.join(line) + '\n')
                for line in H_S:
                    f.write(' '.join(line) + '\n')
                for line in footer:
                    f.write(line)
                f.close()
            shutil.copyfile(f'{file}', f'{file[0].replace("c",el)+file[1:]}')
            shutil.copyfile(f'{file}', f'{file[0]+file[1].replace("c",el)+file[2:]}')
        elif 'c' in file[:2] and not 'cc' in file[:2]:
            shutil.copyfile(f'{file}', f'{file[:2].replace("c",el)+file[2:]}')
    
    compressed_files = [f for f in os.listdir() if f.endswith('-c.spl') and not f.endswith('-uncomp-c.spl')]
    for file in compressed_files:
        if 'cc' in file[:2]:
            H_S, header, footer = read_skf_file(file)
            header[1][1] = str(Energy_p)
            header[1][2] = str(Energy_s)
            header[1][5] = str(Hubbard_p)
            header[1][6] = str(Hubbard_s)
            header[1][-1] = '2.000000000000E+00'
            header[1][-2] = '3.000000000000E+00'
            with open(file[:2].replace('c',el)+file[2:], 'w') as f:
                for line in header:
                    f.write(' '.join(line) + '\n')
                for line in H_S:
                    f.write(' '.join(line) + '\n')
                for line in footer:
                    f.write(line)
                f.close()
            shutil.copyfile(f'{file}', f'{file[0].replace("c",el)+file[1:]}')
            shutil.copyfile(f'{file}', f'{file[0]+file[1].replace("c",el)+file[2:]}')
        elif 'c' in file[:2] and not 'cc' in file[:2]:
            shutil.copyfile(f'{file}', f'{file[:2].replace("c",el)+file[2:]}')
    
    os.chdir('..')
    return

# define a function, which converts a geometry file to a gen file
def convert_to_gen(geo_file: str) -> str:
    """
    Converts a geometry file to a gen file.

    This function takes a geometry file in .xyz, .pdb, .gro, or .gen format and converts it to a .gen file.
    If the input file is in .pdb or .gro format, it first converts it to a temporary .xyz file before converting
    it to a .gen file.

    Parameters:
    geo_file (str): The name of the geometry file to be converted.

    Returns:
    str: The name of the generated .gen file.

    Raises:
    SystemExit: If the file does not exist or if the file extension is not one of the supported formats (.xyz, .pdb, .gro, .gen).

    Notes:
    - The function uses external tools like `xyz2gen` and `obabel` for file conversions.
    - Ensure that the paths to these tools are correctly set in the environment.
    """
    # check if the file exists
    if not os.path.exists(geo_file):
        print('The file does not exist')
        exit()
    # look for the file extension
    if not geo_file.endswith('.xyz') and not geo_file.endswith('.gen') and not geo_file.endswith('.gro') and not geo_file.endswith('.pdb'):
        print('The file is not an xyz, gro, pdb or gen file')
        exit()
    # if the file is an xyz file, convert it to a gen file
    if geo_file.endswith('.xyz'):
        os.system(f'xyz2gen  {geo_file}')
        return geo_file[:-4] + '.gen'
    # if the file is a pdb file, first convert it to an xyz file and then convert it to a gen file
    if geo_file.endswith('.pdb'):
        os.system(f'obabel -ipdb {geo_file} -oxyz -O temp.xyz')
        os.system(f'xyz2gen temp.xyz')
        os.system('rm temp.xyz')
        os.rename('temp.gen', geo_file[:-4] + '.gen')
        return geo_file[:-4] + '.gen'
    # if the file is a gro file, first convert it to an xyz file and then convert it to a gen file
    if geo_file.endswith('.gro'):
        os.system(f'obabel -igro {geo_file} -oxyz -O temp.xyz')
        os.system(f'xyz2gen temp.xyz')
        os.system('rm temp.xyz')
        os.rename('temp.gen', geo_file[:-4] + '.gen')
        return geo_file[:-4] + '.gen'
    
# define a function, which gets as an input the path to the skf directory and the geometry file and the charge
def create_hsd_Hub(skf_dir: str, geo_file: str, hsd_file: str) -> None:
    """Function creates the hsd file for the DFTB2 calculations
    skf_dir = path to the skf directory (string)
    geo_file = path to the geometry file (string)
    hsd_file = path to the hsd file (string)

    Returns nothing, but creates the hsd file"""
    hsd = """Geometry = GenFormat {
<<<  '""" + geo_file + "'\n}"
    
    hsd += """
Hamiltonian = DFTB {
    SCC = Yes
    SCCTolerance = 1.0e-5
    MaxSCCIterations = 500
    Filling = Fermi {
        Temperature [Kelvin] = 50.0
    }
    Mixer = Anderson {
        MixingParameter = 0.2
    }
    Charge = 0
    MaxAngularMomentum = {
"""
    # read the second line of the geometry file
    with open(geo_file, 'r') as f:
        next(f)
        line2 = f.readline()
        f.close()
    # split the line into a list of strings
    line2 = line2.split()
    # define a dictionary with the elements and the maximum angular momentum
    max_ang_mom = {'H': 's', 'C': 'p', 'N': 'p', 'O': 'p', 'F': 'p', 'Cl': 'p',
                    'Br': 'p', 'I': 'p', 'P': 'd', 'S': 'd', 'Se': 'd', 'Te': 'd',
                    'Z': 'p', 'Y': 'p', 'X': 'p', 'W': 'p', 'V': 'p'}
    # cycle through the elements in the second line of the geometry file
    for el in line2:
        # add the element and the maximum angular momentum to the hsd string with " in front and " at the end
        hsd += '        ' + el + ' = "' + max_ang_mom[el] + '"\n'
    # add the rest of the hsd string
    hsd += """    }

    SlaterKosterFiles = Type2FileNames {
        Prefix = '""" + skf_dir + """'
        Separator = ''
        Suffix = '-uncomp-c.spl'
        LowerCaseTypeName = Yes
    }
}"""
    hsd += """
Analysis = {
    WriteEigenvectors = Yes
    EigenvectorsAsText = Yes
}"""
    # write the hsd string to a file
    with open(hsd_file, 'w') as f:
        f.write(hsd)
        f.close()
    return

# mefine a function, which reads two xyz files and matches the atoms in the two files by their positions
def match_xyz(xyz1: str, xyz2: str) -> List[tuple]:
    """Function reads two xyz files and matches the atoms in the two files by their positions
    xyz1 = path to the first xyz file (string)
    xyz2 = path to the second xyz file (string)

    Returns a list of tuples containing the indices of the matched atoms"""
    # read the xyz files
    with open(xyz1, 'r') as f:
        next(f)
        next(f)
        xyz1_lines = f.readlines()
        f.close()
    with open(xyz2, 'r') as f:
        next(f)
        next(f)
        xyz2_lines = f.readlines()
        f.close()
    # define a list to store the matched atoms
    matched_atoms = []
    # cycle through the atoms in the first xyz file
    for i in range(len(xyz1_lines)):
        # split the line into a list of strings
        el1 = xyz1_lines[i].split()[0]
        # get the element and the position of the atom
        pos1 = [float(xyz1_lines[1].split()[1]), 
                float(xyz1_lines[1].split()[2]),
                float(xyz1_lines[1].split()[3])]
        # cycle through the atoms in the second xyz file
        for j in range(len(xyz2_lines)):
            pos2 = [float(xyz2_lines[j].split()[1]),
                    float(xyz2_lines[j].split()[2]),
                    float(xyz2_lines[j].split()[3])]
            if math.isclose(pos1[0], pos2[0], abs_tol=1e-3) and\
                math.isclose(pos1[1], pos2[1], abs_tol=1e-3) and\
                math.isclose(pos1[2], pos2[2], abs_tol=1e-3):
                el2 = xyz2_lines[j].split()[0]
                matched_atoms.append((i, j))
    return matched_atoms

# define a function, which reads the atomic charges from a file
def read_eigenvectors(file: str, MO: int) -> List[List[float]]:
    """Function reads the eigenvectors from a file and normalizes them
    
    Args:
        file = path to the file containing the eigenvectors (string)
        MO = number of the molecular orbital (int)
    
    Returns the eigenvectors as a list of lists"""

    with open(file, 'r') as f:
        lines = f.readlines()
        f.close()
    eigenvector = []
    for i in range(len(lines)):
        if f'Eigenvector:' in lines[i] and str(MO) in lines[i]:
            for j in range(i+1, len(lines)):
                if 'Eigenvector' in lines[j]:
                    break
                elif len(lines[j].split()) == 0:
                    continue
                eigenvector.append(float(lines[j].split()[-2]))
    eigenvector /= np.linalg.norm(eigenvector)
    return eigenvector

# define a function which calculates the rmsd between two vectors of atomic charges
def rmsd(charges1: List[float], charges2: List[float], matched_atoms: List[tuple]) -> float:
    """Function calculates the rmsd between two vectors of atomic charges
    charges1 = vector of atomic charges for the first molecule (list of floats)
    charges2 = vector of atomic charges for the second molecule (list of floats)
    matched_atoms = list of tuples containing the indices of the matched atoms (list of tuples)

    Returns the rmsd between the two vectors of atomic charges"""
    # calculate the rmsd based on the matched atoms
    rmsd = 0
    for i, j in matched_atoms:
        rmsd += (charges1[i] - charges2[j])**2
    rmsd = math.sqrt(rmsd / len(matched_atoms))
    return rmsd

# define a function, which runs the dftb+ calculation for both the neutral and the charged system
def run_dftb_Hub(skf_dir: str, hsd_file: str, geom_file: str) -> Tuple[float , float , List[float], List[float], List[float]]:
    """Function runs the dftb+ (DFTB2) calculation for both the neutral and the charged system
    skf_dir = path to the skf directory (string)
    hsd_file = path to the hsd file (string)
    geom_file = path to the geometry file (string)

    Returns the ionization potential, the electron affinity and the atomic charges"""
    # convert the geometry file to a gen file
    geom_file = convert_to_gen(geom_file)
    # create the hsd file for the neutral system
    create_hsd_Hub(skf_dir, geom_file, hsd_file)
    # run dftb+ for the neutral system
    os.system('dftb+ > neutral.log')
    # chach if the detailed.out file exists
    if not os.path.exists('detailed.out'):
        print('The Calculation for the neutral system failed')
        return None, None, None, None, None
    # read the detailed.out file
    with open('detailed.out', 'r') as f:
        lines = f.readlines()
        f.close()
    for i in range(len(lines)):
        if 'Atomic gross charges' in lines[i]:
            # loop over the following lines, until the empty line is reached
            charges = []
            for j in range(i+2, len(lines)):
                if len(lines[j].split()) != 2:
                    break
                charges.append(float(lines[j].split()[1]))
        elif 'Total Electronic energy' in lines[i]:
            neutral_energy = float(lines[i].split()[-2])
            break
    with open('band.out', 'r') as f:
        next(f)
        lines = f.readlines()
        f.close()
    for i in range(len(lines)-1):
        if float(lines[i].split()[2]) == 0.0:
            LUMO = int(lines[i].split()[0])
            HOMO = int(lines[i-1].split()[0])
            break
    eigenvector_HOMO = read_eigenvectors('eigenvec.out', HOMO)
    eigenvector_LUMO = read_eigenvectors('eigenvec.out', LUMO)
    # adjust the hsd file for the cation
    with open(hsd_file, 'r') as f:
        hsd = f.readlines()
        f.close()
    with open(hsd_file, 'w') as f:
        for line in hsd:
            if 'Charge' in line:
                f.write('    Charge = 1\n')
                continue
            f.write(line)
        f.close()
    os.remove('detailed.out')
    # run dftb+ for the charged system
    os.system('dftb+ > cation.log')
    # check if the detailed.out file exists
    if not os.path.exists('detailed.out'):
        print('The Calculation for the cation failed.')
        return None, None, None, None, None
    # read the detailed.out file
    with open('detailed.out', 'r') as f:
        lines = f.readlines()
        f.close()
    for i in range(len(lines)):
        if 'Total Electronic energy' in lines[i]:
            cat_energy = float(lines[i].split()[-2])
            break
    # calculate the ionization potential
    ionization_potential = cat_energy - neutral_energy
    # remove the detailed.out file
    os.remove('detailed.out')
    # adjust the hsd file for the anion
    with open(hsd_file, 'r') as f:
        hsd = f.readlines()
        f.close()
    with open(hsd_file, 'w') as f:
        for line in hsd:
            if 'Charge' in line:
                f.write('    Charge = -1\n')
                continue
            f.write(line)
        f.close()
    # run dftb+ for the anion
    os.system('dftb+ > anion.log')
    # check if the detailed.out file exists
    if not os.path.exists('detailed.out'):
        print('The Calculation for the anion failed.')
        return None, None, None, None, None
    # read the detailed.out file
    with open('detailed.out', 'r') as f:
        lines = f.readlines()
        f.close()
    for i in range(len(lines)):
        if 'Total Electronic energy' in lines[i]:
            an_energy = float(lines[i].split()[-2])
            break
    # remove the detailed.out file
    os.remove('detailed.out')
    # calculate the electron affinity
    electron_affinity = an_energy - neutral_energy
    return ionization_potential, electron_affinity, charges, eigenvector_HOMO, eigenvector_LUMO

# define a function, which does the same as above but for the Hubbard parameters additionally
def single_Hubbard(Energy_p: float, Energy_s: float, Hubbard_p: float, Hubbard_s: float, pseudo_geom: str, ph_ph_charges: List[float], ph_ph_IP: float, ph_ph_EA: float, matched_atoms: List[tuple], el: str, ph_ph_EV_HOMO: List[float], ph_ph_EV_LUMO: List[float]) -> List[float]:
    """
    Function which runs the DFTB2 calculations and calculates the delta IP, delta EA, and delta RMSD.

    Args:
        Energy_p (float): Energy of the p Orbital, it should be changed to.
        Energy_s (float): Energy of the s Orbital, it should be changed to.
        Hubbard_p (float): Hubbard U of the p Orbital, it should be changed to.
        Hubbard_s (float): Hubbard U of the s Orbital, it should be changed to.
        pseudo_geom (str): Name of the file containing the geometry with the pseudo atoms.
        ph_ph_charges (List[float]): Gross Atomic charges of the molecule without the pseudo atoms.
        ph_ph_IP (float): Ionisation potential of the molecule without the pseudo atoms.
        ph_ph_EA (float): Electron Affinity of the molecule without the pseudo atoms.
        matched_atoms (List[tuple]): List of tuples containing the indices of the atoms in the pseudo and non-pseudo atom list.
        el (str): Name of the pseudo atom.
        ph_ph_EV_HOMO (List[float]): Eigenvectors of the HOMO of the molecule without pseudo atom.
        ph_ph_EV_LUMO (List[float]): Eigenvectors of the LUMO of the molecule without pseudo atom.

    Returns:
        List[float]: A list containing the Energy_p, Energy_s, Hubbard_p, Hubbard_s, IP_diff, EA_diff, RMSD, and dot product of the HOMO and LUMO of the molecule with pseudo atom compared to the normal molecule.

    Calculates the delta IP, delta EA, delta RMSD, and the Hubbard parameters between the pseudo and non-pseudo system.
    """
    # round the Hubbard U parameter and the Energy_s parameter to three decimal places
    Energy_p = round(Energy_p, 6)
    Energy_s = round(Energy_s, 6)
    Hubbard_p = round(Hubbard_p, 6)
    Hubbard_s = round(Hubbard_s, 6)
    # create a directory for the calculation
    calc_dir = f"{str(Energy_p).replace('-','_')}_{str(Energy_s).replace('-','_')}_{str(Hubbard_p).replace('-','_')}_{str(Hubbard_s).replace('-','_')}_calc"
    os.makedirs(calc_dir, exist_ok=True)
    os.chdir(calc_dir)
    shutil.copyfile(f'../{pseudo_geom}', f'{pseudo_geom}')
    cwd = os.getcwd()
    skf_dir = f"{cwd}/{str(Energy_p).replace('-','_')}_{str(Energy_s).replace('-','_')}_{str(Hubbard_p).replace('-','_')}_{str(Hubbard_s).replace('-','_')}_params/"
    # read the second column of the geometry file from the third line
    with open(pseudo_geom, 'r') as f:
        for i in range(2):
            next(f)
        lines = f.readlines()
        f.close()
    # filter for the unique elements in the geometry file and convert them to lower case
    column = []
    for line in lines:
        column.append(line.split()[0])
    elements = np.unique(column)
    elements = [el.lower() for el in elements]
    manipulate_spl_Hubbard('/path/to/original/parameters/', skf_dir, Hubbard_s, Hubbard_p, Energy_p, Energy_s, el, elements)
    # run dftb+ for the pseudo system
    pseudo_IP, pseudo_EA, pseudo_charges, pseudo_EV_HOMO, pseudo_EV_LUMO = run_dftb_Hub(skf_dir, 'dftb_in.hsd', pseudo_geom)
    # if the calculation fails, continue
    if pseudo_IP is None and pseudo_charges is None and pseudo_EA is None and pseudo_EV_HOMO is None and pseudo_EV_LUMO is None:
        os.chdir('..')
        os.system(f'rm -r {calc_dir}')
        return [Energy_p, Energy_s, Hubbard_p, Hubbard_s, 100.0, 100.0, 100.0, 100.0, 100.0, 10.0, 10.0]
    dot_HOMO = np.dot(pseudo_EV_HOMO, ph_ph_EV_HOMO[:len(pseudo_EV_HOMO)])
    dot_LUMO = np.dot(pseudo_EV_LUMO, ph_ph_EV_LUMO[:len(pseudo_EV_LUMO)])
    # calculate the RMSD
    rmsd_val = rmsd(ph_ph_charges, pseudo_charges, matched_atoms)
    # calculate the difference in the ionization potentials
    IP_diff = abs(pseudo_IP - ph_ph_IP)
    # calculate the difference in the electron affinities
    EA_diff = abs(pseudo_EA - ph_ph_EA)
    os.chdir('..')
    os.system(f'rm -r {calc_dir}')
    
    fct_IP = IP_diff + rmsd_val + 1 - dot_HOMO
    fct_EA = EA_diff + rmsd_val + 1 - dot_LUMO
    return [Energy_p, Energy_s, Hubbard_p, Hubbard_s, fct_IP, fct_EA, IP_diff, EA_diff, rmsd_val, dot_HOMO, dot_LUMO]

def parse_fct_IP(Variables: List[float], pseudo_geom: str, ph_ph_charges: List[float], ph_ph_IP: float, ph_ph_EA: float, matched_atoms: List[tuple], el: str, ph_ph_EV_HOMO: List[float], ph_ph_EV_LUMO: List[float]) -> float:
    """
    Function which runs the DFTB2 calculations (function) and calculates the delta IP, delta EA, and delta RMSD and parses only the IP function to scipy.

    Args:
        Variables (List[float]): Linst Containing the Energies of p and s orbitals and p and s Hubbards.
        pseudo_geom (str): Name of the file containing the geometry with the pseudo atoms.
        ph_ph_charges (List[float]): Gross Atomic charges of the molecule without the pseudo atoms.
        ph_ph_IP (float): Ionisation potential of the molecule without the pseudo atoms.
        ph_ph_EA (float): Electron Affinity of the molecule without the pseudo atoms.
        matched_atoms (List[tuple]): List of tuples containing the indices of the atoms in the pseudo and non-pseudo atom list.
        el (str): Name of the pseudo atom.
        ph_ph_EV_HOMO (List[float]): Eigenvectors of the HOMO of the molecule without pseudo atom.
        ph_ph_EV_LUMO (List[float]): Eigenvectors of the LUMO of the molecule without pseudo atom.

    Returns:
        float: IP function

    Calculates the delta IP, delta EA, delta RMSD, and the Hubbard parameters between the pseudo and non-pseudo system.
    """

    # run the single_Hubbard function
    output = single_Hubbard(Variables[0], Variables[1], Variables[2], Variables[3], pseudo_geom, ph_ph_charges, ph_ph_IP, ph_ph_EA, matched_atoms, el, ph_ph_EV_HOMO, ph_ph_EV_LUMO)
    return output[4]

def parse_fct_EA(Variables: List[float], pseudo_geom: str, ph_ph_charges: List[float], ph_ph_IP: float, ph_ph_EA: float, matched_atoms: List[tuple], el: str, ph_ph_EV_HOMO: List[float], ph_ph_EV_LUMO: List[float]) -> float:
    """
    Function which runs the DFTB2 calculations (function) and calculates the delta IP, delta EA, and delta RMSD and parses only the IP function to scipy.

    Args:
        Variables (List[float]): Linst Containing the Energies of p and s orbitals and p and s Hubbards.
        pseudo_geom (str): Name of the file containing the geometry with the pseudo atoms.
        ph_ph_charges (List[float]): Gross Atomic charges of the molecule without the pseudo atoms.
        ph_ph_IP (float): Ionisation potential of the molecule without the pseudo atoms.
        ph_ph_EA (float): Electron Affinity of the molecule without the pseudo atoms.
        matched_atoms (List[tuple]): List of tuples containing the indices of the atoms in the pseudo and non-pseudo atom list.
        el (str): Name of the pseudo atom.
        ph_ph_EV_HOMO (List[float]): Eigenvectors of the HOMO of the molecule without pseudo atom.
        ph_ph_EV_LUMO (List[float]): Eigenvectors of the LUMO of the molecule without pseudo atom.

    Returns:
        float: EA function

    Calculates the delta IP, delta EA, delta RMSD, and the Hubbard parameters between the pseudo and non-pseudo system.
    """

    # run the single_Hubbard function
    output = single_Hubbard(Variables[0], Variables[1], Variables[2], Variables[3], pseudo_geom, ph_ph_charges, ph_ph_IP, ph_ph_EA, matched_atoms, el, ph_ph_EV_HOMO, ph_ph_EV_LUMO)
    return output[5]

def scipy_optimise(Energy_p: float, Energy_s: float, Hubbard_p: float, Hubbard_s: float, pseudo_geom: str, ph_ph_IP: float, ph_ph_EA: float, ph_ph_charges: List[float], ph_ph_EV_HOMO: List[float], ph_ph_EV_LUMO: List[float], matched_atoms: List[tuple], el: str) -> List[float]:
    """
    Function which runs scipy minimizations of the IP and EA functions.

    Args:
        Energy_p (float): Energy of the p Orbital, it should be changed to.
        Energy_s (float): Energy of the s Orbital, it should be changed to.
        Hubbard_p (float): Hubbard U of the p Orbital, it should be changed to.
        Hubbard_s (float): Hubbard U of the s Orbital, it should be changed to.
        pseudo_geom (str): Name of the file containing the geometry with the pseudo atoms.
        ph_ph_charges (List[float]): Gross Atomic charges of the molecule without the pseudo atoms.
        ph_ph_IP (float): Ionisation potential of the molecule without the pseudo atoms.
        ph_ph_EA (float): Electron Affinity of the molecule without the pseudo atoms.
        matched_atoms (List[tuple]): List of tuples containing the indices of the atoms in the pseudo and non-pseudo atom list.
        el (str): Name of the pseudo atom.
        ph_ph_EV_HOMO (List[float]): Eigenvectors of the HOMO of the molecule without pseudo atom.
        ph_ph_EV_LUMO (List[float]): Eigenvectors of the LUMO of the molecule without pseudo atom.

    Returns:
        List[float]: optimized parameters (s and p orbital energies and Hubbard) of the IP optimization , its function, optimized parameters of the EA optimization, its function and the average time per optimization

    Calculates the delta IP, delta EA, delta RMSD, and the Hubbard parameters between the pseudo and non-pseudo system.
    """
    # Define the initial guess for the variables
    initial_guess = [Energy_p, Energy_s, Hubbard_p, Hubbard_s]

    # Define the bounds for the variables
    bounds = [(-10.0, 10.0), (-10.0, 10.0), (0.0, 10.0), (0.0, 10.0)]

    # Minimize the IP function
    start = time()
    result_IP = minimize(parse_fct_IP, initial_guess, args=(pseudo_geom, ph_ph_charges, ph_ph_IP, ph_ph_EA, matched_atoms, el, ph_ph_EV_HOMO, ph_ph_EV_LUMO), method='L-BFGS-B', bounds=bounds, options={'maxiter': 10000})
    calc = time() - start
    optimal_x_IP = result_IP.x
    optimal_fct_IP = result_IP.fun
    #print('Finished IP optimization', optimal_x_IP, optimal_fct_IP)

    # Minimize the EA function
    start = time()
    result_EA = minimize(parse_fct_EA, initial_guess, args=(pseudo_geom, ph_ph_charges, ph_ph_IP, ph_ph_EA, matched_atoms, el, ph_ph_EV_HOMO, ph_ph_EV_LUMO), method='L-BFGS-B', bounds=bounds, options={'maxiter': 10000})
    calc += time() - start
    calc /= 2
    optimal_x_EA = result_EA.x
    optimal_fct_EA = result_EA.fun
    #print('Finished EA optimization',optimal_x_EA, optimal_fct_EA)

    return [optimal_x_IP[0], optimal_x_IP[1], optimal_x_IP[2], optimal_x_IP[3], optimal_fct_IP, optimal_x_EA[0], optimal_x_EA[1], optimal_x_EA[2], optimal_x_EA[3], optimal_fct_EA, calc]

# function to run the gradient optimisation
def run_grad(Energy_p_values: List[float], Energy_s_values: List[float], Hubbard_p_values: List[float], Hubbard_s_values: List[float], ph_geom: str, pseudo_geom: str, el: str) -> List[float]:
    """Function runs optimisation of Hubbard and orbital energies

    Args:
        Energy_p_values (List[float]): List of all possible p orbital energies
        Energy_s_values (List[float]): List of all possible s orbital energies
        Hubbard_p_values (List[float]): List of all possible p Hubbards
        Hubbard_s_values (List[float]): List of all possible s Hubbards
        ph_geom (str): Path to original geometry containing linker atom
        pseudo_geom (str): Path to geometry containing pseudo atom
        el (str): Name of Pseudo Atom

    Returns:
        List[float]: minimum of IP function, corresponding p and s orbital energy, p and s Hubbards, IP difference, charges RMSD, dot product of the HOMO;
                    minimum of EA function, corresponding p and s orbital energy, p and s Hubbards, EA difference, charges RMSD, dot product of the LUMO
    """
    cwd = os.getcwd()
    os.system('mkdir -p original')
    os.chdir('original')
    shutil.copyfile(cwd + '/' + ph_geom, ph_geom)
    ph_ph_IP, ph_ph_EA, ph_ph_charges, ph_ph_EV_HOMO, ph_ph_EV_LUMO = run_dftb_Hub('/path/to/original/parameters/', 'dftb_in.hsd', ph_geom)
    os.chdir('..')
    print('The IP and EA values have been calculated for the original system.\n')
    # Delete the original directory
    os.system('rm -r original')

    # match the geometries to each other
    matched_atoms = match_xyz(ph_geom, pseudo_geom)

    start = time()
    input_list = []
    limit = 0.0005
    for i in range(int(0.05*max([len(Energy_p_values), len(Energy_s_values), len(Hubbard_p_values), len(Hubbard_s_values)]))):
        Energy_s = np.random.choice(Energy_s_values)
        Energy_p = np.random.choice(Energy_p_values)
        Hubbard_s = np.random.choice(Hubbard_s_values)
        Hubbard_p = np.random.choice(Hubbard_p_values)
        # Check if the values are within 0.25 of any existing values in the input_list
        input_list.append((Energy_s, Energy_p, Hubbard_s, Hubbard_p, pseudo_geom, ph_ph_IP, ph_ph_EA, ph_ph_charges, ph_ph_EV_HOMO, ph_ph_EV_LUMO, matched_atoms, el))
        # remove the entries in the list within the limit (so that differing minima are reached)
        Energy_s_values = np.delete(Energy_s_values, np.where((Energy_s_values >= Energy_s - limit) & (Energy_s_values <= Energy_s + limit)))
        Energy_p_values = np.delete(Energy_p_values, np.where((Energy_p_values >= Energy_p - limit) & (Energy_p_values <= Energy_p + limit)))
        Hubbard_s_values = np.delete(Hubbard_s_values, np.where((Hubbard_s_values >= Hubbard_s - limit) & (Hubbard_s_values <= Hubbard_s + limit)))
        Hubbard_p_values = np.delete(Hubbard_p_values, np.where((Hubbard_p_values >= Hubbard_p - limit) & (Hubbard_p_values <= Hubbard_p + limit)))

    print("Starting the optimization process:\n")

    optimal_params = []
    with multiprocessing.Pool() as pool:
        optimal_params = pool.starmap(scipy_optimise, input_list)

    total_time = time() - start
    if total_time < 60:
        print(f'\nTime taken for all calculations: {math.floor(total_time)} s\n')
        print(f'Per Optimization: {sum([elem[10] for elem in optimal_params])/len([elem[10] for elem in optimal_params]):.2f} s\n')
    elif total_time < 3600:
        print(f'\nTime taken for all calculations: {math.floor(total_time/60)} min {total_time%60:.0f} s\n')
        print(f'Per Optimization: {sum([elem[10] for elem in optimal_params])/len([elem[10] for elem in optimal_params]):.2f} s\n')
    elif total_time < 86400:
        print(f'\nTime taken for all calculations: {math.floor(total_time/3600)} h {math.floor((total_time%3600)/60)} min {total_time%60:.0f} s\n')
        print(f'Per Optimization: {sum([elem[10] for elem in optimal_params])/len([elem[10] for elem in optimal_params]):.2f} s\n')
    else:
        print(f'\nTime taken for all calculations: {math.floor(total_time/86400)} d {math.floor((total_time%86400)/3600)} h {math.floor((total_time%3600)/60)} min {total_time%60:.0f} s\n')
        print(f'Per Optimization: {sum([elem[10] for elem in optimal_params])/len([elem[10] for elem in optimal_params]):.2f} s\n')
    
    min_fct_IP = min([elem[4] for elem in optimal_params])
    min_fct_EA = min([elem[9] for elem in optimal_params])
    min_fct_IP_index = optimal_params.index([elem for elem in optimal_params if elem[4] == min_fct_IP][0])
    min_fct_EA_index = optimal_params.index([elem for elem in optimal_params if elem[9] == min_fct_EA][0])
    Energy_p_IP = optimal_params[min_fct_IP_index][0]
    Energy_s_IP = optimal_params[min_fct_IP_index][1]
    Hubbard_p_IP = optimal_params[min_fct_IP_index][2]
    Hubbard_s_IP = optimal_params[min_fct_IP_index][3]
    Energy_p_IP, Energy_s_IP, Hubbard_p_IP, Hubbard_s_IP, min_fct_IP, fct_EA, IP_diff_best, EA_diff, rmsd_IP, dot_HOMO_best, dot_LUMO = single_Hubbard(Energy_p_IP, Energy_s_IP, Hubbard_p_IP, Hubbard_s_IP, pseudo_geom, ph_ph_charges, ph_ph_IP, ph_ph_EA, matched_atoms, el, ph_ph_EV_HOMO, ph_ph_EV_LUMO)
    Energy_p_EA = optimal_params[min_fct_EA_index][5]
    Energy_s_EA = optimal_params[min_fct_EA_index][6]
    Hubbard_p_EA = optimal_params[min_fct_EA_index][7]
    Hubbard_s_EA = optimal_params[min_fct_EA_index][8]
    Energy_p_EA, Energy_s_EA, Hubbard_p_EA, Hubbard_s_EA, fct_IP, min_fct_EA, IP_diff, EA_diff_best, rmsd_EA, dot_HOMO, dot_LUMO_best = single_Hubbard(Energy_p_EA, Energy_s_EA, Hubbard_p_EA, Hubbard_s_EA, pseudo_geom, ph_ph_charges, ph_ph_IP, ph_ph_EA, matched_atoms, el, ph_ph_EV_HOMO, ph_ph_EV_LUMO)
    # Print the optimal parameters for IP and EA
    print(f"""\n\nOptimal parameters for Ionization Potential:
Energy_p = {Energy_p_IP:.4f} Ha
Energy_s = {Energy_s_IP:.4f} Ha
Hubbard_p = {Hubbard_p_IP:.4f} Ha*e^-2
Hubbard_s = {Hubbard_s_IP:.4f} Ha*e^-2
IP difference: {IP_diff_best:.6f} eV
RMSD for IP: {rmsd_IP:.6f}
Dot product of HOMO: {dot_HOMO_best:.6f}""")
    print(f"""\n\nOptimal parameters for Electron Affinity:
Energy_p = {Energy_p_EA:.4f} Ha
Energy_s = {Energy_s_EA:.4f} Ha
Hubbard_p = {Hubbard_p_EA:.4f} Ha*e^-2
Hubbard_s = {Hubbard_s_EA:.4f} Ha*e^-2
EA difference: {EA_diff_best:.6f} eV
RMSD for EA: {rmsd_EA:.6f}
Dot product of LUMO: {dot_LUMO_best:.6f}""")

    # Write the optimal parameters and differences to a file
    with open('optimal_parameters.txt', 'w') as f:
        f.write(f"""Optimal parameters for Ionization Potential:
Energy_p = {Energy_p_IP:.4f} Ha
Energy_s = {Energy_s_IP:.4f} Ha
Hubbard_p = {Hubbard_p_IP:.4f} Ha*e^-2
Hubbard_s = {Hubbard_s_IP:.4f} Ha*e^-2
IP difference: {IP_diff_best:.6f} eV
RMSD for IP: {rmsd_IP:.6f}
Dot product of HOMO: {dot_HOMO_best:.6f}\n""")
        f.write(f"""\nOptimal parameters for Electron Affinity:
Energy_p = {Energy_p_EA:.4f} Ha
Energy_s = {Energy_s_EA:.4f} Ha
Hubbard_p = {Hubbard_p_EA:.4f} Ha*e^-2
Hubbard_s = {Hubbard_s_EA:.4f} Ha*e^-2
EA difference: {EA_diff_best:.6f} eV
RMSD for EA: {rmsd_EA:.6f}
Dot product of LUMO: {dot_LUMO_best:.6f}\n""")
        f.close()

    plot_energy(optimal_params, Energy_p_IP, Energy_s_IP, Energy_p_EA, Energy_s_EA)
    plot_Hubbard(optimal_params, Hubbard_p_IP, Hubbard_s_IP, Hubbard_p_EA, Hubbard_s_EA)
    print('Plots have been generated and saved as: IP_fit_energy.png, IP_fit_Hubbard.png, EA_fit_energy.png and EA_fit_Hubbard.png')

    # Run the final calculations
    final(ph_geom, pseudo_geom, Energy_p_IP, Energy_s_IP, Energy_p_EA, Energy_s_EA, Hubbard_p_IP, Hubbard_s_IP, Hubbard_p_EA, Hubbard_s_EA, el) 

    return min_fct_IP, Energy_p_IP, Energy_s_IP, Hubbard_p_IP, Hubbard_s_IP, IP_diff_best, rmsd_IP, dot_HOMO_best, min_fct_EA, Energy_p_EA, Energy_s_EA, Hubbard_p_EA, Hubbard_s_EA, EA_diff_best, rmsd_EA, dot_LUMO_best

# define a function for the final calculations
def final_calc(geo_file: str,skf_dir: str, HOMO_flag: bool, LUMO_flag: bool) -> None:
    """
    Function which runs DFTB2 for the optimal Hubbard and Energies.

    Args:
        geo_file (str): Geometry file for the DFTB+ calculation.
        skf_dir (str): Path where the SKF files are located.
        HOMO_flag (bool): Flag to output the HOMO.
        LUMO_flag (bool): Flag to output the LUMO.

    Returns:
        None: Runs the DFTB+ calculations.
    """
    # convert the geometry file to a gen file
    geo_file = convert_to_gen(geo_file)
    # create the hsd file for the pseudo system
    hsd = """Geometry = GenFormat {
<<<  '""" + geo_file + "'\n}"
    
    hsd += """
Hamiltonian = DFTB {
    SCC = Yes
    SCCTolerance = 1.0e-5
    MaxSCCIterations = 500
    Charge = 0
    MaxAngularMomentum = {
"""
    # read the second line of the geometry file
    with open(geo_file, 'r') as f:
        next(f)
        line2 = f.readline()
        f.close()
    # split the line into a list of strings
    line2 = line2.split()
    # define a dictionary with the elements and the maximum angular momentum
    max_ang_mom = {'H': 's', 'C': 'p', 'N': 'p',
                   'O': 'p', 'F': 'p', 'Cl': 'p',
                   'Br': 'p', 'I': 'p', 'P': 'd',
                   'S': 'd', 'Se': 'd', 'Te': 'd',
                   'Z': 'p', 'Y': 'p', 'X': 'p',
                   'W': 'p', 'V': 'p'}
    # cycle through the elements in the second line of the geometry file
    for el in line2:
        # add the element and the maximum angular momentum to the hsd string with " in front and " at the end
        hsd += '        ' + el + ' = "' + max_ang_mom[el] + '"\n'
    # add the rest of the hsd string
    hsd += """    }

    SlaterKosterFiles = Type2FileNames {
        Prefix = '""" + skf_dir + """'
        Separator = ''
        Suffix = '-uncomp-c.spl'
        LowerCaseTypeName = Yes
    }
}
Options = {
    WriteDetailedXML = Yes
}
Analysis = {
    WriteEigenvectors = Yes
}"""
    # write the hsd string to a file
    with open('dftb_in.hsd', 'w') as f:
        f.write(hsd)
        f.close()
    # run dftb+ for the pseudo system
    os.system('dftb+ > dftb+.log')
    # read the band.out file
    with open('band.out', 'r') as f:
        next(f)
        lines = f.readlines()
        f.close()
    # find the HOMO and LUMO
    for i in range(len(lines)):
        if len(lines[i].split()) >= 2:
            if float(lines[i].split()[2]) == 0.0:
                LUMO = int(lines[i].split()[0])
                HOMO = int(lines[i-1].split()[0])
                if HOMO_flag:
                    HOMO_energy = float(lines[i-1].split()[1])
                    print(f"HOMO energy: {HOMO_energy} eV")
                if LUMO_flag:
                    LUMO_energy = float(lines[i].split()[1])
                    print(f"LUMO energy: {LUMO_energy} eV")
                break
    # Open the waveplot_in.hsd file
    with open('waveplot_in.hsd', 'r') as f:
        waveplot_lines = f.readlines()
        f.close()

    # Replace the 'PlottedLevels' line with 'HOMO LUMO'
    for i in range(len(waveplot_lines)):
        if 'PlottedLevels' in waveplot_lines[i]:
            waveplot_lines[i] = 'PlottedLevels = { ' + f"{HOMO} {LUMO}"+' }\n'
            break

    # Save the adjusted file under the same name
    with open('waveplot_in.hsd', 'w') as f:
        f.writelines(waveplot_lines)
        f.close()
    
    # Run waveplot
    os.system('waveplot > waveplot.log')
    return

# function for making the energy plot
def plot_energy(function: List[List[float]], Energy_p_IP: float, Energy_s_IP: float, Energy_p_EA: float, Energy_s_EA: float) -> None:
    """Function which plots the energy values of the function

    Args:
        function (List[List[float]]): list of lists containing the Energy_p, Energy_s and the function values of both the IP and EA function
        Energy_p_IP (float): Energy of the p Orbital for the Ionisation Potential 
        Energy_s_IP (float): Energy of the s Orbital for the Ionisation Potential
        Energy_p_EA (float): Energy of the p Orbital for the Electron Affinity
        Energy_s_EA (float): Energy of the s Orbital for the Electron Affinity

    Returns:
        None: but creates and saves the plots
    """
    # plot the function in 2D with the Hubbard U and Energy_s parameters on the x and y axes and the Fit value on the z axis as a color map
    x = [element[0] for element in function]
    y = [element[1] for element in function]
    z = [element[4] for element in function]
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=z, cmap='gnuplot')
    plt.xlim(min(x), max(x))
    plt.ylim(min(y), max(y))
    plt.colorbar()
    plt.xlabel('p Orbital Energy [Ha]')
    plt.ylabel('s Orbital Energy [Ha]')
    plt.title('Fit of Ionization Potential')
    # mark the minimum of the function
    plt.scatter(Energy_p_IP, Energy_s_IP, color='white', marker='x', label='Minimum')
    # additionally make a dottet line from the edge of the plot to the minimum
    #plt.plot([Energy_p_min-(Energy_p_max-Energy_p_min)/steps/2, Energy_p_IP], [Energy_s_IP, Energy_s_IP], 'w--')
    #plt.plot([Energy_p_IP, Energy_p_IP], [Energy_s_min-(Energy_s_max-Energy_s_min)/steps/2, Energy_s_IP], 'w--')
    plt.savefig('IP_fit_energy.png')
    plt.close()

    # plot the function in 2D with the Hubbard U and Energy_s parameters on the x and y axes and the Fit value on the z axis as a color map
    x = [element[5] for element in function]
    y = [element[6] for element in function]
    z = [element[9] for element in function]
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=z, cmap='gnuplot')
    plt.xlim(min(x), max(x))
    plt.ylim(min(y), max(y))
    plt.colorbar()
    plt.xlabel('p Orbital Energy [Ha]')
    plt.ylabel('s Orbital Energy [Ha]')
    plt.title('Fit of Electron Affinity')
    # mark the minimum of the function
    plt.scatter(Energy_p_EA, Energy_s_EA, color='white', marker='x', label='Minimum')
    # additionally make a dottet line from the edge of the plot to the minimum
    #plt.plot([Energy_p_min-(Energy_p_max-Energy_p_min)/steps/2, Energy_p_EA], [Energy_s_EA, Energy_s_EA], 'w--')
    #plt.plot([Energy_p_EA, Energy_p_EA], [Energy_s_min-(Energy_s_max-Energy_s_min)/steps/2, Energy_s_EA], 'w--')
    plt.savefig('EA_fit_energy.png')
    plt.close()

# function for plotting the Hubbard values
def plot_Hubbard(function: List[List[float]], Hubbard_p_IP: float, Hubbard_s_IP: float, Hubbard_p_EA: float, Hubbard_s_EA: float) -> None:
    """Function which plots the Hubbard values of the function

    Args:
        function (List[List[float]]): list of lists containing the Hubbard_p, Hubbard_s, IP_diff values of both the IP and EA function
        Hubbard_p_IP (float): Hubbard U of the p Orbital for the Ionisation Potential
        Hubbard_s_IP (float): Hubbard U of the s Orbital for the Ionisation Potential
        Hubbard_p_EA (float): Hubbard U of the p Orbital for the Electron Affinity
        Hubbard_s_EA (float): Hubbard U of the s Orbital for the Electron Affinity
    
    Returns:
        None, but creates and saves the plots
    """
    # plot the function in 2D with the Hubbard U and Energy_s parameters on the x and y axes and the Fit value on the z axis as a color map
    x = [element[2] for element in function]
    y = [element[3] for element in function]
    z = [element[4] for element in function]
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=z, cmap='gnuplot')
    plt.xlim(min(x), max(x))
    plt.ylim(min(y), max(y))
    plt.colorbar()
    plt.xlabel('Hubbard p [Ha*e^-2]')
    plt.ylabel('Hubbard s [Ha*e^-2]')
    plt.title('Fit of Ionization Potential')
    # mark the minimum of the function
    plt.scatter(Hubbard_p_IP, Hubbard_s_IP, color='white', marker='x', label='Minimum')
    # additionally make a dottet line from the edge of the plot to the minimum
    plt.savefig('IP_fit_Hubbard.png')
    plt.close()

    # plot the function in 2D with the Hubbard U and Energy_s parameters on the x and y axes and the Fit value on the z axis as a color map
    x = [element[7] for element in function]
    y = [element[8] for element in function]
    z = [element[9] for element in function]
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=z, cmap='gnuplot')
    plt.xlim(min(x), max(x))
    plt.ylim(min(y), max(y))
    plt.colorbar()
    plt.xlabel('Hubbard p [Ha*e^-2]')
    plt.ylabel('Hubbard s [Ha*e^-2]')
    plt.title('Fit of Electron Affinity')
    # mark the minimum of the function
    plt.scatter(Hubbard_p_EA, Hubbard_s_EA, color='white', marker='x', label='Minimum')
    # additionally make a dottet line from the edge of the plot to the minimum
    plt.savefig('EA_fit_Hubbard.png')
    plt.close()

# function for the final calculations
def final(ph_geom: str, pseudo_geom: str, Energy_p_IP: float, Energy_s_IP: float, Energy_p_EA: float, Energy_s_EA: float, Hub_p_IP: float, Hub_s_IP: float, Hub_p_EA: float, Hub_s_EA: float, el: str) -> None:
    """
    Function which runs the final DFTB2 calculations and calculates the optimal parameters.

    Args:
        ph_geom (str): Name of the file containing the geometry without the pseudo atoms.
        pseudo_geom (str): Name of the file containing the geometry with the pseudo atoms.
        Energy_p_IP (float): Energy of the p Orbital for the Ionisation Potential.
        Energy_s_IP (float): Energy of the s Orbital for the Ionisation Potential.
        Energy_p_EA (float): Energy of the p Orbital for the Electron Affinity.
        Energy_s_EA (float): Energy of the s Orbital for the Electron Affinity.
        Hub_p_IP (float): Hubbard U of the p Orbital for the Ionisation Potential.
        Hub_s_IP (float): Hubbard U of the s Orbital for the Ionisation Potential.
        Hub_p_EA (float): Hubbard U of the p Orbital for the Electron Affinity.
        Hub_s_EA (float): Hubbard U of the s Orbital for the Electron Affinity.
        el (str): Element of the pseudo atom.

    Returns:
        None: Runs the DFTB+ calculations.
    """
    cwd = os.getcwd()
    # read the second column of the geometry file from the third line
    with open(ph_geom, 'r') as f:
        for i in range(2):
            next(f)
        lines = f.readlines()
        f.close()
    # filter for the unique elements in the geometry file and convert them to lower case
    column = []
    for line in lines:
        column.append(line.split()[0])
    elements = np.unique(column)
    elements = [el.lower() for el in elements]
    # copy the skf files to a new directory
    manipulate_spl_Hubbard('/path/to/original/parameters/', cwd+'/params_opt_IP/', Hub_s_IP, Hub_p_IP, Energy_p_IP, Energy_s_IP, el, elements)
    # additionally copy all other skf files from the original directory into the new directory, using shutil 
    for file in os.listdir('/path/to/original/parameters/'):
        if file.endswith('.spl') and not file in os.listdir(f'{cwd}/params_opt_IP/'):
            shutil.copyfile(f'/path/to/orirginal/parameters/{file}', f'{cwd}/params_opt_IP/{file}')
    # print output message
    print('\n The optimal parameters for the IP have been saved in params_opt_IP')
    # run dftb+ for the pseudo system
    if os.path.exists('optimal_IP'):
        shutil.rmtree('optimal_IP')
    os.makedirs('optimal_IP')
    os.chdir('optimal_IP')
    shutil.copyfile(f'../{pseudo_geom}', f'{pseudo_geom}')
    shutil.copyfile('/path/to/waveplot_in.hsd', 'waveplot_in.hsd')
    final_calc(pseudo_geom, cwd+'/params_opt_IP/', True, False)
    os.chdir('..')
    print('A calculation for the optimal IP parameters has been performed, the results are in the optimal_IP directory')

    # copy the skf files to a new directory
    manipulate_spl_Hubbard('/path/to/original/parameters/', cwd+'/params_opt_EA/', Hub_s_EA, Hub_p_EA, Energy_p_EA, Energy_s_EA, el, elements)
    # additionally copy the skf files to a new directory
    for file in os.listdir('/path/to/original/parameters/'):
        if file.endswith('.spl') and not file in os.listdir(f'{cwd}/params_opt_EA/'):
            shutil.copyfile(f'/path/to/original/parameters/{file}', f'{cwd}/params_opt_EA/{file}')
    # print output message
    print('\n The optimal parameters for the EA have been saved in params_opt_EA')
    # run dftb+ for the pseudo system
    if os.path.exists('optimal_EA'):
        shutil.rmtree('optimal_EA')
    os.makedirs('optimal_EA')
    os.chdir('optimal_EA')
    shutil.copyfile(f'../{pseudo_geom}', f'{pseudo_geom}')
    shutil.copyfile('/path/to/waveplot_in.hsd', 'waveplot_in.hsd')
    final_calc(pseudo_geom, cwd+'/params_opt_EA/', False, True)
    os.chdir('..')
    print('A calculation for the optimal EA parameters has been performed, the results are in the optimal_EA directory\n')

    # do an additional calculation using the original parameters using the ph_geom file
    os.makedirs('original')
    os.chdir('original')
    shutil.copyfile(f'../{ph_geom}', f'{ph_geom}')
    shutil.copyfile('/path/to/waveplot_in.hsd', 'waveplot_in.hsd')
    final_calc(ph_geom, '/path/to/original/parameters/', True, True)
    os.chdir('..')
    print('A calculation using the original parameters has been performed, the results are in the original directory')
    return
