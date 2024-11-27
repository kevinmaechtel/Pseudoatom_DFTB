import os
from modules_grad import run_grad
import numpy as np
import sys
import datetime

# define the range of the Hubbard and orbital Energy parameters
Energy_p_min = -5.0
Energy_p_max = 5.0
Energy_s_min = -5.0
Energy_s_max = 5.0
Hubbard_p_min = 0.0001
Hubbard_p_max = 5.0001
Hubbard_s_min = 0.0001
Hubbard_s_max = 5.0001
num_points = 100000

# check if the correct number of arguments is given
if len(sys.argv) != 4:
    print(f"Usage: python3 {sys.argv[0]} <original_file> <pseudo_file> <pseudo_symbol>")
    print('Please provide the path to the original and pseudo geometry files')
    print('The pseudo atom will one electron in the s and no electrons in the p orbital.')
    exit()
ph_geom = sys.argv[1]
pseudo_geom = sys.argv[2]
el = sys.argv[3]
# get the current working directory
cwd = os.getcwd()
# check if the file exists
if not os.path.exists(ph_geom) or not os.path.exists(pseudo_geom):
    print('The  geometry files do not exist')
    exit()
# check if the file exists
if not os.path.exists(pseudo_geom):
    print('The file does not exist')
    exit()

print(f'Starting at: {datetime.datetime.now()}\n')

# Create a list for all of the parameters with 100000 entries in each
Energy_p_values = np.linspace(Energy_p_min, Energy_p_max, num_points)
Energy_s_values = np.linspace(Energy_s_min, Energy_s_max, num_points)
Hubbard_p_values = np.linspace(Hubbard_p_min, Hubbard_p_max, num_points)
Hubbard_s_values = np.linspace(Hubbard_s_min, Hubbard_s_max, num_points)

# Run the gradient optimization
min_fct_IP, Energy_p_IP, Energy_s_IP, Hubbard_p_IP, Hubbard_s_IP, IP_diff_best, rmsd_IP, dot_HOMO_best, min_fct_EA, Energy_p_EA, Energy_s_EA, Hubbard_p_EA, Hubbard_s_EA, EA_diff_best, rmsd_EA, dot_LUMO_best = run_grad(Energy_p_values, Energy_s_values, Hubbard_p_values, Hubbard_s_values, ph_geom, pseudo_geom, el)

print(f'Fininshed with all Calculations at: {datetime.datetime.now()}')
