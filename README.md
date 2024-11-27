# README

This repository contains the script for optimization of the pseudoatom parameters for my Master's thesis.

## Files

- **grad_opt.py**: This script is the main python script initialising everything and running the functions in the modules_grad.py script.
- **modules_grad.py**: This script contains all functions, which are called by grad_opt.py
- **make_params_pseudo.py**: This script completes the parameter set, after it has been optimized for the pseudoatom with only the minimum necessary amount of files.
- **reorder_geometries.py**: This script reorders the geometries, so that the linker atom is at the end of its corresponding geometry file, as it is necessary for the optimization of the pseudoatom parameters.
- **run_opt.sh**: This script calls all of the pythn scripts in the correct order for the optimization of the pseudoatom parameters.
- **requirements.txt**: This file lists all the dependencies and libraries required to run the scripts.
- **README.md**: This file provides an overview of the repository and explains the purpose of each file.

## Requirements

- Python (3.11 or newer)
- DFTB+

## Usage

1. Clone the repository:
        ```
        git clone https://github.com/kevinmaechtel/Pseudoatom_DFTB
        ```
2. Navigate to the directory:
        ```
        cd Pseudoatom_DFTB
        ```
3. Install the required dependencies:
        ```
        pip install -r requirements.txt
        ```
4. Run the scripts as needed:
        ```
        bash run_opt.sh <geometry_containing_linker_atom> <geometry_containing_pseudo_atom> <name_pseudo_atom>
        ```

## Contact

For any questions or issues, please contact [kevin.maechtel.kmc@gmail.com](mailto:kevin.maechtel.kmc@gmail.com).
