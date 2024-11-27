import os
import sys
import shutil

# check if one additional argument is provided
if len(sys.argv) != 2:
    print('Usage: python3 make_params_pseudo.py <pseudo_atom>')
    sys.exit(1)

# the argument given is the ne name of the pseudo atom
pseudo_atom = sys.argv[1]

# get the current working directory
cwd = os.getcwd()

# get the list of files in the directory
files = os.listdir(cwd)

# filter the files to only include the .spl files
files = [f for f in files if f.endswith('.spl')]

# loop over the files, which contain only one c in the first two characters of the filename
for f in files:
    # get the first two characters of the filename
    first_two = f[:2]
    # if the first two characters are c
    if 'c' in first_two and not 'cc' in first_two and not pseudo_atom in first_two:
        # copy the file and replace the c with the pseudo atom
        shutil.copy(f, f[:2].replace('c', pseudo_atom)+f[2:])
        print(f[:2].replace('c', pseudo_atom)+f[2:])
    elif 'cc' in first_two:
        # read the file
        with open(f, 'r') as file:
            lines = file.readlines()
            file.close()
        # remove the last column of the first line
        lines[0] = lines[0].split()[:-1]
        # join the first line back together
        lines[0] = ' '.join(lines[0])+'\n'
        # remove the second line
        lines.pop(1)
        # the first entry in the third line (now second) should be 0.000000000000E+00
        lines[1] = lines[1].split()
        lines[1][0] = '0.000000000000E+00'
        lines[1] = '  '.join(lines[1])+'\n'
        # write the file, by replacing the first c with the pseudo atom
        with open(f[0].replace('c', pseudo_atom)+f[1:], 'w') as file:
            file.writelines(lines)
            file.close()
        print(f[0].replace('c', pseudo_atom)+f[1:])
        # additionaly save the file with the second c replaced by the pseudo atom
        with open(f[0]+f[1].replace('c', pseudo_atom)+f[2:], 'w') as file:
            file.writelines(lines)
            file.close()
        print(f[0]+f[1].replace('c', pseudo_atom)+f[2:])
