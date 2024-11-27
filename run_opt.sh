#!/bin/bash
python3 reorder_geometries.py $2 $1
python3 grad_opt.py $1 $2 $3
python3 make_params_pseudo.py $3
