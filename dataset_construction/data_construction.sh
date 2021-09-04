#!/bin/bash
source /vol/medic01/users/bh1511/_venv/tf-2.x/bin/activate
export DECORD_DUPLICATE_WARNING_THRESHOLD=1.0
python3 /homes/mx520/Desktop/data_construction.py $1


