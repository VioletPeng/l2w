#!/bin/bash

for input in small_test.txt #disc_train.txt valid.txt test.txt
do
    python generate.py --cuda --data "$1$input" --lm "$2" --dic "$3" --beam_size 4 --out "$1$input.generated_continuation" --print --both --gen_disc_data
done
