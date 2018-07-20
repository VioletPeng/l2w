#!/bin/bash
#/nas/home/npeng/Plan-and-write/languange-model/models/ROCstory_titlesepkey_story_e1000_h1500_edr0.2_hdr0.1.pt

for input in disc_train.txt valid.txt test.txt
do
    #python pytorch_src/generate.py --train-data /nas/home/npeng/Plan-and-write/languange-model/rocstory_data/train_all.txt --valid-data /nas/home/npeng/Plan-and-write/languange-model/rocstory_data/valid.txt --test-data /nas/home/npeng/Plan-and-write/languange-model/rocstory_data/test.txt --checkpoint $2  --cuda --outf $1$input.generated_continuation --temperature $3 --task cond_generate --conditional_data $1$input
    python generate.py --cuda --data "$1$input" --lm "$2" --dic "$3" --gen_disc_data --beam_size 4 --out "$1$input.generated_continuation"
done
