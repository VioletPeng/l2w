#!/bin/bash
#/nas/home/npeng/Plan-and-write/languange-model/models/ROCstory_titlesepkey_story_e1000_h1500_edr0.2_hdr0.1.pt

for input in disc_train.txt valid.txt test.txt
do
    python /nas/home/npeng/Plan-and-write/language-model/pytorch_src/generate.py --train-data /nas/home/npeng/Plan-and-write/language-model/rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.train --valid-data /nas/home/npeng/Plan-and-write/language-model/rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.dev --test-data /nas/home/npeng/Plan-and-write/language-model/rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.test --checkpoint ../language-model/models/ROCstory_titlesepkey_story_e1000_h1500_edr0.2_hdr0.1.pt --cuda --temperature $2 --task cond_generate --conditional-data $1${input}.context --outf $1$input.generated_continuation 
    #python generate.py --cuda --data "$1$input" --lm "$2" --dic "$3" --gen_disc_data --beam_size 4 --out "$1$input.generated_continuation"
done
