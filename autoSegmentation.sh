#!/bin/bash

#Input
readonly DATA="$HOME/Desktop/data/kits19"
readonly SAVE="$HOME/Desktop/data/patch/32-64-64-2/cancer/0_noscale_cross/criteria/segmentation"
readonly WEIGHT="$HOME/Desktop/data/modelweight/32-64-64-2/cancer/0-noscale-cross/criteria/latest.pkl"
readonly IMAGE="imaging.nii.gz"
readonly MASK="segmentation.nii.gz"
readonly SAVE_NAME="label.mha"

#NUMBERS=(001 017 020 022 043 082 094 115 120 137 173 174 205)
#NUMBERS=(019 023 054 093 096 123 127 136 141 153 188 191 201)
#NUMBERS=(173 002 068 133 155 114 090 105 112 175 183 208 029 065 157 162 141 062 031 156 189 135 020 077 000 009 198 036)
NUMBERS=(001 017 020 022 043 082 094 115 120 137 173 174 205 019 023 054 093 096 123 127 136 141 153 188 191 201)
#NUMBERS=(097 132 152 155 156 164 167 175 182 190 193 198 203 071 100 106 121 124 125 128 129 138 140 146 150 158)

echo -n GPU_ID:
read id
for number in ${NUMBERS[@]}
do


    save="${SAVE}/case_00${number}/${SAVE_NAME}"
    image="${DATA}/case_00${number}/${IMAGE}"
    mask="${DATA}/case_00${number}/${MASK}"

    echo "IMAGE:${image}"
    echo "WEIGHT:${WEIGHT}"
    echo "SAVE:${save}"
    echo "MASK:${mask}"
    echo "GPU ID: ${id}"

    python3 segmentation.py $image $WEIGHT $save --mask_path $mask -g $id 


done
