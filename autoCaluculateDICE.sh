#!/bin/bash

#Input
readonly TRUE="$HOME/Desktop/data/kits19"
readonly TEXT="$HOME/Desktop/data/result/32-64-64-2/cancer/0_noscale_cross/criteria"
readonly RESULT="$HOME/Desktop/data/patch/32-64-64-2/cancer/0_noscale_cross/criteria/segmentation"
readonly PREFIX="DICE"


mkdir -p $TEXT
text="${TEXT}/${PREFIX}.txt"

echo ${TRUE}
echo $RESULT
echo $text

python3 caluculateDICE.py ${TRUE} ${RESULT} > $text
if [ $? -eq 0 ]; then
 echo "Done."
 echo ${RESULT} >> $text
 cat ${text}

else
 echo "Fail"

fi


