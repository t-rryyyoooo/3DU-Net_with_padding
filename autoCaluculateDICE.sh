#!/bin/bash

#Input
readonly TRUE="$HOME/Desktop/data/kits19"
readonly TEXT="$HOME/Desktop/data/result"
readonly RESULT="$HOME/Desktop/data/patch/label3d/segmentation_loss"
readonly PREFIX="test"


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


