#!/bin/bash

INPATH=$1
OUTPATH="$INPATH/"
CFILESPATH="$INPATH/"
LIST_OF_FILES=$(ls $INSPATH | grep cc)

cd $INPATH
for file in $LIST_OF_FILES
do
    root -l -b -q $file
    ROOTFILE=$(echo $file | awk -F'.cc' '{print $1}').root
done
cd -
