#!/bin/bash

INPATH=$1
OUTPATH="$INPATH/rootfiles/"
CFILESPATH="$INPATH/cfiles"
LIST_OF_FILES=$(ls $CFILESPATH | grep cc)

cd $INPATH
for file in $LIST_OF_FILES
do
    root -l -b -q $file
    ROOTFILE=$(echo $file | awk -F'.cc' '{print $1}').root
    mv $ROOTFILE $OUTPATH
done
cd -
