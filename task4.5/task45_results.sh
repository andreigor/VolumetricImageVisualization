#!/bin/bash
DATA="../data/brain.scn"
LABEL="../data/brain_label.scn"
OUT_BASENAME="out.png"
OUT_DIR="task45_report"
rm -rf "${OUT_DIR}"
mkdir "${OUT_DIR}"

for depth in 2 10 18 24; do
    ./curvilinear_cut $DATA $LABEL 180 0 $depth "${OUT_DIR}/a_${depth}_${OUT_BASENAME}"
done

for depth in 2 10 18 24; do
    ./curvilinear_cut $DATA $LABEL 90 0 $depth "${OUT_DIR}/b_${depth}_${OUT_BASENAME}"
done
