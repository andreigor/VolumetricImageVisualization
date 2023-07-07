#!/bin/bash
X_0=70
Y_0=140
Z_0=110
DATA="../data/brain.scn"
LABEL="../data/brain_label.scn"
OUT_BASENAME="out.png"
OUT_DIR="task5_report"
# rm -rf "${OUT_DIR}"
# mkdir "${OUT_DIR}"
# for tilt in $(seq 0 30 90); do
#     for spin in $(seq 0 -30 -90); do
#         ./surface_rendering $DATA $LABEL $tilt $spin "${OUT_DIR}/a_${tilt}_${spin}_${OUT_BASENAME}" "1 1 1" "${X_0} ${Y_0} ${Z_0}"
#     done
# done

# for tilt in $(seq 90 30 180); do
#     for spin in 0; do
#         ./surface_rendering $DATA $LABEL $tilt $spin "${OUT_DIR}/b_${tilt}_${spin}_${OUT_BASENAME}" "1 1 1" "${X_0} ${Y_0} ${Z_0}"
#     done
# done

for tilt in $(seq -90 30 90); do
        ./surface_rendering $DATA $LABEL 180 $tilt "${OUT_DIR}/c_${tilt}_${tilt}_${OUT_BASENAME}" "1 1 1" "${X_0} ${Y_0} ${Z_0}"
done

rm task5_report/sagital*
rm task5_report/coronal*
rm task5_report/axial*

