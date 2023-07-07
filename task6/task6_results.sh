#!/bin/bash
X_0=70
Y_0=140
Z_0=110
DATA="../data/thorax.scn"
LABEL="../data/thorax_label.nii.gz"
OUT_BASENAME="out.png"
OUT_DIR="task6_report"
# rm -rf "${OUT_DIR}"
# mkdir "${OUT_DIR}"

# ./transparent_surface_rendering $DATA $LABEL 90 0 "${OUT_DIR}/a_1_${OUT_BASENAME}.png" 1 "1 1 1" "1 0 0" "${X_0} ${Y_0} ${Z_0}"
# ./transparent_surface_rendering $DATA $LABEL 90 0 "${OUT_DIR}/a_08_${OUT_BASENAME}.png" 1 "1 0.8 0.8" "1 1 1" "${X_0} ${Y_0} ${Z_0}"
# ./transparent_surface_rendering $DATA $LABEL 90 0 "${OUT_DIR}/a_06_${OUT_BASENAME}.png" 1 "1 0.6 0.6" "1 1 1" "${X_0} ${Y_0} ${Z_0}"
# ./transparent_surface_rendering $DATA $LABEL 90 0 "${OUT_DIR}/a_04_${OUT_BASENAME}.png" 1 "1 0.4 0.4" "1 1 1" "${X_0} ${Y_0} ${Z_0}"
# ./transparent_surface_rendering $DATA $LABEL 90 0 "${OUT_DIR}/a_02_${OUT_BASENAME}.png" 1 "1 0.2 0.2" "1 1 1" "${X_0} ${Y_0} ${Z_0}"
# ./transparent_surface_rendering $DATA $LABEL 90 0 "${OUT_DIR}/a_0_${OUT_BASENAME}.png" 1 "1 1 1" "0 0 1" "${X_0} ${Y_0} ${Z_0}"


for tilt in 90; do
    for spin in $(seq 0 30 90); do
        ./transparent_surface_rendering $DATA $LABEL $tilt $spin "${OUT_DIR}/scene_${tilt}_${spin}_${OUT_BASENAME}" 0 "0.6 0.6 0.6" "1 1 1" "${X_0} ${Y_0} ${Z_0}"
    done
done

for tilt in 90; do
    for spin in $(seq 0 30 90); do
        ./transparent_surface_rendering $DATA $LABEL $tilt $spin "${OUT_DIR}/object_${tilt}_${spin}_${OUT_BASENAME}" 1 "0.6 0.6 0.6" "1 1 1" "${X_0} ${Y_0} ${Z_0}"
    done
done

# for tilt in $(seq 90 30 180); do
#     for spin in 0; do
#         ./surface_rendering $DATA $LABEL $tilt $spin "${OUT_DIR}/b_${tilt}_${spin}_${OUT_BASENAME}" "1 1 1" "${X_0} ${Y_0} ${Z_0}"
#     done
# done

# for tilt in $(seq -90 30 90); do
#         ./surface_rendering $DATA $LABEL 180 $tilt "${OUT_DIR}/c_${tilt}_${tilt}_${OUT_BASENAME}" "1 1 1" "${X_0} ${Y_0} ${Z_0}"
# done

# rm task5_report/sagital*
# rm task5_report/coronal*
# rm task5_report/axial*

