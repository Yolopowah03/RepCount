#!/bin/bash

cd ../../.. || exit
SAPIENS_CHECKPOINT_ROOT='/datatmp2/joan/tfg_joan/sapiens/sapiens_lite_host'

MODE='torchscript'
SAPIENS_CHECKPOINT_ROOT=$SAPIENS_CHECKPOINT_ROOT/$MODE

# Directorio raíz de entrada y salida
INPUT_ROOT='/datatmp2/joan/tfg_joan/LSTM_dataset/images/deadlift'
OUTPUT_ROOT="/datatmp2/joan/tfg_joan/LSTM_dataset/labels/deadlift"

MODEL_NAME='sapiens_1b'
CHECKPOINT=/datatmp2/joan/tfg_joan/sapiens/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_1b/sapiens_1b_coco_best_coco_AP_821_torchscript.pt2

DETECTION_CONFIG_FILE='/datatmp2/joan/tfg_joan/sapiens/pose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py'
DETECTION_CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/detector/checkpoints/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth

LINE_THICKNESS=3
RADIUS=6
KPT_THRES=0.3

RUN_FILE='/datatmp2/joan/tfg_joan/sapiens/lite/demo/vis_pose.py'

JOBS_PER_GPU=1; TOTAL_GPUS=1; VALID_GPU_IDS=(0)
BATCH_SIZE=8

export TF_CPP_MIN_LOG_LEVEL=2

find "$INPUT_ROOT" -mindepth 1 -type d | while read -r INPUT; do
  REL_PATH=${INPUT#"$INPUT_ROOT"}   # ahora sí quita bien el prefijo
  OUTPUT="$OUTPUT_ROOT/$REL_PATH"
  mkdir -p "$OUTPUT"
  echo "Procesando $INPUT -> $OUTPUT"

  IMAGE_LIST="${INPUT}/image_list.txt"
  find "${INPUT}" -maxdepth 1 -type f \( -iname \*.jpg -o -iname \*.png \) | sort > "${IMAGE_LIST}"

  if [ ! -s "${IMAGE_LIST}" ]; then
    echo "⚠️  No hay imágenes en $INPUT, saltando..."
    continue
  fi

  NUM_IMAGES=$(wc -l < "${IMAGE_LIST}")
  TOTAL_JOBS=$(( (NUM_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE ))
  echo "  → ${NUM_IMAGES} imágenes distribuidas en ${TOTAL_JOBS} jobs."

  for ((i=0; i<TOTAL_JOBS; i++)); do
    TEXT_FILE="${INPUT}/image_paths_$((i+1)).txt"
    head -n $((BATCH_SIZE * (i+1))) "${IMAGE_LIST}" | tail -n ${BATCH_SIZE} > "${TEXT_FILE}"

    GPU_ID=$((i % TOTAL_GPUS))
    CUDA_VISIBLE_DEVICES=${VALID_GPU_IDS[GPU_ID]} python ${RUN_FILE} \
      ${CHECKPOINT} \
      --num_keypoints 17 \
      --det-config ${DETECTION_CONFIG_FILE} \
      --det-checkpoint ${DETECTION_CHECKPOINT} \
      --batch-size ${BATCH_SIZE} \
      --input "${TEXT_FILE}" \
      --output-root="${OUTPUT}" \
      --radius ${RADIUS} \
      --kpt-thr ${KPT_THRES}

    sleep 1
    rm "${TEXT_FILE}"
  done

  rm "${IMAGE_LIST}"
done

echo "✅ Procesamiento completo."
echo "Resultados guardados en $OUTPUT_ROOT"