#!/bin/bash
MODEL_PATH="/path-to-model-output-dir/checkpoint-1031"
EXPERIMENT_NAME="UNIQUE-ID"








N_TASKS_PER_GPU=1

N_TASKS_PER_GPU=2

USE_UNIFIED_PROMPT=1 # 只能是1/0

# 参数解析
if [ $# -lt 1 ]; then
    echo "请传入要使用的GPU序号，比如: bash $0 2,3,6,7 [1]"
    exit 1
fi

GPU_STR="$1"   # 需要的gpu号列表字符串如"2,3,6,7"

# 判断debug模式
is_debug=0
if [ $# -ge 2 ]; then
    if [[ "$2" =~ ^[0-9]+$ ]] && [[ "$2" -eq 1 ]]; then
        is_debug=1
    else
        echo "第二个参数如设置应为数字1（debug），否则不要传。"
        exit 1
    fi
fi

# 转为数组：IFS 逗号分割
IFS=',' read -ra GPU_LIST <<< "$GPU_STR"
CHUNKS=${#GPU_LIST[@]}
TOTAL_TASKS=$((CHUNKS * N_TASKS_PER_GPU))

CONFIG_PATH="ovon/configs/ovon_citywalker_front_view_only.yaml"
SAVE_PATH=${MODEL_PATH}/${EXPERIMENT_NAME}

current_date_string=$(date "+%Y%m%d-%H%M%S")

if [[ "$is_debug" -eq 1 ]]; then
    SAVE_PATH="./result_"${current_date_string}
    echo ''
fi












echo ">> SAVE_PATH: $SAVE_PATH"
mkdir -p $SAVE_PATH
echo "将使用${CHUNKS}张卡：${GPU_LIST[*]}"
echo "每张卡跑${N_TASKS_PER_GPU}个任务, 共${TOTAL_TASKS}个任务"

for IDX in $(seq 0 $((TOTAL_TASKS-1))); do
    GPU_IDX=$((IDX % CHUNKS))         # 第GPU_IDX号卡
    GPU_ID=${GPU_LIST[$GPU_IDX]}      # 得到实际显卡号
    if [[ "$is_debug" -eq 1 ]]; then
        echo "等于1（debug模式）"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python -u run_qwen_ovon.py \
            --exp-config $CONFIG_PATH \
            --split-num $TOTAL_TASKS \
            --split-id $IDX \
            --model-path $MODEL_PATH \
            --result-path $SAVE_PATH \
            --use-unified-prompt ${USE_UNIFIED_PROMPT}
    else
        echo "不等于1（多卡并行）"
        CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python -u run_qwen_ovon.py \
            --exp-config $CONFIG_PATH \
            --split-num $TOTAL_TASKS \
            --split-id $IDX \
            --model-path $MODEL_PATH \
            --result-path $SAVE_PATH \
            --use-unified-prompt ${USE_UNIFIED_PROMPT} \
            > "${EXPERIMENT_NAME}_${current_date_string}_log_${IDX}.txt" 2>&1 &
    fi
done

