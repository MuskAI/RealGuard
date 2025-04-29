#!/bin/bash

MAX_JOBS=1  # 并行的最大数量，可修改为你想要的数，比如 2、4、8
PIDS=()     # 用于保存后台子进程 PID

# 捕获 Ctrl+C 信号并杀掉所有后台任务
cleanup() {
    echo -e "\n🛑 Ctrl+C detected. Terminating running jobs..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait
    exit 1
}
trap cleanup SIGINT

for ((i=0; i<=2; i++))
do
    CKPT=/raid5/chr/AIGCD/AIDE/results/cnnspot-sd14-mil_fix-res50-rgb/checkpoint-$i.pth
    if [ -f "$CKPT" ]; then
        echo "🚀 Launching evaluation for checkpoint-$i.pth ..."
        CUDA_VISIBLE_DEVICES=0 python eval_all.py \
            --model PMIL \
            --batch_size 1 \
            --resnet_path "$CKPT" \
            --output_dir /raid5/chr/AIGCD/AIDE/eval_results/cnnspot-sd14-mil_fix-res50-rgb/ \
            --csv_file_name cnnspot-sd14-mil_fix-res50-rgb-e$i \
            --no_crop \
            --data_mode cnnspot &


        pid=$!
        PIDS+=($pid)

        # 控制最多只能有 $MAX_JOBS 个并行任务
        while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
            sleep 1
        done
    else
        echo "❌ Missing: $CKPT"
    fi
done

# 等待所有任务完成
wait
echo "✅ All evaluations finished."