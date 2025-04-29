#!/bin/bash

MAX_JOBS=1  # 并行的最大数量，可修改为你想要的数，比如 2、4、8

# 捕获 Ctrl+C（SIGINT）或 kill（SIGTERM）时终止所有子进程
trap "echo '⛔ Interrupted. Killing all subprocesses...'; kill 0; exit" SIGINT SIGTERM

for ((i=0; i<=0; i++))
do
    CKPT=/raid5/chr/AIGCD/AIDE/results/cnnspot-sd14-mil_fix-res50-rgb/checkpoint-$i.pth
    if [ -f "$CKPT" ]; then
        echo "Launching evaluation for checkpoint-$i.pth ..."
        CUDA_VISIBLE_DEVICES=0 python eval_all.py \
            --model PMIL \
            --batch_size 1 \
            --mil_eval_mode only_eval \
            --resnet_path "$CKPT" \
            --output_dir /raid5/chr/AIGCD/AIDE/eval_results/cnnspot-sd14-mil_fix-res50-rgb/ \
            --csv_file_name cnnspot-sd14-mil_fix-res50-rgb-e$i \
            --data_mode mil &

        # 控制最多只能有 $MAX_JOBS 个并行任务
        while (( $(jobs -r | wc -l) >= MAX_JOBS ))
        do 
            sleep 1
        done
    else
        echo "❌ Missing: $CKPT"
    fi
done

wait
echo "✅ All evaluations finished."