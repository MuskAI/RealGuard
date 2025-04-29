import os
import pandas as pd
import matplotlib.pyplot as plt
import re

def collect_and_plot(csv_dir):
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

    def extract_epoch(filename):
        match = re.search(r'e(\d+)', filename)
        return int(match.group(1)) if match else -1

    csv_files.sort(key=extract_epoch)

    all_data = []

    for filename in csv_files:
        filepath = os.path.join(csv_dir, filename)
        epoch = extract_epoch(filename)

        df = pd.read_csv(filepath, skiprows=1)  # 跳过注释行
        df['epoch'] = epoch
        all_data.append(df)

    # 合并所有结果
    full_df = pd.concat(all_data)

    # 获取文件夹名称用于命名
    dir_name = os.path.basename(os.path.normpath(csv_dir))

    # Accuracy matrix
    acc_df = full_df.pivot(index='testset', columns='epoch', values='accuracy')
    acc_df = acc_df.sort_index(axis=1)
    acc_save_path = os.path.join(csv_dir, f"{dir_name}_accuracy_matrix.csv")
    acc_df.to_csv(acc_save_path)
    print(f"✅ Accuracy matrix 已保存至: {acc_save_path}")

    # Avg Precision matrix
    ap_df = full_df.pivot(index='testset', columns='epoch', values='avg precision')
    ap_df = ap_df.sort_index(axis=1)
    ap_save_path = os.path.join(csv_dir, f"{dir_name}_avg_precision_matrix.csv")
    ap_df.to_csv(ap_save_path)
    print(f"✅ Avg Precision matrix 已保存至: {ap_save_path}")

    # 绘制 Accuracy 折线图
    plt.figure(figsize=(14, 8))

    # 多种 marker 自动轮换
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '<', '>', '1', '2', '3', '4', '|', '_']
    marker_cycle = (markers * ((len(acc_df.index) // len(markers)) + 1))[:len(acc_df.index)]

    for testset, marker in zip(acc_df.index, marker_cycle):
        plt.plot(acc_df.columns, acc_df.loc[testset], marker=marker, label=testset)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Testset Accuracy over Epochs ({dir_name})")
    plt.xticks(acc_df.columns)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Testset Accuracy over Epochs ({dir_name})")
    plt.xticks(acc_df.columns)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(csv_dir, f"{dir_name}_accuracy.png"))

collect_and_plot('/raid5/chr/AIGCD/AIDE/eval_results/cnnspot-sd14-res50-rgb')