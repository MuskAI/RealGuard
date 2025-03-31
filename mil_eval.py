
def accuracy(output, target):
    """
    Computes accuracy for binary classification, returns as Tensor.
    """
    pred = output.argmax(dim=1)
    correct = (pred == target).sum()
    accuracy = correct.float() * 100.0 / target.size(0)
    return accuracy  # 返回的是 Tensor（有 .item()）

def mil_accuracy(output, target, patch=4):
    """
    计算 MIL 多实例学习下的 accuracy。
    
    参数：
    - output (Tensor): shape 为 (batch_size * patch, 2)，表示所有 patch 的预测 logits。
    - target (Tensor): shape 为 (batch_size,)，每个图像的真实标签（0：real, 1：fake）。
    - patch (int): 每张图像包含的 patch 数量。
    
    处理流程：
    1. 将 output 重塑为 (batch_size, patch, 2)。
    2. 对每个 patch 进行 argmax 得到预测类别，得到 shape (batch_size, patch) 的 tensor。
    3. 按 MIL 的规则：如果一张图中所有 patch 的预测都是 0，则该图预测为 0（real）；否则（即至少有一个 patch 为 1）则该图预测为 1（fake）。
    4. 将图像级别的预测与 target 比较，计算总体准确率（百分比）。
    """
    batch_size = int(target.size(0) / patch)
    # 1. 将 output 重塑为 (batch_size, patch, 2)
    output_reshaped = output.view(batch_size, patch, 2)
    target_reshaped = target.view(batch_size, patch)
    target_reshaped = target_reshaped[:,1]
    # 2. 对每个 patch 得到预测类别（0 或 1）
    patch_pred = output_reshaped.argmax(dim=2)  # shape: (batch_size, patch)
    # 3. MIL 聚合：如果所有 patch 均为 0，则 image 预测为 0，否则为 1
    # 这里利用 sum：如果所有 patch 为 0，则 sum==0，否则 sum>0
    image_pred = (patch_pred.sum(dim=1) > 0).long()
    # 4. 计算准确率
    correct = (image_pred == target_reshaped).sum()
    acc = correct.float() * 100.0 / batch_size
    return acc