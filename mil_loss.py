import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchAwareCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')  # we handle reduction manually

    def forward(self, outputs, targets):
        """
        outputs: Tensor of shape (B*P, num_classes)
        targets: Tensor of shape (B*P,)
        """
        # outputs = torch.stack(outputs,dim=0)
        B_P = outputs.shape[0]
        # assert B_P == targets.shape[0], "Mismatch in batch-patch size"

        # 1. 恢复到 (B, P, ...)
        # 假设 patch 数量 P 是固定的
        P = 4  # or: compute dynamically if needed
        B = B_P // P

        outputs = outputs.view(B, P, 2)   # (B, P, C)
        targets = targets.view(B, P)                # (B, P)

        # 获取每个样本的 batch-level 标签（假设 patch 之间标签一致）
        batch_labels = targets[:, 0]                # (B,)

        # 2. 正负样本划分
        pos_mask = batch_labels == 1
        neg_mask = batch_labels == 0

        # 3. 正样本: 对 patch-level logits 做 max-pooling，再计算 CE loss
        if pos_mask.sum() > 0:
            pos_outputs = outputs[pos_mask]         # (B_pos, P, C)
            pos_targets = batch_labels[pos_mask]    # (B_pos,)

            # max over patches: (B_pos, P, C) → (B_pos, C)
            pos_outputs_max = pos_outputs.max(dim=1).values
            loss1 = self.ce_loss(pos_outputs_max, pos_targets)
        else:
            loss1 = 0.0 * outputs.sum()  # 保持计算图

        # 4. 负样本: 所有 patch 单独计算 loss 然后取 mean
        if neg_mask.sum() > 0:
            neg_outputs = outputs[neg_mask]         # (B_neg, P, C)
            neg_targets = targets[neg_mask]         # (B_neg, P)

            neg_outputs_flat = neg_outputs.view(-1,2)   # (B_neg*P, C)
            neg_targets_flat = neg_targets.view(-1)                # (B_neg*P,)

            loss2 = self.ce_loss(neg_outputs_flat, neg_targets_flat).mean()
        else:
            loss2 = 0.0 * outputs.sum()  # 保持计算图

        total_loss = loss1.mean() + loss2
        return total_loss
    
    
class DualLevelCELoss(nn.Module):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.standard_ce = nn.CrossEntropyLoss()
        self.patch_ce = PatchAwareCELoss()

    def forward(self, outputs, targets):
        loss_main = self.standard_ce(outputs, targets)
        loss_mil = self.patch_ce(outputs, targets)
        total_loss = (1 - self.alpha) * loss_main + self.alpha * loss_mil
        return total_loss,loss_main,loss_mil