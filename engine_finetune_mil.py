# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
from typing import Iterable, Optional

import torch
import torch.distributed as dist
from timm.data import Mixup
from timm.utils import ModelEma

import utils
from utils import adjust_learning_rate
from scipy.special import softmax
from sklearn.metrics import (
    average_precision_score, 
    accuracy_score
)
## added by haora 
import numpy as np

import torch.profiler
from mil_eval import mil_accuracy,accuracy
##########################


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, 
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    update_freq = args.update_freq
    use_amp = args.use_amp
    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # if data_iter_step > 1:
        #     break
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % update_freq == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # Reshape to (B*P, C, H, W) and (B*P,)
        B, P, C, H, W = samples.shape
        samples = samples.view(B * P, C, H, W)
        targets = targets.view(-1)

        ########### 
        if mixup_fn is not None: # 非常confusing啊
            samples, targets = mixup_fn(samples, targets)

        if use_amp: 
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss, loss_main,loss_mil = criterion(output, targets)
        else: # full precision
            output = model(samples) 
            loss, loss_main,loss_mil = criterion(output, targets)

        loss_value = loss.item()
        loss_main_value = loss_main.item()
        loss_mil_value = loss_mil.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_main=loss_main_value)
        metric_logger.update(loss_mil=loss_mil_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_main=loss_main_value, head="loss")
            log_writer.update(loss_mil=loss_mil_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()
  
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    
    

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False,criterion=None):
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for index, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = batch[0]
        target = batch[-1]
        ### Special for MIL EVAL
        if len(images.shape) == 6:
            
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            B, P, _, C, H, W = images.shape
            images = images.view(B * P, _,C, H, W)
            target = target.view(-1)
        ######################
        
        
        else:
                
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # Reshape to (B*P, C, H, W) and (B*P,)
            B, P, C, H, W = images.shape
            images = images.view(B * P, C, H, W)
            target = target.view(-1)

        
        # compute output
        if use_amp:
            with torch.cuda.amp.autocast(dytpe=torch.bfloat16):
                output = model(images)
                if isinstance(output, dict):
                    output = output['logits']
                if criterion is not None:
                    loss,_,_ = criterion(output, target)
                else:
                    loss = torch.tensor(0.)
                
        else:
            output = model(images) #[bs, num_cls]
            if isinstance(output, dict):
                output = output['logits']
            if criterion is not None:
                loss,_,_ = criterion(output, target)
            else:
                loss = torch.tensor(0.)
    
        if index == 0:
            predictions = output
            labels = target
        else:
            predictions = torch.cat((predictions, output), 0)
            labels = torch.cat((labels, target), 0)

        torch.cuda.synchronize()
        # Here i need implement MIL EVAL
        mil_acc = mil_accuracy(output,target)
        #######################################
        acc1 = accuracy(output, target) # 非常confusing,这里top5的acc有意义吗？
        batch_size = images.shape[0]
        
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['mil_acc'].update(mil_acc.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.mil_acc, losses=metric_logger.loss))

    # added by haoran
    if dist.is_initialized():
        output_ddp = [torch.zeros_like(predictions) for _ in range(utils.get_world_size())]
        dist.all_gather(output_ddp, predictions)
        labels_ddp = [torch.zeros_like(labels) for _ in range(utils.get_world_size())]
        dist.all_gather(labels_ddp, labels)
        output_all = torch.concat(output_ddp, dim=0)
        labels_all = torch.concat(labels_ddp, dim=0)

    else: 
        # fallback for single GPU
        # 这里出错了，不能这样搞
        output_all = predictions
        labels_all = labels

    # 平均的acc计算有问
    y_pred = softmax(output_all.detach().cpu().numpy(), axis=1)[:, 1]
    y_true = labels_all.detach().cpu().numpy()
    y_true = y_true.astype(int)
    

    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred) # 这个为什么是0
    # real acc: label = 0
    real_mask = y_true == 0
    real_acc = accuracy_score(y_true[real_mask], (y_pred[real_mask] > 0.5)) if np.any(real_mask) else 0.0

    # fake acc: label = 1
    fake_mask = y_true == 1
    fake_acc = accuracy_score(y_true[fake_mask], (y_pred[fake_mask] > 0.5)) if np.any(fake_mask) else 0.0
    print("real acc: {}, fake acc: {}" .format(real_acc, fake_acc))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, acc, ap
