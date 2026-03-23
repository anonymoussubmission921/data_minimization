from turtle import pos
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import *
import traceback

from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Config import cfg
from libraray.privacy_dataloader.pahmdb_dl import ATTRS, build_loaders
from models.resnet50 import build_resnet_predictor

# Find optimal algorithms for the hardware.
torch.backends.cudnn.benchmark = True




# Training epoch.
def train_epoch(epoch, train_dataloader, priv_model, criterion, optimizer, writer, use_cuda):
    print(f'Train Epoch {epoch}')
    losses= []

    priv_model.train()

    for i, (clip, priv_label) in enumerate(train_dataloader):
        optimizer.zero_grad()

        if use_cuda:
            clip = clip.cuda()
            if not torch.is_tensor(priv_label):
                priv_label = torch.as_tensor(priv_label)
            priv_label = priv_label.cuda()
            priv_label = priv_label.max(dim=1).values.float()

        output = priv_model(clip)
        loss = criterion(output, priv_label)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        if i % 100 == 0: 
            print(f'Training Epoch {epoch}, Batch {i}, Loss: {np.mean(losses) :.5f}', flush=True)
        
    print(f'####----Training Epoch: {epoch}, Loss: {np.mean(losses):.4f}----####')
    writer.add_scalar('Training Loss', np.mean(losses), epoch)

    del loss, clip, output, priv_label

    return priv_model, np.mean(losses)



# Validation epoch.
def val_epoch(epoch, validation_dataloader, priv_model, criterion, use_cuda, writer):
    priv_model.eval()
    losses = []

    all_probs = []
    all_gt = []

    for i, (clips, priv_label) in enumerate(validation_dataloader):
        
        if use_cuda:
            clips = clips.cuda()
            if not torch.is_tensor(priv_label):
                priv_label = torch.as_tensor(priv_label)
            priv_label = priv_label.cuda()
            priv_label = priv_label.max(dim=1).values.float()

        with torch.no_grad():
            output = priv_model(clips)              
            loss = criterion(output, priv_label)   
            losses.append(loss.item())

            probs = torch.sigmoid(output)           

        all_probs.append(probs.detach().cpu().numpy())
        all_gt.append(priv_label.detach().cpu().numpy())

        if i % 100 == 0:
            print(f"Validation Epoch {epoch}, Batch {i} - Loss: {np.mean(losses):.6f}", flush=True)

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    print(f"#####----- Validation Epoch {epoch} - Loss: {mean_loss:.6f} ------#####")
    writer.add_scalar("Validation Loss", mean_loss, epoch)

    probs = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 5))
    gt = np.concatenate(all_gt, axis=0) if all_gt else np.empty((0, 5))

    pred = (probs >= 0.5).astype(int)

    prec, recall, f1, _ = precision_recall_fscore_support(gt, pred, average=None, zero_division=0)

    ap = average_precision_score(gt, probs, average=None)
    macro_ap = float(np.mean(ap)) if ap.size else float("nan")

    print(f"GT shape: {gt.shape}")
    print(f"Prob shape: {probs.shape}")
    print(f"Macro F1: {np.mean(f1):.4f}")
    print(f"Macro Prec: {np.mean(prec):.4f}")
    print(f"Macro Recall: {np.mean(recall):.4f}")
    print(f"Classwise AP: {ap}")
    print(f"Macro AP: {macro_ap:.4f}")

    writer.add_scalar("Validation/mAP", macro_ap, epoch)
    for j, v in enumerate(ap):
        writer.add_scalar(f"Validation/AP_attr{j}", float(v), epoch)

    return macro_ap
    

# Main code loop.
def train_classifier():
    use_cuda = torch.cuda.is_available()
    
    log_dir = os.path.join(cfg.log_root, cfg.run_id)
    os.makedirs(log_dir, exist_ok=True) 
    writer = SummaryWriter(log_dir=log_dir)

    base_dir = os.path.dirname(cfg.MODEL_PATH)   
    save_dir = os.path.join(base_dir, cfg.run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # Initialize privacy prediction model.
    priv_model = build_resnet_predictor(num_attrs=cfg.num_pa, pretrained=True, pool="max")
    print(f"#######--------Model Initialized with pretrained weights--------########")
    
    epoch1 = 1

    criterion = nn.BCEWithLogitsLoss()

    if torch.cuda.device_count() > 1:
        print(f'Multiple GPUS found!')
        priv_model = nn.DataParallel(priv_model)
        criterion.cuda()
        priv_model.cuda()
    else:
        print('Only 1 GPU is available')
        criterion.cuda()
        priv_model.cuda()

    optimizer = optim.Adam(priv_model.parameters(), lr=cfg.lr)
    
    
    train_dataset, validation_dataset = build_loaders(cfg.PAHMDB_data_path, 
                                               cfg.PAHMDB_privacy_json_dir, 
                                               seq_len=10, 
                                               size=(256, 256), 
                                               train_ratio=0.6, 
                                               seed=42)
    
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=cfg.BATCH_SIZE, 
                                  shuffle=True, 
                                  num_workers=cfg.NUM_WORKERS)
    
    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Train dataset steps per epoch: {len(train_dataset)/cfg.BATCH_SIZE}')


    validation_dataloader = DataLoader(validation_dataset, 
                                       batch_size=cfg.BATCH_SIZE, 
                                       shuffle=True, 
                                       num_workers=cfg.NUM_WORKERS)
    
    print(f'Validation dataset length: {len(validation_dataset)}')
    print(f'Validation dataset steps per epoch: {len(validation_dataset)/cfg.BATCH_SIZE}')


    val_array = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45] + [50 + x for x in range(100)]
    
    best_score = 0

    for epoch in range(epoch1, cfg.EPOCHS + 1):
        print()
        print(f'Epoch {epoch} started')
        print()
        start = time.time()
        try:
            priv_model, train_loss = train_epoch(epoch, train_dataloader, priv_model, criterion, optimizer, writer, use_cuda)

            if epoch in val_array:
                macro_ap = val_epoch(epoch, validation_dataloader, priv_model, criterion, use_cuda, writer)

            if macro_ap > best_score:
                best_score = macro_ap
                print('++++++++++++++++++++++++++++++')
                print(f'Epoch {epoch} is the best model till now for {cfg.run_id}!')
                print('++++++++++++++++++++++++++++++')
                
                save_file_path = os.path.join(save_dir, f'model_{epoch}_loss_{macro_ap:.6f}.pth')
                states = {
                    'epoch': epoch + 1,
                    'priv_model_state_dict': priv_model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(states, save_file_path)

            # else:
            save_dir = os.path.join(base_dir, cfg.run_id)
            save_file_path = os.path.join(save_dir, 'model_temp.pth')
            states = {
                'epoch': epoch + 1,
                'priv_model_state_dict': priv_model.state_dict(),
                'optimizer': optimizer.state_dict()
                }
            torch.save(states, save_file_path)
        except:
            print(f'Epoch {epoch} failed.')
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)
            continue
        taken = time.time()-start
        print(f'Time taken for Epoch-{epoch} is {taken}')


if __name__ == '__main__':

    train_classifier()
