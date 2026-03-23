import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

# from library.minimization_dataloader.temporal_subsampling import AvenueDataset
# from library.minimization_dataloader.temporal_subsampling_ROI import AvenueDataset
from library.Avenue_dataloader import AvenueDataset
# from models.model import ConvLSTMAE         # for grayscale input
from models.model_rgb import ConvLSTMAE     # for RGB input
# from library.ucsd_dataloader import get_training_set
# from library.ltd_dataset import LTD_dataset
from Config import cfg


######------------- Training Epoch -------------######
def train_epoch(model, train_dataloader, criterion, optimizer, device, use_cuda, writer, epoch, scaler):
    print(f"Train Epoch {epoch}")

    losses = []
    model.train()

    for i, batch in enumerate(train_dataloader):
        if isinstance(batch, (list, tuple)):
            clips = batch[0]
        else:
            clips = batch

        if use_cuda:
            clips = clips.to(device, non_blocking=True)
            # clips = clips.permute(0,1,4,2,3)  # ensure channels last for UCSD data

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_cuda):
            recon_clip = model(clips)
            loss = criterion(recon_clip, clips)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())

        if i % 100 == 0:
            print(f"Training Epoch {epoch}, Batch {i}, Loss: {np.mean(losses):.5f}", flush=True)

    epoch_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    print(f"Training Epoch: {epoch}, Loss: {epoch_loss:.4f}", flush=True)

    writer.add_scalar("Training Loss", epoch_loss, epoch)

    del clips, recon_clip, loss
    if use_cuda:
        torch.cuda.empty_cache()

    return epoch_loss


#####-----------------plot loss-------------------------#####
def plot_loss(train_loss, plot_file):
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(train_loss) + 1), train_loss, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()


def save_checkpoint(model, optimizer, epoch, train_loss_epoch, save_file_path, scaler=None):
    states = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss_epoch,
    }
    if scaler is not None:
        states["scaler_state_dict"] = scaler.state_dict()
    torch.save(states, save_file_path)


######------------- Training Epoch -------------######
def train(devices):
    use_cuda = torch.cuda.is_available()

    os.makedirs(os.path.join(cfg.logs, cfg.run_id), exist_ok=True)
    os.makedirs(os.path.join(cfg.saved_models_dir, cfg.run_id), exist_ok=True)

    writer = SummaryWriter(os.path.join(cfg.logs, cfg.run_id))

    # model = ConvLSTMAE(in_channels=1)      # for grayscale
    model = ConvLSTMAE(in_channels=3)        # for RGB
    
    # train_dataset = get_training_set(mode = cfg.mode)  # for UCSD Ped1
    # train_dataset = LTD_dataset(data_split="train",shuffle=True,data_percentage=1.0,sequence_size=10,strides=(1, 2),step=1)
    train_dataset = AvenueDataset(root_dir=cfg.Avenue_train_path, 
                                  split="training_videos", 
                                  shuffle=False,  
                                  sequence_size=10, 
                                  stride=cfg.stride, 
                                  step=cfg.step,
                                  mode=cfg.mode)

    print("Training clips:", len(train_dataset))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.batch,
                                  shuffle=True,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True,
                                  persistent_workers=True if cfg.num_workers > 0 else False,
                                  prefetch_factor=2 if cfg.num_workers > 0 else None,
                                  drop_last=True)

    criterion = nn.MSELoss()

    if use_cuda:
        device_name = f'cuda:{devices[0]}'
        device = torch.device(device_name)
        print(f'Device name is {device_name}')

        if len(devices) > 1:
            print(f"Using multiple GPUs: {devices}")
            model = model.to(device)
            model = nn.DataParallel(model, device_ids=devices)
            criterion = criterion.to(device)
        else:
            print(f"Using single GPU")
            model = model.to(device)
            criterion = criterion.to(device)
    else:
        device = torch.device("cpu")
        print("Using CPU")
        model = model.to(device)
        criterion = criterion.to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5, eps=1e-6)
    scaler = GradScaler(enabled=use_cuda)

    train_loss = []
    start_epoch = 1
    best_train_loss = float("inf")

    for epoch in range(start_epoch, cfg.epochs + 1):
        print(f"#####-------------- Epoch {epoch} started --------------#####")
        start_time = time.time()

        train_loss_epoch = train_epoch(model=model,
                                        train_dataloader=train_dataloader,
                                        criterion=criterion,
                                        optimizer=optimizer,
                                        device=device,
                                        use_cuda=use_cuda,
                                        writer=writer,
                                        epoch=epoch,
                                        scaler=scaler)

        train_loss.append(train_loss_epoch)
        writer.add_scalar("train/loss_epoch", train_loss_epoch, epoch)

        save_dir = os.path.join(cfg.saved_models_dir, cfg.run_id)

        if train_loss_epoch < best_train_loss:
            best_train_loss = train_loss_epoch
            print("++++++++++++++++++++++++++++++")
            print(f"Epoch {epoch} has the best model till now with Train Loss: {best_train_loss:.6f}")
            print("++++++++++++++++++++++++++++++")

            best_model_path = os.path.join(
                save_dir, f"model_epoch_{epoch}_loss_{best_train_loss:.6f}.pth"
            )
            save_checkpoint(model, optimizer, epoch, train_loss_epoch, best_model_path, scaler=scaler)

        temp_model_path = os.path.join(save_dir, "model_temp.pth")
        save_checkpoint(model, optimizer, epoch, train_loss_epoch, temp_model_path, scaler=scaler)

        losses_file = os.path.join(cfg.logs, cfg.run_id, "train_loss.txt")
        with open(losses_file, "w") as f:
            for loss in train_loss:
                f.write(f"{loss}\n")

        plot_file = os.path.join(cfg.logs, cfg.run_id, "train_loss_plot.png")
        plot_loss(train_loss, plot_file)

        taken = time.time() - start_time
        print(f"Time taken for Epoch-{epoch} is {taken:.2f} seconds")

    writer.close()
    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devices",
        nargs="+",
        type=int,
        default=[0],
        help="List of GPU device ids, e.g. --devices 0 or --devices 0 1"
    )
    args = parser.parse_args()

    train(args.devices)



