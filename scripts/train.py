#!/bin/env python
"""Train a demosaicking model."""
import os
import time
import argparse
import torch as th
from torch._C import device
from torch.optim import optimizer
from torch.serialization import save
from torch.utils.data import DataLoader
import numpy as np
import ttools
from ttools.modules.image_operators import crop_like
from torch.utils.tensorboard import SummaryWriter
import torchvision
import pdb
import demosaicnet
from custDataset import custDataset
import logging

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)




def save_weights(checkpoint, step, checkpoint_path, backup=False):
    if backup:
        name_path = os.path.join(checkpoint_path, "backup", f"model_{step}.pth")
        th.save(checkpoint, name_path)
        print('\n')
        print(f"saved model in backup model_{step}.pth")
    
    name_path = os.path.join(checkpoint_path, "model_newest.pth")
    th.save(checkpoint, name_path)
    print('\n')
    print(f"saved newest model in model_newest.pth in {step}")

def main(args):
    """Entrypoint to the training."""

    model_device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

    data = custDataset(args.data, mode=args.mode)

    dataloader = DataLoader(
        data, batch_size=args.batchsize,
        pin_memory=True, shuffle=True)

    model = demosaicnet.BayerDemosaick(depth=args.depth,
                                           width=args.width,
                                           pad=False)

    model.to(model_device)
    model_optimizer = th.optim.Adam(model.parameters(), lr=1e-4)
    model_loss = th.nn.MSELoss()
    model_start_steps = 0
    total_steps = int(len(dataloader) * args.epoch)

    '''if ckp available, load model'''
    if args.checkpoint:
        previous_weight = th.load(args.checkpoint)
        model_optimizer.load_state_dict(previous_weight["state_optimizer"])
        # model_loss.load_state_dict(previous_weight["state_loss"])
        model.load_state_dict(previous_weight["state_model"])
        model_start_steps = int(previous_weight["step"])
    
    start_time = time.time()
    intel_time = start_time
    current_step = model_start_steps

    # enable tensorboard writer
    writer = SummaryWriter(args.logpath+f"/train_process")

    while current_step < total_steps:
        for mosaic, target in dataloader:

            # updating steps
            if current_step > total_steps:
                save_weights
                print(f"***** Training end ******** in {time.time()- start_time}")
            current_step += 1

            rawInput = mosaic.to(model_device)
            output = model(rawInput)

            gt_target = target.to(model_device)
            gt_target = crop_like(gt_target, output)

            # back-ward
            model_optimizer.zero_grad()
            mse_loss = model_loss(output, gt_target)
            mse_loss.backward()
            model_optimizer.step()

            if (current_step+1) % args.interval/2 == 0:
                print(f"***** {current_step+1}/{total_steps}, loss: {mse_loss}, run_time: {time.time() - intel_time}****")
                intel_time = time.time()

            if (current_step+1) % args.interval == 0:
                # record loss
                writer.add_scalar("loss", mse_loss, current_step + 1)
                # pdb.set_trace()
                # record images
                pdb.set_trace()
                writer.add_image("gt_image", torchvision.utils.make_grid(gt_target.permute(0,2,3,1)[:8]), current_step+1)
                writer.add_image("generated_image", torchvision.utils.make_grid(output.permute(0,2,3,1)[:8]), current_step+1)
                writer.add_image("input_image", torchvision.utils.make_grid(rawInput.permute(0,2,3,1)[:8]), current_step+1)
                
                
                
                checkpoint_info = {
                    "step": current_step,
                    "state_model": model.state_dict,
                    "state_optimizer": model_optimizer.state_dict
                }
                save_weights(
                    checkpoint=checkpoint_info,
                    step=current_step+1,
                    checkpoint_path=args.save_model,
                    backup=((current_step+1) % (args.interval ** 2) == 0)
                    )

    writer.close()    
            
            

if __name__ == "__main__":
    parser  = argparse.ArgumentParser(description="parser for DJDD custom training")
    # parser = ttools.BasicArgumentParser()
    parser.add_argument("--data", help="dataset path", type=str, required=True)
    parser.add_argument("-ckp", "--checkpoint", help="path to checkpoint for resume train",
                        type=str, default="")
    parser.add_argument("--save_model", help="path to folder where save models", type=str,
                        default="/workspace/demosaicnet/checkpoints")
    parser.add_argument("-logpath", help="log path save tensorboard", type=str, 
                        default="/workspace/demosaicnet/log")
    # parser.add_argument("--mode", help="define the mode", type=str, default="Bayer")
    parser.add_argument("--depth", default=15,
                        help="number of net layers.")
    parser.add_argument("--width", default=64,
                        help="number of features per layer.")
    parser.add_argument("--mode", default=demosaicnet.BAYER_MODE,
                        choices=[demosaicnet.BAYER_MODE,
                                 demosaicnet.XTRANS_MODE],
                        help="number of features per layer.")
    parser.add_argument("--epoch", help="epoch number", type=int, default=15)
    parser.add_argument("--batchsize", help="batch size", type=int, default=24)
    parser.add_argument("--interval", help="intervel steps for saving weights and tensorboard",
                                type=int, default=100)
    args = parser.parse_args()
    main(args)
