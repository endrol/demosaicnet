#!/bin/env python
"""Evaluate a demosaicking model."""
import argparse
import os
from tarfile import filemode
import time
from pkg_resources import require

import torch as th
from torch.utils.data import DataLoader
import numpy as np
import ttools
from ttools.modules.image_operators import crop_like
from imageio import imread
import pdb
import pandas as pd 

import demosaicnet
LOG = ttools.get_logger(__name__)

class PSNR(th.nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()
        self.mse = th.nn.MSELoss()
    def forward(self, out, ref):
        mse = self.mse(out, ref)
        return -10*th.log10(mse)

def main(args):
    """Entrypoint to the training."""
    gt_path = args.ground_truth
    djdd_path = args.djdd
    bjdd_path = args.bjdd

    mse_fn = th.nn.MSELoss()
    psnr_fn = PSNR()

    device = "cpu"
    # if th.cuda.is_available():
    #     device = "cuda"

    pdf = pd.DataFrame(columns=["filename","imgid", "PSNR_for_DJDD", "MSE_for_DJDD", "PSNR_for_BJDD", "MSE_for_BJDD"])

    count = 0
    msedjdd = 0.0
    psnrdjdd = 0.0

    msebjdd = 0.0
    psnrbjdd = 0.0

    for root, _, files in os.walk(gt_path):
        for idx, name in enumerate(files):
            
            # djdd image
            output_djdd = np.array(imread(os.path.join(djdd_path, name+"_0_output.png"))).astype(np.float32) / (2**8-1)
            output_djdd = th.from_numpy(np.transpose(output_djdd, [2,0,1])).to(device).unsqueeze(0)

            #bjdd image
            output_bjdd = np.array(imread(os.path.join(bjdd_path, name.split('.')[0]+"_sigma_0_bayer_PIPNet.png"))).astype(np.float32) / (2**8-1)
            output_bjdd = th.from_numpy(np.transpose(output_bjdd, [2,0,1])).to(device).unsqueeze(0)

            # gt image
            target = np.array(imread(os.path.join(root, name))).astype(np.float32) / (2**8-1)
            target = th.from_numpy(np.transpose(target, [2, 0, 1])).to(device).unsqueeze(0)


            target_djdd = crop_like(target, output_djdd)
            target_bjdd = crop_like(target, output_bjdd)

            psnr_djdd = psnr_fn(output_djdd, target_djdd).item()
            mse_djdd = mse_fn(output_djdd, target_djdd).item()

            psnr_bjdd = psnr_fn(output_bjdd, target_bjdd).item()
            mse_bjdd = mse_fn(output_bjdd, target_bjdd).item()

            psnrdjdd += psnr_djdd
            msedjdd += mse_djdd
            psnrbjdd += psnr_bjdd
            msebjdd += mse_bjdd

            count += 1

            LOG.info(f"imgid: {idx}, PSNR_BJDD: {psnr_bjdd}, MSE_BJDD: {mse_bjdd}, PSNR_DJDD: {psnr_djdd}, MSE_DJDD: {mse_djdd}")
            pdf = pdf.append({
                "filename": name,
                "imgid": idx,
                "PSNR_for_DJDD": psnr_djdd,
                "MSE_for_DJDD": mse_djdd,
                "PSNR_for_BJDD": psnr_bjdd,
                "MSE_for_BJDD": mse_bjdd
            }, ignore_index=True)
            # pdb.set_trace()

    msebjdd /= count
    psnrbjdd /= count

    msedjdd /= count
    psnrdjdd /= count

    LOG.info("--------------BJDD---------------------")
    LOG.info("Average, PSNR = %.1f dB, MSE = %.5f", psnrbjdd, msebjdd)

    LOG.info("--------------DJDD---------------------")
    LOG.info("Average, PSNR = %.1f dB, MSE = %.5f", psnrdjdd, msedjdd)
    pdb.set_trace()
    pdf.to_csv("/workspace/presentation_compare.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-gt", "--ground_truth", required=True, help="root directory for the demosaicnet dataset.")
    parser.add_argument("-djdd","--djdd", required=True)
    parser.add_argument("-bjdd", "--bjdd", required=True)
    args = parser.parse_args()
    ttools.set_logger(False)
    main(args)
