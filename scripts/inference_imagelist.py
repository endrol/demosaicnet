#!/usr/bin/env python
"""Demo script on using demosaicnet for inference."""

import os
import pdb
from numpy.core.fromnumeric import size
from pkg_resources import resource_filename

import argparse
import numpy as np
import torch as th
import imageio
from tqdm import tqdm
import demosaicnet

# TODO add noise
_TEST_INPUT = resource_filename("demosaicnet", "data/test_input.png")


def main(args):
    output = "/workspace/demosaicnet/modeloutput_uniform_distribution"
    print(f"output to {output}")
    bayer = demosaicnet.BayerDemosaick()
    xtrans = demosaicnet.XTransDemosaick()

    imageList = []
    for root, _, files in os.walk(args.input):
        for name in tqdm(files):
            for noise in [0, 5, 10, 15]:

                # Load some ground-truth image
                gt = imageio.imread(os.path.join(root, name)).astype(np.float32) / 255.0
                gt = np.array(gt)

                h, w, _ = gt.shape

                # Network expects channel first
                gt = np.transpose(gt, [2, 0, 1])
                mosaicked = demosaicnet.bayer(gt)

                # Run the model (expects batch as first dimension)
                # add noise
                # pdb.set_trace()
                # mosaicked = mosaicked + np.random.normal(loc=0.0, scale=noise/100, size=mosaicked.shape)
                # mosaicked = mosaicked.astype(np.float32)
                mosaicked = th.from_numpy(mosaicked)
                mosaicked = mosaicked + th.randn(mosaicked.size()).uniform_(0, 1.) * (noise/100) + 0.0
                mosaicked = mosaicked.unsqueeze(0)


                with th.no_grad():  # inference only
                    out = bayer(mosaicked).squeeze(0).cpu().numpy()
                    out = np.clip(out, 0, 1)
                print("done")

                imageio.imsave(
                    os.path.join(output, f"{name}_{noise}_output.png"), np.transpose(out, [1, 2, 0])
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=_TEST_INPUT,
        help="test input, uses the default test input provided if no argument.",
    )
    args = parser.parse_args()
    main(args)
