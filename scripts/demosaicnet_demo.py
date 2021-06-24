#!/usr/bin/env python
"""Demo script on using demosaicnet for inference."""

import os
from pkg_resources import resource_filename
import torchvision.transforms as transforms
import argparse
import numpy as np
import torch as th
import imageio
from PIL import Image

import demosaicnet

_TEST_INPUT = resource_filename("demosaicnet", 'data/test_input.png')
normMean = [0.5, 0.5, 0.5]
normStd = [0.5, 0.5, 0.5]

class AddGaussianNoise(object):
    def __init__(self, noiseLevel):
        self.var = 0.1
        self.mean = 0.0
        self.noiseLevel = noiseLevel
        
    def __call__(self, tensor):
        sigma = self.noiseLevel/100.
        noisyTensor = tensor + th.randn(tensor.size()).uniform_(0, 1.) * sigma  + self.mean
        return noisyTensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.var)

def main(args):
  print("Running demosaicnet demo on {}, outputing to {}".format(_TEST_INPUT, args.output))
  bayer = demosaicnet.BayerDemosaick()
  xtrans = demosaicnet.XTransDemosaick()

  # Load some ground-truth image
  gt = imageio.imread(args.input).astype(np.float32) / 255.0
  gt = np.array(gt)

  h, w, _ = gt.shape

  # Make the image size a multiple of 6 (for xtrans pattern)
  gt = gt[:6*(h//6), :6*(w//6)]

  transform = transforms.Compose([ transforms.ToTensor(),
                                        transforms.Normalize(normMean, normStd),
                                        AddGaussianNoise(noiseLevel=15)])

  # Network expects channel first
  gt = np.transpose(gt, [2, 0, 1])
  mosaicked = demosaicnet.bayer(gt)
  xmosaicked = demosaicnet.xtrans(gt)

  # Run the model (expects batch as first dimension)
  # mosaicked = th.from_numpy(mosaicked).unsqueeze(0)
  # xmosaicked = th.from_numpy(xmosaicked).unsqueeze(0)

  mosaicked = Image.fromarray(mosaicked.astype(np.uint8))
  xmosaicked = Image.fromarray(xmosaicked.astype(np.uint8))
  mosaicked = transform(mosaicked).unsqueeze(0)
  xmosaicked = transform(xmosaicked).unsqueeze(0)

  with th.no_grad():  # inference only
    out = bayer(mosaicked).squeeze(0).cpu().numpy()
    out = np.clip(out, 0, 1)
    xout = xtrans(xmosaicked).squeeze(0).cpu().numpy()
    xout = np.clip(xout, 0, 1)
  print("done")

  os.makedirs(args.output, exist_ok=True)
  output = args.output

  imageio.imsave(os.path.join(output, "bayer_mosaick.png"), mosaicked.squeeze(0).permute([1, 2, 0]))
  imageio.imsave(os.path.join(output, "bayer_result.png"), np.transpose(out, [1, 2, 0]))
  imageio.imsave(os.path.join(output, "xtrans_mosaick.png"), xmosaicked.squeeze(0).permute([1, 2, 0]))
  imageio.imsave(os.path.join(output, "xtrans_result.png"), np.transpose(xout, [1, 2, 0]))

  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("output", help="output directory")
  parser.add_argument("--input", default=_TEST_INPUT, help="test input, uses the default test input provided if no argument.")
  args = parser.parse_args()
  main(args)
  
