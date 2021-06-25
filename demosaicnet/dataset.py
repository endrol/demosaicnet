"""Dataset loader for demosaicnet."""
import os

from tqdm import tqdm

import numpy as np
from imageio import imread
from torch.utils.data import Dataset as TorchDataset


from .mosaic import bayer, xtrans

__all__ = ["BAYER_MODE", "XTRANS_MODE", "Dataset",
           "TRAIN_SUBSET", "VAL_SUBSET", "TEST_SUBSET"]


BAYER_MODE = "bayer"
"""Applies a Bayer mosaic pattern."""

XTRANS_MODE = "xtrans"
"""Applies an X-Trans mosaic pattern."""

TRAIN_SUBSET = "train"
"""Loads the 'train' subset of the data."""

VAL_SUBSET = "val"
"""Loads the 'val' subset of the data."""

TEST_SUBSET = "test"
"""Loads the 'test' subset of the data."""


# add custom_dataset
# add noise specially
class custDataset(TorchDataset):
    def __init__(self, data_dir, mode=BAYER_MODE) -> None:
        super().__init__()
        if mode not in [BAYER_MODE, XTRANS_MODE]:
            raise ValueError("Dataset mode should be '%s' or '%s', got"
                             " %s" % (BAYER_MODE, XTRANS_MODE, mode))
        
        self.mode = mode
        self.files = []
        for root, _, files in os.walk(data_dir):
            for name in tqdm(files):
                self.files.append(os.path.join(root, name))
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Fetches a mosaic / demosaicked pair of images.

        Returns
            mosaic(np.array): with size [3, h, w] the mosaic data with separated color channels.
            img(np.array): with size [3, h, w] the groundtruth image.
        """
        fname = self.files[idx]
        img = np.array(imread(fname)).astype(np.float32) / (2**8-1)
        img = np.transpose(img, [2, 0, 1])

        if self.mode == BAYER_MODE:
            mosaic = bayer(img)
        else:
            mosaic = xtrans(img)

        return mosaic, img

class Dataset(TorchDataset):
    """Dataset of challenging image patches for demosaicking.

    Args:
        download(bool): if True, automatically download the dataset.
        mode(:class:`BAYER_MODE` or :class:`XTRANS_MODE`): mosaic pattern to apply to the data.
        subset(:class:`TRAIN_SUBET`, :class:`VAL_SUBSET` or :class:`TEST_SUBSET`): subset of the data to load.
    """

    def __init__(self, root, download=False,
                 mode=BAYER_MODE, subset="train"):

        super(Dataset, self).__init__()

        self.root = os.path.abspath(root)

        if subset not in [TRAIN_SUBSET, VAL_SUBSET, TEST_SUBSET]:
            raise ValueError("Dataset subet should be '%s', '%s' or '%s', got"
                             " %s" % (TRAIN_SUBSET, TEST_SUBSET, VAL_SUBSET,
                                      subset))

        if mode not in [BAYER_MODE, XTRANS_MODE]:
            raise ValueError("Dataset mode should be '%s' or '%s', got"
                             " %s" % (BAYER_MODE, XTRANS_MODE, mode))
        self.mode = mode

        listfile = os.path.join(self.root, subset, "filelist.txt")
        LOG.debug("Reading image list from %s", listfile)

        if not os.path.exists(listfile):
            if download:
                _download(self.root)
            else:
                LOG.error("Filelist %s not found", listfile)
                raise ValueError("Filelist %s not found" % listfile)
        else:
            LOG.debug("No need no download the data, filelist exists.")

        self.files = []
        with open(listfile, "r") as fid:
            for fname in fid.readlines():
                self.files.append(os.path.join(self.root, subset, fname.strip()))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Fetches a mosaic / demosaicked pair of images.

        Returns
            mosaic(np.array): with size [3, h, w] the mosaic data with separated color channels.
            img(np.array): with size [3, h, w] the groundtruth image.
        """
        fname = self.files[idx]
        img = np.array(imread(fname)).astype(np.float32) / (2**8-1)
        img = np.transpose(img, [2, 0, 1])

        if self.mode == BAYER_MODE:
            mosaic = bayer(img)
        else:
            mosaic = xtrans(img)

        return mosaic, img
