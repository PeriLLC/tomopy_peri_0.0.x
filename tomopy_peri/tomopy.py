from __future__ import absolute_import

import tomopy
import logging
import numpy as np
from tomograperi.pml_cuda import pml_cuda

# --------------------------------------------------------------------

def _pml_cuda(self, emission=True,
         iters=1, num_grid=None, beta=1,
         init_matrix=None, overwrite=True,
         channel=None):
    # Dimensions:
    num_pixels = self.data.shape[3]

    # This works with radians.
    if np.max(self.theta) > 90:  # then theta is obviously in radians.
        self.theta *= np.pi / 180

    # Pad data.
    if emission:
        data = self.apply_padding(overwrite=False, pad_val=0)
    else:
        data = self.apply_padding(overwrite=False, pad_val=1)
        data = -np.log(data)

    # Adjust center according to padding.
    if not hasattr(self, 'center'):
        self.center = self.data.shape[3] / 2
    center = self.center + (data.shape[3] - num_pixels) / 2.

    # Set default parameters.
    if num_grid is None or num_grid > num_pixels:
        num_grid = np.floor(data.shape[3] / np.sqrt(2))
    if init_matrix is None:
        init_matrix = np.ones((data.shape[2], num_grid, num_grid),
                              dtype='float32')

    # Check again.
    if not isinstance(data, np.float32):
        data = np.array(data, dtype='float32', copy=False)

    if not isinstance(self.theta, np.float32):
        theta = np.array(self.theta, dtype='float32')

    if not isinstance(center, np.float32):
        center = np.array(center, dtype='float32')

    if not isinstance(iters, np.int32):
        iters = np.array(iters, dtype='int32')

    if not isinstance(num_grid, np.int32):
        num_grid = np.array(num_grid, dtype='int32')

    if not isinstance(init_matrix, np.float32):
        init_matrix = np.array(init_matrix, dtype='float32', copy=False)

    # Initialize and perform reconstruction.
    if channel:
        data_recon = pml_cuda(data[channel,:,:,:], theta, center, num_grid, iters, beta, init_matrix)
    else:
        data_list=[]
        for channel in range(data.shape[0]):
            try:
                self.logger.info("pml_cuda: now reconstructing {:s}".format(self.channel_names[channel]))
            except:
                pass
            data_list.append(pml_cuda(data[channel,:,:,:], theta, center, num_grid, iters, beta, init_matrix))
        recon_shape = data_list[0].shape
        data_recon = np.zeros((data.shape[0], recon_shape[0], recon_shape[1], recon_shape[2]))
        for i, recon in enumerate(data_list):
            data_recon[i,:,:,:] = recon

    # Update log.
    self.logger.debug("pml_cuda: emission: " + str(emission))
    self.logger.debug("pml_cuda: iters: " + str(iters))
    self.logger.debug("pml_cuda: center: " + str(center))
    self.logger.debug("pml_cuda: num_grid: " + str(num_grid))
    self.logger.debug("pml_cuda: beta: " + str(beta))
    self.logger.info("pml_cuda [ok]")

    # Update returned values.
    if overwrite:
        self.data_recon = data_recon
    else:
        return data_recon


setattr(tomopy.xftomo.xftomo_dataset.XFTomoDataset, 'pml_cuda', _pml_cuda)
