#!/usr/bin/env python2

from __future__ import division, print_function, absolute_import

import numpy as np
import glob
import os.path
import sys
import netCDF4
import skimage.measure

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

RADIUS = 200
LAND_MASK = (1<<2)

def getcrop(row, col, shape):
    r0 = row-RADIUS
    rn = row+RADIUS
    c0 = col-RADIUS
    cn = col+RADIUS

    if r0 < 0:
        rn -= r0
        r0 = 0
    if c0 < 0:
        cn -= c0
        c0 = 0
    if rn > shape[0]:
        r0 -= rn-shape[0]
        rn = shape[0]
    if cn > shape[1]:
        c0 -= cn-shape[1]
        cn = shape[1]

    return np.s_[r0:rn, c0:cn]

def createsstfig(sst, fronts, landmask, cloudmask, filename, extent=None, vmin=None, vmax=None):
    fig, ax = plt.subplots()
    cax = ax.imshow(sst, interpolation='nearest', extent=extent, cmap='jet', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(cax)

    overlay = np.zeros(sst.shape+(4,), dtype='f8')
    overlay[fronts,:] = (0, 0, 0, 1)
    overlay[cloudmask,:] = (0, 0, 0, 1)      # cloud or invalid
    overlay[landmask,:] = (92/255.0, 51/255.0, 23/255.0, 1)
    ax.imshow(overlay, interpolation='nearest', extent=extent)
    plt.savefig(filename, dpi=100)
    plt.close()

def main():
    if len(sys.argv) != 2:
        print("usage: %s <dir>" % (sys.argv[0],))
        sys.exit(1)

    ncfiles = sorted(glob.glob(os.path.join(sys.argv[1], "*.nc")))

    for k, path in enumerate(ncfiles):
        filename = os.path.basename(path)
        print(k, filename)
        f = netCDF4.Dataset(path)
        try:
            acspo = np.array(f.variables["acspo_mask"])
            sptmask = np.array(f.variables["spt_mask"])
            sst = np.array(f.variables["sst_regression"]).astype('f8')
        except KeyError as e:
            print("reading failed:", e)
            continue

        labels, nlabels = skimage.measure.label((acspo>>6) != (sptmask&3),
            return_num=True, neighbors=4)
        orderedlabels = np.arange(1, nlabels)
        sizes = np.bincount(np.ravel(labels))[1:]
        ind = np.argsort(sizes)[::-1]
        sizes = sizes[ind]
        orderedlabels = orderedlabels[ind]
        print("cluster sizes:", sizes)

        for i, lab in enumerate(orderedlabels[:3]):
            rows, cols = np.where(labels == lab)
            r = int(np.round(np.mean(rows)))
            c = int(np.round(np.mean(cols)))
            print("center: (%d, %d)" % (r, c))

            ind = getcrop(r, c, sst.shape)
            shape = sst[ind].shape
            extent = [ind[1].start, ind[1].stop, ind[0].start, ind[0].stop]

            sstcrop = sst[ind]
            sptcloud = (sptmask[ind]&3) != 0
            acspocloud = (acspo[ind]>>6) != 0
            landmask = (acspo[ind]&LAND_MASK) != 0
            fronts = (sptmask[ind]>>2) != 0

            vmin = 270
            vmax = 300
            if np.any(~sptcloud):
                vmin = np.nanmin(sstcrop[~sptcloud])
                vmax = np.nanmax(sstcrop[~sptcloud])

            createsstfig(sstcrop, fronts, landmask, acspocloud, "%s_%d_acspo.png" % (filename[:-3], i),
                extent=extent, vmin=vmin, vmax=vmax)
            createsstfig(sstcrop, fronts, landmask, sptcloud, "%s_%d_spt.png" % (filename[:-3], i),
                extent=extent, vmin=vmin, vmax=vmax)

        #fig, ax = plt.subplots()
        #cax = ax.imshow((acspo>>6) != (sptmask&3), interpolation='nearest')
        #cbar = fig.colorbar(cax)
        #plt.savefig(filename[:-3] + "_diff.png", dpi=300, bbox_inches='tight')
        #plt.close()
        f.close()


if __name__ == '__main__':
    main()
