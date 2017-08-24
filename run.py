#!/usr/bin/env python2

from __future__ import division, print_function, absolute_import

import numpy as np
import sys

#DATAPATH = "/n/worker/cephfs/fhs/data/out/spt/california/20150708.1/ACSPO_V2.40b05_NPP_VIIRS_2015-01-28_0810-0820_20150206.235707.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/spt/california/20150710/ACSPO_V2.40b04_NPP_VIIRS_2015-01-27_0830-0840_20150202.225311.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/spt/california/20150712/ACSPO_V2.40b05_NPP_VIIRS_2015-02-06_0840-0850_20150208.205637.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/spt/california/20150712/ACSPO_V2.40b05_NPP_VIIRS_2015-02-08_2100-2110_20150210.232641.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/spt/california/ACSPO_V2.40b05_NPP_VIIRS_2015-01-31_1030-1040_20150208.183110.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/spt/california/ACSPO_V2.40b05_NPP_VIIRS_2015-02-08_0940-0950_20150210.205545.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/spt/data/day/SPT_ACSPO_V2.31b02_NPP_VIIRS_2014-10-30_0400-0409_20141111.005748.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/spt/data/day/SPT_ACSPO_V2.31b02_NPP_VIIRS_2014-11-04_1710-1719_20141111.224328.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2015-09-15/ACSPO_V2.40_NPP_VIIRS_2015-09-15_0910-0919_20160309.231626.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2015-09-15/ACSPO_V2.40_NPP_VIIRS_2015-09-15_1050-1059_20160309.232312.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2015-09-15/ACSPO_V2.40_NPP_VIIRS_2015-09-15_1230-1239_20160309.233004.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2015-09-15/ACSPO_V2.40_NPP_VIIRS_2015-09-15_1410-1419_20160309.233730.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2015-10-18/ACSPO_V2.40_NPP_VIIRS_2015-10-18_0020-0029_20151107.000314.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2015-10-18/ACSPO_V2.40_NPP_VIIRS_2015-10-18_0330-0339_20151107.002124.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2015-10-18/ACSPO_V2.40_NPP_VIIRS_2015-10-18_0340-0350_20151107.002225.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2015-10-18/ACSPO_V2.40_NPP_VIIRS_2015-10-18_0510-0520_20151107.003028.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2015-10-18/ACSPO_V2.40_NPP_VIIRS_2015-10-18_0510-0520_20151107.003028.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2015-10-18/ACSPO_V2.40_NPP_VIIRS_2015-10-18_1140-1149_20151107.010456.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2015-10-18/ACSPO_V2.40_NPP_VIIRS_2015-10-18_1320-1329_20151107.011410.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2015-10-18/ACSPO_V2.40_NPP_VIIRS_2015-10-18_1320-1329_20151107.011410.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2015-12/ACSPO_V2.40_NPP_VIIRS_2015-12-01_2140-2149_20151214.233445.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2015-12/ACSPO_V2.40_NPP_VIIRS_2015-12-04_1340-1350_20151214.234406.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2016-04-12/ACSPO_V2.40_NPP_VIIRS_2016-04-12_0320-0329_20160415.171346.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2016-04-12/ACSPO_V2.40_NPP_VIIRS_2016-04-12_0520-0530_20160415.172058.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2016-04-12/ACSPO_V2.40_NPP_VIIRS_2016-04-12_0130-0140_20160415.170618.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2016-05-19/ACSPO_V2.41B05_NPP_VIIRS_2016-05-19_0320-0329_20160601.193124.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2016-05-19/ACSPO_V2.41B05_NPP_VIIRS_2016-05-19_0120-0129_20160601.192357.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2016-05-19/ACSPO_V2.41B05_NPP_VIIRS_2016-05-19_0140-0150_20160601.192516.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2016-05-19/ACSPO_V2.41B05_NPP_VIIRS_2016-05-19_1540-1549_20160601.201632.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2016-05-19/ACSPO_V2.41B05_NPP_VIIRS_2016-05-19_1950-1959_20160601.203354.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2016-05-19/ACSPO_V2.41B05_NPP_VIIRS_2016-05-19_0320-0329_20160601.193124.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2016-05-19/ACSPO_V2.41B05_NPP_VIIRS_2016-05-19_0410-0419_20160601.193429.nc"
DATAPATH = "/cephfs/fhs/data/out/tmp/ACSPO_V2.41B05_NPP_VIIRS_2016-05-19_0320-0329_20160601.193124.nc"
#DATAPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2016-05-19/ACSPO_V2.41B05_NPP_VIIRS_2016-05-19_1330-1339_20160601.200826.nc"

#MARKUPPATH = "/n/worker/cephfs/fhs/data/out/sstfronts/2015-09-15/markups/ACSPO_V2.40_NPP_VIIRS_2015-09-15_0910-0919_20160309.231626_frontlabels.png"
MARKUPPATH = None

def compile():
    import os

    n = os.system("make -j")
    if n != 0:
        sys.exit(1)

def run():
    import subprocess

    cmd = ["./spt", DATAPATH]
    if MARKUPPATH is not None:
        cmd.append(MARKUPPATH)
    p = subprocess.Popen(cmd,
            stdout=subprocess.PIPE,
            stderr=sys.stdout,
            bufsize=1)
    lines = []
    for line in iter(p.stdout.readline, ""):
        print(line, end="")
        lines.append(line)
    if p.wait() != 0:
        sys.exit(1)
    return ''.join(lines)

def loadnc(filename):
    import netCDF4
    return np.array(netCDF4.Dataset(filename).variables["data"])

if __name__ == '__main__':
    compile()
    exec run()
