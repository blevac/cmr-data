import h5py
import numpy as np
from matplotlib import pyplot as plt
import hdf5storage
import argparse
from tqdm import tqdm
import sigpy as sp
import torch
from sigpy.mri.app import EspiritCalib, SenseRecon
import os


# read in  arguments
parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default='/csiNAS3/mridata/CMRxRecon/TrainingSet/P001/cine_lax_ks.mat', help='path to the kspace data')
parser.add_argument('--out_file', type=str, default='test.pt', help='file name')
args = parser.parse_args()


# read in arguments
file_name = args.file_name
calib_name = file_name.replace('ks.mat','calib.mat')
hf_m = h5py.File(file_name)
hf_calib = h5py.File(calib_name)
# print(hf_m.keys())
# print(hf_calib.keys())
# here to load the kspace data
newvalue = hf_m['Recon_ks']
fullmulti = newvalue["real"] + 1j*newvalue["imag"]
[nframe, nslice, ncoil, ny, nx] = fullmulti.shape
newvalue_calib = hf_calib['Calib']
fullcalib = newvalue_calib["real"] + 1j*newvalue_calib["imag"]
fullcalib = sp.resize(fullcalib, fullmulti.shape)

espirit_maps_log = np.zeros((nframe,nslice,ncoil,ny,nx), dtype=np.complex64)

# espirit calibration

for time in tqdm(range(fullmulti.shape[0])):
    for slice in tqdm(range(fullmulti.shape[1])):
        espirit_maps = EspiritCalib(fullcalib[time,slice,...], calib_width=24, kernel_width=6, thresh=0.02, crop=0.9, show_pbar=False).run()
        espirit_maps_log[time,slice] = espirit_maps

    

file_name_out = '/csiNAS3/mridata/CMRxRecon/TrainingSet/sens_maps/' + args.out_file
dict = {'sens_maps': espirit_maps_log}
torch.save(dict, file_name_out)