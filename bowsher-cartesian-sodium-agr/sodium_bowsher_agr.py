import numpy as np
import nibabel as nib
import h5py
from pathlib import Path

import matplotlib.pyplot as plt

from scipy.optimize import fmin_l_bfgs_b
from scipy.ndimage import zoom, gaussian_filter

from nearest_neighbors import next_neighbors, nearest_neighbors, is_nearest_neighbor_of
from readout_time import setup_readout_bins, max_k_from_lookup
from cost_functions import (
    multi_echo_bowsher_cost,
    multi_echo_bowsher_grad,
    multi_echo_bowsher_cost_gamma,
)
from cost_functions import multi_echo_bowsher_grad_gamma, multi_echo_bowsher_cost_total
from utils import load_config, load_recon_config

from argparse import ArgumentParser
from datetime import datetime

import json
import pymirc.viewer as pv

parser = ArgumentParser()
parser.add_argument("--cfg", help="config json file", default="config.json")
parser.add_argument("--recon_cfg", help="config json file", default="recon_params.json")
args = parser.parse_args()

cfg = load_config(args.cfg)
recon_cfg = load_recon_config(args.recon_cfg)

niter = recon_cfg.niter
n_outer = recon_cfg.n_outer
bet_recon = recon_cfg.bet_recon
bet_gam = recon_cfg.bet_gam
nnearest = recon_cfg.nnearest
nneigh = recon_cfg.nneigh
method = recon_cfg.method
mr_name = recon_cfg.mr_name
anat_prior = not recon_cfg.no_anat_prior

# echo spacing TE2 - TE1 in ms: command line > config file > default (5 ms)
delta_t = args.delta_t if args.delta_t is not None else cfg.delta_t
print(f'using echo spacing delta_t = {delta_t} ms')
asym = 0
n = cfg.recon_shape[0]  # only used in the input file names (e.g. echo1_{n}.npy)
recon_shape_3d = tuple(int(x) for x in cfg.recon_shape)
data_shape_3d = tuple(int(x) for x in cfg.data_shape)

if nneigh not in (18, 80):
    raise ValueError(f'nneigh must be 18 or 80, got {nneigh}')

# k-space edge (in the units of the readout time model)
if cfg.k_edge is not None:
    k_edge = cfg.k_edge
elif cfg.ksp_file is not None:
    k_edge = float(max_k_from_lookup(cfg.ksp_file))
else:
    k_edge = 1.5

readout_kwargs = {} if cfg.ksp_file is None else {'lookup': cfg.ksp_file}

init_filter_sigma = recon_cfg.init_filter_sigma
low_signal_threshold = recon_cfg.low_signal_threshold

nechos = 2
gam_lower_bound = 0.001

# every value the reconstruction actually uses (after defaults, overrides and
# derived values are resolved). Written to effective_params.json in the output
# directory and as attributes of recons.h5
effective_params = {
    'run_started': datetime.now().isoformat(timespec='seconds'),
    'run_status': 'started',
    'cfg_file': str(Path(args.cfg).resolve()),
    'recon_cfg_file': str(Path(args.recon_cfg).resolve()),
    'recon_dir': str(cfg.recon_dir),
    'mr_name': mr_name,
    # acquisition / k-space model
    'delta_t_ms': delta_t,
    'nechos': nechos,
    'ksp_file': cfg.ksp_file,
    'k_edge': k_edge,
    'n_readout_bins': cfg.n_readout_bins,
    'readout_duration_ms': cfg.readout_duration_ms,
    'data_shape': data_shape_3d,
    'recon_shape': recon_shape_3d,
    'sodium_fov_mm': cfg.sodium_fov_mm,
    'init_filter_sigma': init_filter_sigma,
    'low_signal_threshold': low_signal_threshold,
    # reconstruction / prior
    'niter': niter,
    'n_outer': n_outer,
    'bet_recon': bet_recon,
    'bet_gam': bet_gam,
    'method': method,
    'anat_prior': anat_prior,
    'nneigh': nneigh,
    'nnearest': nnearest,
    'asym': asym,
    'gam_bounds': [gam_lower_bound, 1.],
}

def _to_builtin(x):
    """make numpy types json serializable"""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, tuple):
        return list(x)
    raise TypeError(f'not serializable: {type(x)}')
 
 
def save_effective_params():
    """(re)write effective_params.json - called at several stages, so that
    a run that crashes still documents what it used up to that point"""
    (odir / 'effective_params.json').write_text(
        json.dumps(effective_params, indent=2, default=_to_builtin))
 
 
print('effective parameters:', json.dumps(effective_params, indent=2, default=_to_builtin))

# %%
# load the data
# -------------

pdir = cfg.recon_dir

if anat_prior:
    odir = (
        Path(pdir)
        / f'{datetime.now().strftime("%y%m%d-%H%M%S")}__br_{bet_recon:.1E}__bg_{bet_gam:.1E}__nn_{nneigh}__ne_{nnearest}'
    )
else:
    odir = (
        Path(pdir)
        / f'{datetime.now().strftime("%y%m%d-%H%M%S")}__br_{bet_recon:.1E}__bg_{bet_gam:.1E}__no_anat'
    )

if not odir.exists():
    odir.mkdir(exist_ok=True)

# copy config files
(odir / "config.json").write_bytes(Path(args.cfg).read_bytes())
(odir / "recon_params.json").write_bytes(Path(args.recon_cfg).read_bytes())
effective_params['output_dir'] = str(odir.resolve())
save_effective_params()


t1_vol = np.load(Path(pdir) / mr_name)
sens = np.load(Path(pdir) / f"sens_{n}.npy").astype(np.complex128).view("(2,)float")
echo1 = np.load(Path(pdir) / f"echo1_{n}.npy").astype(np.complex128).view("(2,)float")
echo2 = np.load(Path(pdir) / f"echo2_{n}.npy").astype(np.complex128).view("(2,)float")

ncoils = echo1.shape[0]

signal = np.zeros((ncoils, nechos) + echo1.shape[1:])

for i in range(ncoils):
    signal[i, 0, ...] = echo1[i, ...]
    signal[i, 1, ...] = echo2[i, ...]

# %%
# calc readout times
# ------------------

if signal.shape[2:-1] != recon_shape_3d:
    raise ValueError(f'echo data have shape {signal.shape[2:-1]}, '
                     f'but recon_shape in the config is {recon_shape_3d}')

# bin k-space into shells of equal |k|, every shell gets one readout time (ms)
# (same trajectory assumed for both echos)
readout_inds, tr, read_out_img = setup_readout_bins(recon_shape_3d,
                                                    data_shape_3d,
                                                    k_edge,
                                                    cfg.n_readout_bins,
                                                    readout_kwargs,
                                                    kedge_radius=data_shape_3d[0] / 2,
                                                    readout_duration_ms=cfg.readout_duration_ms)

effective_params['readout_times_ms'] = [round(float(x), 4) for x in tr]
effective_params['kspace_fraction_acquired'] = float((read_out_img > 0).mean())
save_effective_params()

print(f'readout bins: {cfg.n_readout_bins}, '
      f'readout time of first / last bin: {tr[0]:.2f} / {tr[-1]:.2f} ms, '
      f'fraction of recon k-space grid inside acquired k-space: {(read_out_img > 0).mean():.3f}')

kmask = np.zeros(signal.shape)
for j in range(ncoils):
    for i in range(nechos):
        kmask[j, i, ..., 0] = (read_out_img > 0).astype(np.float64)
        kmask[j, i, ..., 1] = (read_out_img > 0).astype(np.float64)

# multiply signal with readout mask
signal *= kmask
abs_signal = np.linalg.norm(signal, axis=-1)

ifft = np.zeros(signal.shape)
ifft_filtered = np.zeros(signal.shape)
abs_ifft = np.zeros(signal.shape[:-1])
abs_ifft_filtered = np.zeros(signal.shape[:-1])

for j in range(ncoils):
    for i in range(nechos):
        s = signal[j, i, ...].view(dtype=np.complex128).squeeze().copy()
        ifft[j, i, ...] = np.ascontiguousarray(
            np.fft.ifftn(s, norm="ortho").view("(2,)float")
        )
        if init_filter_sigma > 0:
            ifft_filtered[j, i, ...] = np.ascontiguousarray(
                gaussian_filter(np.fft.ifftn(s, norm='ortho'),
                                init_filter_sigma).view('(2,)float'))
        else:
            ifft_filtered[j, i, ...] = ifft[j, i, ...]
        abs_ifft[j, i, ...] = np.linalg.norm(ifft[j, i, ...], axis=-1)
        abs_ifft_filtered[j, i, ...] = np.linalg.norm(ifft_filtered[j, i, ...],
                                                      axis=-1)
# %%
# set up the anatomical prior
# ---------------------------

aimg = t1_vol.copy()

if nneigh == 18:
    s = np.array(
        [
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        ]
    )
elif nneigh == 80:
    s = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
            ],
            [
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
            ],
            [
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
        ]
    )

if anat_prior:
    ninds = np.zeros((np.prod(aimg.shape), nnearest), dtype=np.uint32)
    nearest_neighbors(aimg, s, nnearest, ninds)
else:
    ninds = np.zeros((np.prod(aimg.shape), aimg.ndim), dtype=np.uint32)
    next_neighbors(aimg.shape, ninds)

ninds2 = is_nearest_neighbor_of(ninds)

# %%
# initialize variables
# --------------------

tmp1 = abs_ifft_filtered[:, 0, ...].sum(0)
tmp2 = abs_ifft_filtered[:, 1, ...].sum(0)

Gam_recon = tmp2 / tmp1
Gam_recon[tmp1 < low_signal_threshold * tmp1.max()] = 1
Gam_recon[Gam_recon > 1] = 1

recon = np.zeros(signal.shape[2:])
recon[..., 0] = tmp1

recon_shape = recon.shape
Gam_bounds = np.tile(np.array([[gam_lower_bound, 1.]]), (Gam_recon.size, 1))

# rescale signal and initial recon to get an image with max approx 1
scale_fac = 1.0 / tmp1.max()

effective_params['ncoils'] = int(ncoils)
effective_params['signal_scale_fac'] = float(scale_fac)
save_effective_params()

signal *= scale_fac
tmp1 *= scale_fac
tmp2 *= scale_fac
recon *= scale_fac

abs_recon = np.linalg.norm(recon, axis=-1)

cost = []

fig1, ax1 = plt.subplots(2, n_outer + 1, figsize=((n_outer + 1) * 3, 6))
vmax = 1.2
zc = recon_shape_3d[2] // 2  # slice shown in convergence plots
ax1[0, 0].imshow(Gam_recon[..., zc], vmin=0, vmax=1, cmap=plt.cm.Greys_r)
ax1[1, 0].imshow(abs_recon[..., zc], vmin=0, vmax=vmax, cmap=plt.cm.Greys_r)

# %% run iterative Bowsher AGR
# ----------------------------

for i in range(n_outer):

    print("LBFGS to optimize for recon")

    recon = recon.flatten()

    cb = lambda x: cost.append(
        multi_echo_bowsher_cost_total(
            x,
            recon_shape,
            signal,
            readout_inds,
            Gam_recon,
            tr,
            delta_t,
            nechos,
            kmask,
            bet_recon,
            bet_gam,
            ninds,
            method,
            sens,
            asym,
        )
    )

    res = fmin_l_bfgs_b(
        multi_echo_bowsher_cost,
        recon,
        fprime=multi_echo_bowsher_grad,
        args=(
            recon_shape,
            signal,
            readout_inds,
            Gam_recon,
            tr,
            delta_t,
            nechos,
            kmask,
            bet_recon,
            ninds,
            ninds2,
            method,
            sens,
            asym,
        ),
        callback=cb,
        maxiter=niter,
    )

    recon = res[0].reshape(recon_shape)
    print(f'  outer {i+1}/{n_outer} recon:  {res[2]["nit"]} iterations, cost {res[1]:.6e}')
    abs_recon = np.linalg.norm(recon, axis=-1)

    ax1[1, i + 1].imshow(abs_recon[..., 64], vmin=0, vmax=vmax, cmap=plt.cm.Greys_r)

    # ---------------------------------------

    print("LBFGS to optimize for gamma")

    Gam_recon = Gam_recon.flatten()

    cb = lambda x: cost.append(
        multi_echo_bowsher_cost_total(
            recon,
            recon_shape,
            signal,
            readout_inds,
            x,
            tr,
            delta_t,
            nechos,
            kmask,
            bet_recon,
            bet_gam,
            ninds,
            method,
            sens,
            asym,
        )
    )

    res = fmin_l_bfgs_b(
        multi_echo_bowsher_cost_gamma,
        Gam_recon,
        fprime=multi_echo_bowsher_grad_gamma,
        args=(
            recon_shape,
            signal,
            readout_inds,
            recon,
            tr,
            delta_t,
            nechos,
            kmask,
            bet_gam,
            ninds,
            ninds2,
            method,
            sens,
            asym,
        ),
        callback=cb,
        maxiter=niter,
        bounds=Gam_bounds,
    )

    Gam_recon = res[0].reshape(recon_shape[:-1])
    print(f'  outer {i+1}/{n_outer} gamma:  {res[2]["nit"]} iterations, cost {res[1]:.6e}')

    # reset values in low signal regions
    Gam_recon[tmp1 < low_signal_threshold * tmp1.max()] = 1

    ax1[0, i + 1].imshow(Gam_recon[..., zc], vmin=0, vmax=1, cmap=plt.cm.Greys_r)

    fig1.tight_layout()
    fig1.show()
    fig1.savefig(Path(odir) / "convergence.png")

# %%
# save the recons to HDF5
# -----------------------

output_file = Path(odir) / "recons.h5"
with h5py.File(output_file, "w") as hf:
    grp = hf.create_group("images")
    grp.create_dataset("Gam_recon", data=Gam_recon)
    grp.create_dataset("recon", data=recon)
    grp.create_dataset("abs_recon", data=abs_recon)
    grp.create_dataset("prior_image", data=aimg)
    grp.create_dataset("cost", data=cost)
    grp.create_dataset('readout_bin_img', data=read_out_img.astype(np.uint16))
    for key, val in effective_params.items():
        if key == 'run_status':
            continue
        hf.attrs[key] = 'None' if val is None else val
# %% 
# save the recons to NIFTI
# ------------------------
 
# voxel size per axis (same FOV along all axes), image centered at 0
vox_mm = np.array([cfg.sodium_fov_mm / x for x in recon_shape_3d])
output_affine_ras = np.diag(np.concatenate([vox_mm, [1.]]))
output_affine_ras[:-1, -1] = 0.5 * (-cfg.sodium_fov_mm + vox_mm)
 
nib.save(nib.Nifti1Image(abs_recon, output_affine_ras), odir  / f'AGR_{odir.name}.nii')
nib.save(nib.Nifti1Image(Gam_recon, output_affine_ras), odir  / f'GAM_{odir.name}.nii')
 
# AGR and GAM are written -> mark the run as completed
effective_params['run_status'] = 'completed'
effective_params['run_finished'] = datetime.now().isoformat(timespec='seconds')
save_effective_params()
nib.save(nib.Nifti1Image(aimg, output_affine_ras), odir  / f'T1_proton_aligned.nii')
 
# save the interpolated SOS images
#sos1 = np.abs(np.flip(np.fromfile(cfg.TE05soskw0_filename, dtype = np.complex64).reshape(cfg.data_shape).swapaxes(0,2), (1,2)))
#sos2 = np.abs(np.flip(np.fromfile(cfg.TE05soskw1_filename, dtype = np.complex64).reshape(cfg.data_shape).swapaxes(0,2), (1,2)))
 
# To allow the below to complete...
 
sos1 = np.abs(np.flip(np.fromfile(cfg.TE05c0_filename, dtype = np.complex64).reshape(cfg.data_shape).swapaxes(0,2), (1,2)))
sos2 = np.abs(np.flip(np.fromfile(cfg.TE05c0_filename, dtype = np.complex64).reshape(cfg.data_shape).swapaxes(0,2), (1,2)))
 
sos1 = zoom(sos1, np.array(cfg.recon_shape)/ np.array(cfg.data_shape), order = 1, prefilter = False)
sos2 = zoom(sos2, np.array(cfg.recon_shape)/ np.array(cfg.data_shape), order = 1, prefilter = False)
 
nib.save(nib.Nifti1Image(sos1, output_affine_ras),  odir  / f'sos_virt_channels_no_smoothing.nii')
nib.save(nib.Nifti1Image(sos2, output_affine_ras),  odir  / f'sos_virt_channels_smoothed.nii')
 
 
# %%
# show the results
# ----------------
 
# scale sos images
scale_fac = abs_recon.sum() / sos2.sum()
sos1 *= scale_fac
sos2 *= scale_fac
 
vmax = np.percentile(abs_recon, 99.9)
 
ims2 = [{
    'cmap': plt.cm.Greys_r,
    'vmax': np.percentile(aimg, 99.9)
}] + 3 * [{
    'cmap': plt.cm.Greys_r,
    'vmin': 0,
    'vmax': vmax
}] + [{
    'cmap': plt.cm.Greys_r,
    'vmin': 0,
    'vmax': 1.
}]
vi2 = pv.ThreeAxisViewer([
    np.flip(x, (0, 1))
    for x in [aimg, sos1, sos2, abs_recon, Gam_recon]
],
                         imshow_kwargs=ims2)
vi2.fig.savefig(Path(odir) / 'recons.png')
 