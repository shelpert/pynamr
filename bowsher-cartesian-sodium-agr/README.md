Workflow for running Siemens birdcage sodium AGR
================================================

Setup / Install
---------------

1. Install conda or mamba, e.g. from https://github.com/conda-forge/miniforge

2. Setup a new virtual environment that contains all dependencies
   ```
   mamba env create -f environment.yml     (or, but slower)
   conda env create -f environment.yml
   ```

3. Activate the new env. 
   ```
   conda activate bowsher-cartesian-sodium-agr
   ```

4. run the recons as described below

Running recons
--------------

1. configure input file names in a config.json file (start from demo_config.json)
     in the config.json you have to specify:
       - "TE05c0_filename": abs. path to the TE05 c0 sodium image you want to use (kw0 recommended)
       - "TE5c0_filename": abs. path to the TE5 c0 sodium image you want to use (kw0 recommended)
       - "TE05soskw0_filename": abs. path to the TE05 kw0 sos file (for export only)
       - "TE05soskw1_filename": abs. path to the TE05 kw1 sos file (for export only)
       - "MPRAGE_filename": abs. path of the MPRAGE "raw" float image 

2. preprocess the sodium data, run N4 of the proton MPRAGE and align the proton MPRAGE
   ```
   python preprocess.py --cfg my_config.json
   ``` 
3. run 3 AGRs with different beta values (pre-defined in the recon parameter jsons)
   ```
   python sodium_bowsher_agr.py --cfg my_config.json --recon_cfg recon1_params1.json
   python sodium_bowsher_agr.py --cfg my_config.json --recon_cfg recon1_params2.json
   python sodium_bowsher_agr.py --cfg my_config.json --recon_cfg recon1_params3.json
   ```
Configuration reference
-----------------------

Data config (`--cfg`), in addition to the file names:

| key | default | meaning |
|---|---|---|
| `data_shape` | `[64,64,64]` | acquired matrix size |
| `recon_shape` | `[128,128,128]` | recon grid (data are zero-padded in k-space to this) |
| `sodium_fov_mm` | `220` | FOV (same along all axes), used for the NIfTI affine |
| `delta_t` | `5.0` | echo spacing TE2 - TE1 in ms. Can be overridden with `--delta_t` |
| `ksp_file` | `null` | k-space readout time table (col 0: time in ms, col 1: k). `null` = built-in analytic TPI model (matches the g16 trajectory). Same trajectory assumed for both echoes |
| `k_edge` | `null` | k at the edge of the acquired k-space. `null` = max k in `ksp_file`, or 1.5 without a ksp file |
| `kedge_radius` | `null` | radius of the acquired k-space edge in recon grid units. `null` = `data_shape / 2` |
| `readout_duration_ms` | `null` | total readout time in ms (time at `k_edge`). If set, model times are scaled to match it (valid when only the gradient strength differs). `null` = model times unchanged |
| `n_readout_bins` | `32` | number of k-space shells / readout times. Must be small enough that no shell is empty |

Recon config (`--recon_cfg`), in addition to the prior settings:

| key | default | meaning |
|---|---|---|
| `init_filter_sigma` | `1.4` | Gaussian filter (voxels) on the initial images. `0` = unfiltered (the old `sodium_bowsher_agr-ge1.py`) |
| `low_signal_threshold` | `0.05` | voxels below this fraction of the max of the initial echo-1 image have gamma fixed to 1 |

Every output folder contains `effective_params.json` with the values actually used
(also stored as attributes in `recons.h5`, together with the k-space readout bin image
`readout_bin_img`, which can be used to check the k-space coverage).