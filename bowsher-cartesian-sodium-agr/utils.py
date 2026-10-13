from __future__ import annotations
import dataclasses
from typing import List

import json
import SimpleITK as sitk
import numpy as np

@dataclasses.dataclass
class AGRReconParameter:
    """num inner iterations"""
    niter: int = 10
    """num outer iterations"""
    n_outer: int = 15
    """prior weight for sodium image"""
    bet_recon: float = 6.
    """prior weight for gamma image"""
    bet_gam: float = 5
    """num nearest neighbors for Bowsher"""
    nnearest: int = 4
    """num neighbors for Bowsher"""
    nneigh: int = 80
    """don't use anat. info for prior"""
    no_anat_prior: bool = False
    """name of 1H MR for prior"""
    mr_name: str = 'T1_N4_corrected_aligned.npy'
    """prior method"""
    method: int = 0

@dataclasses.dataclass
class BirdCageBowsherParameter:
    """TE05 c0 sodium data file"""
    TE05c0_filename: str
    """TE5 c0 sodium data file"""
    TE5c0_filename: str
    """TE05 un-smoothed SOS sodium data file"""
    TE05soskw0_filename: str
    """TE05 smoothed SOS sodium data file"""
    TE05soskw1_filename: str
    """MPRAGE file (raw dump)"""
    MPRAGE_filename: str
    """output directory"""
    recon_dir: str
    """recon shape"""
    data_shape: tuple[int, int, int] = (64, 64, 64)
    """recon shape"""
    recon_shape: tuple[int, int, int] = (128, 128, 128)
    """sodium FOV in mm"""
    sodium_fov_mm: float = 220.
    """Shift in x-axis"""
    tx: float = 0.
    """Shift in y-axis"""
    ty: float = 0.
    """Shift in z-axis"""
    tz: float = 0.
    """alignment filter sigma"""
    alignment_filter_sigma: float = 1.3

@dataclasses.dataclass
class RegistrationParameter:
    """Fixed Image Filename (the file to register to)"""
    fixed_filename: str
    """Scratch Moving Image Filename (the file to register from scratch)"""
    moving_filename_scratch: str=""
    """Scratch Aligned Image Filename (the file to save the aligned image (from scratch) to)"""
    aligned_filename_scratch: str=""
    """Param Moving Image Filenames (the file(s) to register using the final registration parameters)"""
    moving_filenames_param: List[str]=dataclasses.field(default_factory=list)
    """Param Aligned Image Filenames (the file(s) to save the aligned image (from params) to. Must be same size and in the same order as moving_filenames_param)"""
    aligned_filenames_param: List[str]=dataclasses.field(default_factory=list)
    """Final Metric Value"""
    final_metric_value: float = 0.
    """Final Registration Parameters"""
    final_parameters: tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0)
    """Optimizer Stop Condition"""
    stop_condition: str=""
    """Transform File Path"""
    transform_file: str=""
   
def load_config(config_fname: str) -> BirdCageBowsherParameter:
    """Load configuration from a json file

    Parameters
    ----------
    config_fname : str
        abs path to the json config file

    Returns
    -------
    BirdCageBowsherParameter
    """
    return BirdCageBowsherParameter(**json.load(open(config_fname, 'r')))

def load_recon_config(config_fname: str) -> AGRReconParameter:
    """Load configuration from a json file

    Parameters
    ----------
    config_fname : str
        abs path to the json config file

    Returns
    -------
    AGRReconParameter
    """
    return AGRReconParameter(**json.load(open(config_fname, 'r')))

def load_registration_config(config_fname: str) -> RegistrationParameter:
    """Load configuration from a json file

    Parameters
    ----------
    config_fname : str
        abs path to the json config file

    Returns
    -------
    RegistrationParameter
    """
    return RegistrationParameter(**json.load(open(config_fname, 'r')))

def numpy_volume_to_sitk_image(vol: np.ndarray, voxel_size: np.ndarray,
                               origin: np.ndarray) -> sitk.Image:
    """convert a numpy array to a sitk image"""
    image = sitk.GetImageFromArray(np.swapaxes(vol, 0, 2))
    image.SetSpacing(voxel_size.astype(np.float64))
    image.SetOrigin(origin.astype(np.float64))

    return image


def sitk_image_to_numpy_volume(image: sitk.Image) -> np.ndarray:
    """convert a sitk image to a numpy array"""
    vol = np.swapaxes(sitk.GetArrayFromImage(image), 0, 2)

    return vol


if __name__ == '__main__':
    params = BirdCageBowsherParameter(
        TE05c0_filename='/abs_path/to/TE05_kw0.co',
        TE5c0_filename='/abs_path/to/TE5_kw0.co',
        TE05soskw0_filename='/abs_path/to/TE05_kw0.sos',
        TE05soskw1_filename='/abs_path/to/TE05_kw1.sos',
        MPRAGE_filename='/abs_path/to/mprage.nii',
        recon_dir='/abs_path/to/recon_dir')

    with open('demo_config.json', 'w') as f:
        json.dump(dataclasses.asdict(params), f, indent=2)

    recon_params = AGRReconParameter()

    with open('recon_params.json', 'w') as f:
        json.dump(dataclasses.asdict(recon_params), f, indent=2)
   
