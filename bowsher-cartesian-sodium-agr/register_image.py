import numpy as np
import nibabel as nb
import SimpleITK as sitk
import pymirc.viewer as pv

from argparse import ArgumentParser
from pathlib import Path
import json

from utils import load_registration_config, numpy_volume_to_sitk_image, sitk_image_to_numpy_volume

def main():
    parser = ArgumentParser()
    parser.add_argument('--reg_cfg', help='registration config json file', default='registration_params.json')
    parser.add_argument('--resample_int', default='linear', help='which resampline interpolator to use: linear (default), nn (nearest neighbor), or bspline')
    args = parser.parse_args()

    reg_cfg_path = args.reg_cfg
    reg_cfg = load_registration_config(reg_cfg_path)
    transform_file = reg_cfg.transform_file
    transform = []

    # load fixed image
    fixed_nii = nb.load(Path(reg_cfg.fixed_filename))

    # only register an image from scratch if the input and output paths are given
    if reg_cfg.moving_filename_scratch and reg_cfg.aligned_filename_scratch:
        # load moving image
        moving_filename_scratch = reg_cfg.moving_filename_scratch
       
        moving_nii = nb.load(Path(moving_filename_scratch))
        # align moving image to fixed image from scratch

        print("Registering image from scratch")
        _, transform = align_image_from_scratch(fixed_nii, 
                                          moving_nii, 
                                          reg_cfg.aligned_filename_scratch, 
                                          params_output_path=reg_cfg_path,
                                          transform_output_path=transform_file,
                                          resample_interpolater=args.resample_int
                                        )
    else:
        print('Missing input or output filename or both. Cannot run registration from scratch.')
    
    if transform_file and not transform:
       transform = sitk.ReadTransform(transform_file)
    # only register an image with params if the transform, input and output paths are given   
    if reg_cfg.moving_filenames_param and reg_cfg.aligned_filenames_param and transform:
        moving_files = reg_cfg.moving_filenames_param
        aligned_files = reg_cfg.aligned_filenames_param
        if len(moving_files) == len(aligned_files):
            # repeat alignment from params for all files specified, but only if there's an input and output path for each
            for i in range(len(moving_files)):
                # load moving image
                moving_nii = nb.load(Path(moving_files[i]))
                print("Registering image from params")
                _ = align_image_from_params(fixed_nii, moving_nii, transform, aligned_files[i], resample_interpolater=args.resample_int)
        else:
            print("Input and output filename number mismatch")
    else: 
        print('Missing input or output filenames or params. Cannot run registration from params.')
        

def align_image_from_scratch(fixed, moving, img_output_path, 
                             params_output_path='', 
                             transform_output_path='',
                             resample_interpolater='linear',
                             sampling_percentage=0.5,
                             shrink_factors=[4, 2, 1],
                             smoothing_sigmas=[2, 1, 0],
                             ):
    # register the moving image to the fixed image with sitk Euler3DTransform
   
    # construct sitk image from nifti
    fixed_nii = nb.as_closest_canonical(fixed)
    fixed_img = fixed_nii.get_fdata()
    fixed_affine = fixed_nii.affine
    fixed_voxsize = np.linalg.norm(fixed_affine[:3, :3], axis=0)
    fixed_origin = fixed_affine[:-1, -1]
    fixed_image = numpy_volume_to_sitk_image(fixed_img.astype(np.float32), fixed_voxsize,
                                            fixed_origin)

    # construct sitk image from nifti
    moving_nii = nb.as_closest_canonical(moving)
    moving_img = moving_nii.get_fdata()
    moving_affine = moving_nii.affine
    moving_voxsize = np.linalg.norm(moving_affine[:3, :3], axis=0)
    moving_origin = moving_affine[:-1, -1]
    moving_image = numpy_volume_to_sitk_image(moving_img.astype(np.float32), moving_voxsize,
                                            moving_origin)

    # Initial Alignment
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )

    # For manual rotations
    # Euler3DTransform parameters are (angleX, angleY, angleZ)
    # angleZ corresponds to rotation on the axial plane
    angle_degrees_axial = 0
    angle_degrees_sagittal = 0
    angle_degrees_coronal = 0
    # Get existing rotations and add to Z
    current_params = list(initial_transform.GetParameters())
    current_params[0] += np.deg2rad(angle_degrees_sagittal)
    current_params[1] += np.deg2rad(angle_degrees_coronal)
    current_params[2] += np.deg2rad(angle_degrees_axial)  # index 2 is angleZ
    initial_transform.SetParameters(current_params)

    # For manual translations
    # Get current parameters and add translation
    # Euler3DTransform parameters are (angleX, angleY, angleZ, translateX, translateY, translateZ)
    current_params = list(initial_transform.GetParameters())
    #CSFH020 #CSFH028 #CSFH029
    current_params[3] += 0#12#120 #130 # X (negative = right)
    current_params[4] += 0#12 #120 #125 # Y (negative = posterior), adjust mm value as needed
    current_params[5] += 0#8  #80 #90 # Z (negative = superior), adjust mm value as needed
    initial_transform.SetParameters(current_params) 

    # init = sitk_image_to_numpy_volume(sitk.Resample(moving_image, fixed_image,
    #                                  initial_transform, sitk.sitkLinear, 0.0,
    #                                  moving_image.GetPixelID()))
    # nb.save(nb.Nifti1Image(init, fixed_affine), '/data/shelpert/stroke/STR_035/acute/intermediaries/T1_post_initialTransform.nii')
    
    # Registration
    registration_method = sitk.ImageRegistrationMethod()
    
    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(sampling_percentage)
    
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescentLineSearch(
        learningRate=0.3,
        numberOfIterations=5000,
        convergenceMinimumValue=1e-8,
        convergenceWindowSize=30,
    )
    
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    final_transform = registration_method.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32),
        sitk.Cast(moving_image, sitk.sitkFloat32),
    )
    
    # Post registration analysis
    # save params to a json file but first read the data and append to it
    if params_output_path:
        if Path(params_output_path).exists():
            with open(params_output_path, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        data['stop_condition'] = registration_method.GetOptimizerStopConditionDescription()
        data['final_metric_value'] = registration_method.GetMetricValue()
        data['final_parameters'] = final_transform.GetParameters()
        if transform_output_path:
            sitk.WriteTransform(final_transform, transform_output_path)
            data['transform_file'] = transform_output_path
        with open(params_output_path, "w") as f:
            json.dump(data, f, indent=2)
    
    if resample_interpolater == 'nn':
        default_value = 0
        interp = sitk.sitkNearestNeighbor
    else:
        default_value = 0.0
        if resample_interpolater == 'linear':
            interp = sitk.sitkLinear
        elif resample_interpolater == 'bspline':
            interp = sitk.sitkBSpline

    moving_resampled = sitk.Resample(moving_image, fixed_image,
                                     final_transform, interp, default_value,
                                     moving_image.GetPixelID())
    
    moving_aligned = sitk_image_to_numpy_volume(moving_resampled)
    
    # save aligned T1
    nb.save(nb.Nifti1Image(moving_aligned.astype(np.float32), fixed_affine), img_output_path)
    return [moving_aligned, final_transform] 

def align_image_from_params(fixed_nii, moving_nii, transform, img_output_path, resample_interpolater='linear'):
    # register moving image to fixed image using parameters from a previous registration

    # get data from nii
    fixed_nii = nb.as_closest_canonical(fixed_nii)
    fixed_affine = fixed_nii.affine
    fixed_voxsize = fixed_nii.header['pixdim'][1:4]
    fixed_origin = fixed_affine[:-1, -1]

    fixed_image = numpy_volume_to_sitk_image(fixed_nii.get_fdata(), fixed_voxsize,
                                                fixed_origin)

    moving_nii = nb.as_closest_canonical(moving_nii)
    moving_affine = moving_nii.affine
    moving_voxsize = moving_nii.header['pixdim'][1:4]
    moving_origin = moving_affine[:-1, -1]

    moving_image = numpy_volume_to_sitk_image(moving_nii.get_fdata(), moving_voxsize,
                                                moving_origin)

    if resample_interpolater == 'nn':
        default_value = 0
        interp = sitk.sitkNearestNeighbor
    else:
        default_value = 0.0
        if resample_interpolater == 'linear':
            interp = sitk.sitkLinear
        elif resample_interpolater == 'bspline':
            interp = sitk.sitkBSpline

    moving_resampled = sitk.Resample(moving_image, fixed_image,
                                     transform, interp, default_value,
                                     moving_image.GetPixelID())
        
    aligned = sitk_image_to_numpy_volume(moving_resampled)
        
    # save aligned image
    nb.save(nb.Nifti1Image(aligned.astype(np.float32), fixed_affine), img_output_path)
    return aligned 

if __name__ == "__main__":  
    main()