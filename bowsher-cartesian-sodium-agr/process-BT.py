"""preprocess sodium birdcage raw data for Bowsher AGR recon"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import nibabel as nib

import SimpleITK as sitk
import matplotlib.pyplot as plt
import pymirc.viewer as pv
import json
from scipy.ndimage import gaussian_filter, zoom
from scipy.stats import rice
import torch
from HD_BET.checkpoint_download import maybe_download_parameters
from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict
import subprocess

from utils import load_config, numpy_volume_to_sitk_image, sitk_image_to_numpy_volume

from register_image import align_image_from_scratch, align_image_from_params

def multiply_along_axis(A, B, axis):
    A = np.array(A)
    B = np.array(B)
    # shape check
    #if axis >= A.ndim:
    #    raise AxisError(axis, A.ndim)
    ##    raise ValueError("'A' and 'B' must have the same length along the given axis")
    # Expand the 'B' according to 'axis':
    # 1. Swap the given axis with axis=0 (just need the swapped 'shape' tuple here)
    swapped_shape = A.swapaxes(0, axis).shape
    # 2. Repeat:
    # loop through the number of A's dimensions, at each step:
    # a) repeat 'B':
    #    The number of repetition = the length of 'A' along the 
    #    current looping step; 
    #    The axis along which the values are repeated. This is always axis=0,
    #    because 'B' initially has just 1 dimension
    # b) reshape 'B':
    #    'B' is then reshaped as the shape of 'A'. But this 'shape' only 
    #     contains the dimensions that have been counted by the loop
    for dim_step in range(A.ndim-1):
        B = B.repeat(swapped_shape[dim_step+1], axis=0)\
             .reshape(swapped_shape[:dim_step+2])
    # 3. Swap the axis back to ensure the returned 'B' has exactly the 
    # same shape of 'A'
    B = B.swapaxes(0, axis)
    return A * B

## %
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cfg', help='config json file', default='config.json')
    parser.add_argument('--uni', help='which uni images are from', default='NYU')
    parser.add_argument('--flip_cimg', action='store_true', default=False, help='whether to flip the SU images')
    parser.add_argument('--align_TE2', action='store_true', default=False, help='whether to align TE2 to TE1 (only if SNR is high enough)')
    parser.add_argument('--sodium_N4', action='store_true', default=False, help='whether to run the N4 correction on the 23Na images')
    parser.add_argument('--phase_correct', action='store_true', default=False, help='whether to apply the phase correction')
    parser.add_argument('--skull_strip', action='store_true', default=False, help='whether to skull strip both the T1 and 23Na images before alignment')

    args = parser.parse_args()
    cfg = load_config(args.cfg)
    uni = args.uni
    flip_cimg = args.flip_cimg
    

    # create the output directory
    odir = Path(cfg.recon_dir)
    odir.mkdir(exist_ok=True)

    # %%
    # load complex Na images
    # ----------------------
    if (uni == 'SU'):
        #cimg = np.fromfile(cfg.TE05c0_filename,dtype=np.complex64).reshape(cfg.data_shape).swapaxes(0, 2)
        #cimg2 = np.fromfile(cfg.TE5c0_filename,dtype=np.complex64).reshape(cfg.data_shape).swapaxes(0, 2)
        #cimg = np.load(cfg.TE05c0_filename).reshape(cfg.data_shape).swapaxes(0, 2)
        #cimg2 = np.load(cfg.TE5c0_filename).reshape(cfg.data_shape).swapaxes(0, 2)
        cimg = nib.load(cfg.TE05c0_filename).get_fdata().swapaxes(0, 2)
        cimg2 = nib.load(cfg.TE5c0_filename).get_fdata().swapaxes(0, 2)
        
        if flip_cimg:
            print('flipping complex image')
            cimg = np.flip(cimg, (1,2))
            cimg2 = np.flip(cimg2, (1,2))

        cimg= 1000* cimg[:,:,:]/np.max(abs(cimg.reshape(cfg.data_shape)))
        cimg2=1000*cimg2[:,:,:]/np.max(abs(cimg2.reshape(cfg.data_shape)))

        #nib.save(nib.Nifti1Image(np.abs(cimg).astype(np.float32), np.eye(4)),str(Path(cfg.recon_dir) / 'echo1_input-SU.nii'))

        #nib.save(nib.Nifti1Image(np.abs(cimg2).astype(np.float32), np.eye(4)),str(Path(cfg.recon_dir) / 'echo2_input-SU.nii'))
    elif (uni == 'NYU'):
        #NYU data only      
        cimg = np.flip(np.fromfile(cfg.TE05c0_filename,dtype=np.complex64).reshape(cfg.data_shape).swapaxes(0, 2),(1, 2))
        cimg2 = np.flip(np.fromfile(cfg.TE5c0_filename,dtype=np.complex64).reshape(cfg.data_shape).swapaxes(0, 2),(1, 2))
        nib.save(nib.Nifti1Image(np.abs(cimg).astype(np.float32), np.eye(4)),str(Path(cfg.recon_dir) / 'echo1_input-NYU.nii'))

        nib.save(nib.Nifti1Image(np.abs(cimg2).astype(np.float32), np.eye(4)),str(Path(cfg.recon_dir) / 'echo2_input-NYU.nii'))
    else:
        print('No valid university selected')
    #
    # Phase corrects the images
    #
    if args.phase_correct:
        print('Applying phase correction')
        ph_cimg=np.angle(cimg)
        ph_cimg2=np.angle(cimg2)

        cimg=cimg  *np.exp(-1j*ph_cimg)
        cimg2=cimg2*np.exp(-1j*ph_cimg2)

    #
    # Corrects spatial shifts in the image
    #
    S1=np.fft.ifftn(cimg,norm='ortho')
    S2=np.fft.ifftn(cimg2,norm='ortho')

    xindex=np.array(list((range(cimg.shape[0]))))
    yindex=np.array(list((range(cimg.shape[1]))))
    zindex=np.array(list((range(cimg.shape[2]))))

    Nx2=(cimg.shape[0])/2
    Ny2=(cimg.shape[1])/2
    Nz2=(cimg.shape[2])/2

    tx=cfg.tx
    ty=cfg.ty
    tz=cfg.tz

    xindex=np.pi*(np.array(list((range(cimg.shape[0])))))/Nx2*tx
    yindex=np.pi*(np.array(list((range(cimg.shape[0])))))/Ny2*ty
    zindex=np.pi*(np.array(list((range(cimg.shape[0])))))/Nz2*tz

    xtrans=np.exp(-1j*xindex)
    ytrans=np.exp(-1j*yindex)
    ztrans=np.exp(-1j*zindex)

    S11=multiply_along_axis(S1,xtrans,0)
    S21=multiply_along_axis(S11,ytrans,1)
    S31=multiply_along_axis(S21,ztrans,2)
    #
    S12=multiply_along_axis(S2,xtrans,0)
    S22=multiply_along_axis(S12,ytrans,1)
    S32=multiply_along_axis(S22,ztrans,2)

    cimg=np.fft.fftn(S31,norm='ortho')
    cimg2=np.fft.fftn(S32,norm='ortho')
    #

    # # Pad k-space data with 0s then get back image space in recon shapte
    # # perform fft to get into k-space
    # cimg_fft = np.fft.fftn(cimg, norm='ortho')
    # # pad data with 0s
    # data = np.fft.fftshift(
    #     np.pad(np.fft.fftshift(cimg_fft),
    #         (cfg.recon_shape[0] - cfg.data_shape[0]) // 2))

    # # perform fft to get into k-space
    # cimg_fft2 = np.fft.fftn(cimg2, norm='ortho')
    # # pad data with 0s
    # data2 = np.fft.fftshift(
    #     np.pad(np.fft.fftshift(cimg_fft2),
    #         (cfg.recon_shape[0] - cfg.data_shape[0]) // 2))

    # cimg = np.fft.ifftn(data, norm='ortho')
    # cimg2 = np.fft.ifftn(data2, norm='ortho')

    # align echo2 to echo1
    cimg_nii_path = str(Path(cfg.recon_dir) / 'intermediaries/echo1_corrected.nii.gz')
    cimg2_nii_path = str(Path(cfg.recon_dir) / 'intermediaries/echo2_corrected.nii.gz')
    cimg_nii = nib.Nifti1Image(np.abs(cimg).astype(np.float32), np.eye(4))
    cimg2_nii = nib.Nifti1Image(np.abs(cimg2).astype(np.float32), np.eye(4))
    nib.save(cimg_nii,cimg_nii_path)
    nib.save(cimg2_nii,cimg2_nii_path)

    te1_n4_path = Path(cfg.recon_dir) / 'intermediaries/echo1_N4_corrected.nii'
    te2_n4_path = Path(cfg.recon_dir) / 'intermediaries/echo2_N4_corrected.nii'
    # if args.sodium_N4:
    #     print('Applying N4 correction to 23Na')
    #     if not te1_n4_path.exists():
    #         print('running N4 bias correction on TE1')
    #         inputImage = sitk.ReadImage(cimg_nii_path, sitk.sitkFloat32)
            
    #         corrector = sitk.N4BiasFieldCorrectionImageFilter()
    #         corrected_image1 = corrector.Execute(inputImage,
    #                                             sitk.OtsuThreshold(inputImage, 0, 1, 200))
    #         sitk.WriteImage(corrected_image1, str(te1_n4_path))
    #     else:
    #         print(f'loading N4 corrected TE1 from {te1_n4_path}')
    #         corrected_image1 = sitk.ReadImage(str(te1_n4_path))
    #     cimg = sitk_image_to_numpy_volume(corrected_image1)
    #     cimg_nii_path = te1_n4_path
    #     if not te2_n4_path.exists():
    #         print('running N4 bias correction on TE2')
    #         inputImage = sitk.ReadImage(cimg2_nii_path, sitk.sitkFloat32)
            
    #         corrector = sitk.N4BiasFieldCorrectionImageFilter()
    #         corrected_image2 = corrector.Execute(inputImage,
    #                                             sitk.OtsuThreshold(inputImage, 0, 1, 200))
    #         sitk.WriteImage(corrected_image2, str(te2_n4_path))
    #     else:
    #         print(f'loading N4 corrected TE2 from {te2_n4_path}')
    #         corrected_image2 = sitk.ReadImage(str(te2_n4_path))
    #     cimg2 = sitk_image_to_numpy_volume(corrected_image2)
    #     cimg2_nii_path = te2_n4_path

    # Initialize paths for alignment
    cimg_nii_for_alignment = cimg_nii
    cimg2_nii_for_alignment = cimg2_nii
   
    if args.skull_strip:
        te1_brain_path = Path(cfg.recon_dir) / f'intermediaries/echo1_brain.nii.gz'
        te2_brain_path = Path(cfg.recon_dir) / f'intermediaries/echo2_brain.nii.gz'
        maybe_download_parameters()
        predictor = get_hdbet_predictor(
            use_tta=True,
            device=torch.device('cuda'),  # or 'cuda' if you have a GPU
            verbose=True
        )
        if not te1_brain_path.exists():
            print('Stripping skull TE1')
            hdbet_predict(
                str(cimg_nii_path),
                str(te1_brain_path),
                predictor,
                keep_brain_mask=True,  # saves the mask
                compute_brain_extracted_image=True
            ) 
        print('Loading TE1 brain')
        cimg_nii_for_alignment = nib.load(str(te1_brain_path))
        if not te2_brain_path.exists():
            hdbet_predict(
                str(cimg2_nii_path),
                str(te2_brain_path),
                predictor,
                keep_brain_mask=True,  # saves the mask
                compute_brain_extracted_image=True
            ) 
        print('Loading TE2 brain')
        cimg2_nii_for_alignment = nib.load(str(te2_brain_path))
    
    if args.align_TE2:
        cimg_gf = gaussian_filter(cimg, cfg.alignment_filter_sigma)
        cimg2_gf = gaussian_filter(cimg2, cfg.alignment_filter_sigma)
        #cimg_nii_for_alignment = nib.Nifti1Image(np.abs(cimg_gf).astype(np.float32), np.eye(4))
        #cimg2_nii_for_alignment = nib.Nifti1Image(np.abs(cimg2_gf).astype(np.float32), np.eye(4))
        
        te2_aligned_path = Path(cfg.recon_dir) / 'intermediaries/echo2_aligned_corrected.nii.gz'
        na_alignment_transform_path = str(Path(cfg.recon_dir) / 'te2_to_te1_transform.tfm')
        if not te2_aligned_path.exists():
            print('Aligning TE2 to TE1')
            # Scale background mask down to data resolution
            # background_mask = nib.load(Path(cfg.recon_dir) / 'seg/background.nii.gz').get_fdata()
            # background2 = cimg2[background_mask == 1]
            # background1 = cimg1[background_mask == 1]

            # # Fit Rician distribution to noise from background and pad the echoes with
            # pad = 15
            # padded_shape = tuple(s + 2*pad for s in np.array(cfg.recon_shape))
            # b1, loc1, scale1 = rice.fit(np.abs(background1), floc=0)
            # b2, loc2, scale2 = rice.fit(np.abs(background2), floc=0)
            # # Pad both the images we use for alignment and the images we will do the final transformation on
            # padded_cimg_for_alignment = rice.rvs(b1, loc=0, scale=scale1, size=padded_shape)
            # padded_cimg = rice.rvs(b1, loc=0, scale=scale1, size=padded_shape)
            # padded_cimg2_for_alignment = rice.rvs(b2, loc=0, scale=scale2, size=padded_shape)
            # padded_cimg2 = rice.rvs(b2, loc=0, scale=scale2, size=padded_shape)
            # # Insert original image in the center
            # padded_cimg_for_alignment[pad:-pad, pad:-pad, pad:-pad] = cimg_nii_for_alignment.get_fdata()
            # padded_cimg[pad:-pad, pad:-pad, pad:-pad] = cimg_nii.get_fdata()
            # padded_cimg2_for_alignment[pad:-pad, pad:-pad, pad:-pad] = cimg2_nii_for_alignment.get_fdata()
            # padded_cimg2[pad:-pad, pad:-pad, pad:-pad] = cimg2_nii.get_fdata()
            
            # # Align padded echo 2
            # cimg2_aligned, _ = align_image_from_scratch(nib.Nifti1Image(np.abs(padded_cimg_for_alignment).astype(np.float32), np.eye(4)), 
            #                                             nib.Nifti1Image(np.abs(padded_cimg2_for_alignment).astype(np.float32), np.eye(4)), 
            #                                             str(te2_aligned_path), str(Path(cfg.recon_dir) / 'te2_to_te1_reg_params.json'), 
            #                                             transform_output_path=na_alignment_transform_path,
            #                                             resample_interpolater='linear',
            #                                             sampling_percentage=0.3,
            #                                             )
            # #cimg2_aligned = align_image_from_params(nib.Nifti1Image(np.abs(padded_cimg_for_alignment).astype(np.float32), np.eye(4)), nib.Nifti1Image(np.abs(padded_cimg2).astype(np.float32), np.eye(4)), sitk.ReadTransform(na_alignment_transform_path), str(te2_aligned_path), resample_interpolater='bspline')
            # cimg2 = cimg2_aligned[pad:-pad, pad:-pad, pad:-pad]
            cimg2_aligned, _ = align_image_from_scratch(cimg_nii_for_alignment, 
                                                        cimg2_nii_for_alignment, 
                                                        str(te2_aligned_path), str(Path(cfg.recon_dir) / 'te2_to_te1_reg_params.json'), 
                                                        transform_output_path=na_alignment_transform_path,
                                                        resample_interpolater='linear',
                                                        sampling_percentage=0.3,
                                                        )
            cimg2 = cimg2_aligned
            nib.save(nib.Nifti1Image(np.abs(cimg2).astype(np.float32), np.eye(4)),str(te2_aligned_path))
        else:
            print('Loading aligned TE2')
            cimg2_loaded = nib.load(str(te2_aligned_path)).get_fdata()
            # if not cimg2_loaded.shape[0] == cfg.recon_shape[0]:
            #     # perform fft to get into k-space
            #     cimg_loaded_fft2 = np.fft.fftn(cimg2_loaded, norm='ortho')
            #     # pad data with 0s
            #     data2_loaded = np.fft.fftshift(
            #         np.pad(np.fft.fftshift(cimg_loaded_fft2),
            #             (cfg.recon_shape[0] - cfg.data_shape[0]) // 2))
            #     cimg2 = np.fft.ifftn(data2_loaded, norm='ortho')
            # else:
            cimg2 = cimg2_loaded

    # Pad k-space data with 0s then get back image space in recon shapte
    # perform fft to get into k-space
    cimg_fft = np.fft.fftn(cimg, norm='ortho')
    # pad data with 0s
    data = np.fft.fftshift(
        np.pad(np.fft.fftshift(cimg_fft),
            (cfg.recon_shape[0] - cfg.data_shape[0]) // 2))

    # perform fft to get into k-space
    cimg_fft2 = np.fft.fftn(cimg2, norm='ortho')
    # pad data with 0s
    data2 = np.fft.fftshift(
        np.pad(np.fft.fftshift(cimg_fft2),
            (cfg.recon_shape[0] - cfg.data_shape[0]) // 2))
    # data = np.fft.fftn(cimg, norm='ortho')
    # data2 = np.fft.fftn(cimg2, norm='ortho')

    # expand dims to make add coil channel dim
    data = np.expand_dims(data, 0)
    data2 = np.expand_dims(data2, 0)
    sens = np.ones_like(data)

    # %%
    # save complex Na data
    # --------------------

    # save the data and the sensitities
    np.save(odir / f'echo1_{cfg.recon_shape[0]}.npy', data)
    np.save(odir / f'echo2_{cfg.recon_shape[0]}.npy', data2)
    np.save(odir / f'sens_{cfg.recon_shape[0]}.npy', sens)

    # # %%
    # # load binary MR data
    # # -------------------

    mr_nifti=nib.load((cfg.MPRAGE_filename))
    mr=(nib.as_closest_canonical(mr_nifti)).get_fdata()
    #     np.swapaxes(mr,0,2)
    nib.save(mr_nifti,str(Path(cfg.recon_dir) / 'T1_input.nii'))
    nib.save(mr_nifti,str(Path(cfg.recon_dir) / f"T1_input-{uni}.nii"))
    nib.save(nib.Nifti1Image(mr.astype(np.float32), np.eye(4)),
    str(Path(cfg.recon_dir) / f"T1_input-{uni}-f.nii"))


    #---------------------------------------------------------------------------------------------------
    # visualizations

    #a = np.abs(gaussian_filter(cimg, cfg.alignment_filter_sigma))
    #b = np.abs(gaussian_filter(cimg2, cfg.alignment_filter_sigma))
    #vmax = np.percentile(np.abs(cimg_gf), 99.99)

    #vi = pv.ThreeAxisViewer([np.abs(cimg_gf), np.abs(cimg2_gf)],
    #                        imshow_kwargs={
    #                            'vmax': vmax,
    #                            'cmap': 'Greys_r'
    #                        })
    #vi.fig.savefig(Path(cfg.recon_dir) / 'sodium_input.png')
    #vi2 = pv.ThreeAxisViewer(mr, imshow_kwargs={'cmap': 'Greys_r'})
    #vi2.fig.savefig(Path(cfg.recon_dir) / 'mr_input.png')

    # %%
    # N4 bias correction of the MPRAGE
    # --------------------------------

    #n4_path = Path(cfg.recon_dir) / 'intermediaries/T1_input_N4_corrected.nii.gz'
    n4_path = Path(cfg.recon_dir) / 'T1_input_N4_corrected.nii'
    if not n4_path.exists():
        print('running N4 bias correction on the MPRAGE')
        inputImage = sitk.ReadImage(str(Path(cfg.recon_dir) / 'T1_input.nii'),
                                    sitk.sitkFloat32)
        
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected_image = corrector.Execute(inputImage,
                                            sitk.OtsuThreshold(inputImage, 0, 1, 200))
        
        sitk.WriteImage(corrected_image, str(n4_path))
    else:
        print(f'loading N4 corrected MPRAGE from {n4_path}')
        corrected_image = sitk.ReadImage(str(n4_path))

    # %%
    # align the N4 corrected MPRAGE image
    # -----------------------------------
    # interpolate to smaller grid
    na_img = np.abs(cimg)
    na_img2 = np.abs(cimg2)
    n = cfg.recon_shape[0]
    output_affine_ras = np.diag([cfg.sodium_fov_mm/n,cfg.sodium_fov_mm/n,cfg.sodium_fov_mm/n,1])
    output_affine_ras[:-1,-1] = 0.5*(-220 + cfg.sodium_fov_mm/n)
    # construct the voxel size for the Na sos image
    na_voxsize = cfg.sodium_fov_mm / np.array(cfg.recon_shape)

    na_img = zoom(na_img,
                    np.array(cfg.recon_shape) / np.array(cfg.data_shape),
                    order=1,
                    prefilter=False)
    na_nii = nib.Nifti1Image(na_img.astype(np.float32), output_affine_ras)
    nib.save(na_nii,
            str(Path(cfg.recon_dir) / 'echo1_interpolated.nii.gz'))

    na_img2 = zoom(na_img2,
                    np.array(cfg.recon_shape) / np.array(cfg.data_shape),
                    order=1,
                    prefilter=False)
    nib.save(nib.Nifti1Image(na_img2.astype(np.float32), output_affine_ras),
            str(Path(cfg.recon_dir) / 'echo2_interpolated.nii.gz'))

    raw_gam_filename = str(Path(cfg.recon_dir) / 'raw_gamma_interpolated.nii.gz')
    raw_gam = na_img2/na_img
    nib.save(nib.Nifti1Image(raw_gam.astype(np.float32), output_affine_ras), raw_gam_filename)
        
    aligned_path = Path(cfg.recon_dir) / f'T1_N4_corrected_aligned.npy'
    aligned_niigz_path = Path(cfg.recon_dir) / f'intermediaries/T1_N4_corrected_aligned.nii.gz'
    if not aligned_path.exists():
        print('running MR alignment')

        # load the N4 corrected T1 in RAS
        t1_nii = nib.load(n4_path)
        t1_nii_for_alignment = t1_nii

        #    na_img = np.abs(gaussian_filter(cimg, cfg.alignment_filter_sigma))
        if args.skull_strip:
            t1_brain_path = Path(cfg.recon_dir) / f'intermediaries/t1_input_brain.nii.gz'
            if not t1_brain_path.exists():
                print('Stripping skull T1')
                hdbet_predict(
                    str(n4_path),
                    str(t1_brain_path),
                    predictor,
                    keep_brain_mask=True,  # saves the mask
                    compute_brain_extracted_image=True
                ) 
            print('Loading TE1 brain')
            t1_nii_for_alignment = nib.load(str(t1_brain_path))

        # pass in Na sitk but specify fixed_affine and fixed_header to match echo1_interpolated.nii
        # pass True in for fixed_header to bypass the header setting block
        t1_alignment_transform_path = str(Path(cfg.recon_dir) / 't1_to_na_te1_transform.tfm')
        na_img_for_alignment = np.abs(na_nii.get_fdata())
        na_nii_for_alignment = nib.Nifti1Image(na_img_for_alignment.astype(np.float32), output_affine_ras)
        align_image_from_scratch(na_nii_for_alignment, 
                                t1_nii_for_alignment, 
                                aligned_niigz_path, 
                                str(Path(cfg.recon_dir) / 't1_to_na_te1_reg_params.json'),
                                transform_output_path=t1_alignment_transform_path,
                                shrink_factors=[4, 2, 1, 1],
                                smoothing_sigmas=[2, 1, 0.5, 0],
                                )
        t1_aligned = align_image_from_params(na_nii, t1_nii, sitk.ReadTransform(t1_alignment_transform_path), str(aligned_niigz_path))
        
        # save aligned T1
        np.save(aligned_path, t1_aligned)

        # %%
        # show the aligned images

        vi3 = pv.ThreeAxisViewer([t1_aligned, na_img, na_img2, t1_aligned],
                                    [None, None, None, na_img],
                                    imshow_kwargs=dict(cmap='Greys_r'))
        vi3.fig.savefig(Path(cfg.recon_dir) / 'mr_input_N4_corrected_aligned.png')

