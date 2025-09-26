#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# This code is an analysis module for WAD-QC 2.0: a server for automated 
# analysis of medical images for quality control.
#
# The WAD-QC Software can be found on 
# https://bitbucket.org/MedPhysNL/wadqc/wiki/Home
# 
# The analysis modules can be found on
# https://github.com/MedPhysQC

"""
Created on Tue Mar 11 09:24:33 2025

@author: Koen Baas, Joost Kuijer
"""

__version__ = '20250926'
__author__ = 'kbaas, jkuijer'

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
#import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import pydicom
from pydicom.uid import generate_uid
import os
from scipy.ndimage import gaussian_filter
    
def generate_mr_geometryXY(image_size=512, diam_x=190, diam_y=190, pixel_size=[1,1], shift=[0,0], SNR=40, sigma = 1):
    """
    Generates a synthetic MR image (phantom) containing an elliptical structure, 
    intended as a digital reference object for validating the geometry XY module 
    of the WAD-QC MR module.The image is saved as a PNG file and includes 
    text annotation with simulation parameters embedded in the pixel data. The
    resulting image can be used to verify in-plane geometry measurements after
    installation or updates of the WAD-QC module.

    Parameters:
    -----------
    image_size : int, optional
        Size (in pixels) of the square image. Default is 512x512.
    diam_x : float, optional
        Diameter of the ellipse in the x-direction (in mm). Default is 190 mm.
    diam_y : float, optional
        Diameter of the ellipse in the y-direction (in mm). Default is 190 mm.
    pixel_size : list of float, optional
        Pixel dimensions in [x, y] (in mm). Default is [1, 1].
    shift : list of float, optional
        Offset of the ellipse center from the image center in [x, y] (in mm). Default is [0, 0].
    SNR : float, optional
        Signal-to-noise ratio used to simulate Gaussian noise. Default is 40.
    sigma : float, optional
        Standard deviation for Gaussian blurring. Default is 1.

    Output:
    --------
    png_filename : str
        Filename of the generated PNG image containing the phantom.

    Notes:
    ------
    - Noise is applied based on the given SNR.
    - Pixel values exceeding 255 may be clipped, though only a warning is issued.
    """
       
    # To simulate partial volume effects in MR images, downsample a high-res
    # binary mask to final image resolution.
    n = 8 # up/downsampling factor
    image_size_up = n * image_size

    # Define parameters
    center = image_size_up // 2
    radius_x_pixels = n * (diam_x / pixel_size[0]) / 2  # Radius in the x-direction (in pixels)
    radius_y_pixels = n * (diam_y / pixel_size[1]) / 2  # Radius in the y-direction (in pixels)
    shift_pixels    = n * np.round(np.divide(shift, pixel_size))

    # Create a blank image with zeros
    phantom_image_up = np.zeros((image_size_up, image_size_up), dtype=np.uint8)

    # Create a grid of coordinates
    y, x = np.ogrid[:image_size_up, :image_size_up]
    
    # Ellipse equation: ((x - center_x) / radius_x)^2 + ((y - center_y) / radius_y)^2 <= 1
    mask_up = (((x - center - shift_pixels[0]) / radius_x_pixels) ** 2 + 
              ((y - center - shift_pixels[1]) / radius_y_pixels) ** 2) <= 1  # Ellipse mask
    phantom_mean = 200
    phantom_image_up = phantom_image_up + phantom_mean * mask_up
    # downsample binary image
    phantom_image = phantom_image_up.reshape(-1, n, image_size, n).mean((-1,-3))
    # smoothing
    phantom_image_sm = gaussian_filter(phantom_image, sigma)
    # add noise
    SD_noise = phantom_mean/SNR
    noise = np.random.normal(0, SD_noise, phantom_image.shape)
    phantom_image_noise = np.sqrt((phantom_image_sm+noise)**2)

    if np.max(phantom_image_noise)>255:
        print("Values within phantom exceed 255, values are clipped")
    phantom_image_noise = np.clip(phantom_image_noise, 0, 255).astype(np.uint8)
        
    # Save as PNG
    output_filename = f"GeomXY_diamX{diam_x}_Y{diam_y}_Shift_{shift[0]}x{shift[1]}_SNR_{SNR}_Sigma_{sigma}"
    png_filename = f"{output_filename}.png"
    Image.fromarray(phantom_image_noise).save(png_filename)
           
    # Open PNG file and add text to pixeldata
    image_text = Image.open(png_filename)
    draw = ImageDraw.Draw(image_text)
    text = f"X diam = {diam_x} mm\nY diam = {diam_y} mm\nOffc: x = {shift[0]} mm, y = {shift[1]} mm\nSNR= {SNR}\n σ = {sigma}"
    font = ImageFont.truetype("DejaVuSans.ttf", 8)
    draw.text((80, 120), text, font=font, fill=128)
    image_text.save(png_filename)
        
    # Display the image
    # plt.imshow(image_text, cmap='gray')
    # plt.title("Simulated MR Phantom Image")
    # plt.axis("off")
    # plt.show()

    print(f"Images saved as {png_filename}")
    return png_filename
    
def generate_mr_geometryZ(image_size=512, rect_size=[190, 147.5], pixel_size=[0.488, 0.488], shift=[0, 0], angle=0, SNR=100, sigma = 1):
    """
    Generates a synthetic MR image (phantom) with a rectangular structure, 
    intended as a digital reference object for validating the geometry Z module 
    of the WAD-QC MR module. The image is saved as a PNG file and includes 
    text annotation with simulation parameters embedded in the pixel data.
    The resulting image can be used to verify slice geometry measurements after
    installation or updates of the WAD-QC module.

    Parameters:
    -----------
    image_size : int, optional
        Size (in pixels) of the square image. Default is 512x512.
    rect_size : list of float, optional
        Size of the rectangle [width, height] in mm. Default is [190, 147.5].
    pixel_size : list of float, optional
        Pixel dimensions in [x, y] (in mm). Default is [0.488, 0.488].
    shift : list of float, optional
        Offset of the rectangle center from the image center in [x, y] (in mm). Default is [0, 0].
    angle : float, optional
        Rotation angle (in degrees) applied to the rectangle. Default is 0 (no rotation).
    SNR : float, optional
        Signal-to-noise ratio for simulating Gaussian noise. Default is 100.
    sigma : float, optional
        Standard deviation for Gaussian blurring. Default is 1.

    Output:
    --------
    png_filename : str
        Filename of the generated PNG image containing the phantom.

    Notes:
    ------
    - Rectangle is added via PIL and can be rotated as needed.
    - Gaussian noise is applied based on the given SNR value.
    - Pixel intensity values above 255 may be clipped; a warning is issued.
    """
    
    # To simulate partial volume effects in MR images, downsample a high-res
    # binary mask to final image resolution.
    n = 8 # up/downsampling factor
    image_size_up = n * image_size

    # Define parameters
    rect_width_pixels_up = n * round(rect_size[0]/pixel_size[0])  # Width of the rectangle (in pixels)
    rect_height_pixels_up = n * round(rect_size[1]/pixel_size[1])  # Height of the rectangle (in pixels)
    center_up = image_size_up // 2

    # Create a blank image with zeros
    phantom_image_up = np.zeros((image_size_up, image_size_up), dtype=np.uint8)

    # Create a grid of coordinates
    y, x = np.ogrid[:image_size_up, :image_size_up]

    # Create a rectangle shape and rotate it
    rect = Image.new('L', (rect_width_pixels_up, rect_height_pixels_up), color=240)  # White rectangle
    rotated_rect = rect.rotate(angle, expand=True, resample=Image.BICUBIC)
    
    # Get the rotated rectangle's position on the phantom image
    rect_x = center_up + round(n*shift[0]/pixel_size[0]) - rotated_rect.width // 2
    rect_y = center_up + round(n*shift[1]/pixel_size[1]) - rotated_rect.height // 2
    
    # Convert to PIL, paste rotated rectangle, convert back to numpy array
    pil_image = Image.fromarray(phantom_image_up)
    pil_image.paste(rotated_rect, (rect_x, rect_y), rotated_rect)
    
    phantom_image_up = np.array(pil_image)
    phantom_mean = 200
    phantom_image_up[phantom_image_up > 0] = phantom_mean
    # downsample binary image
    phantom_image = phantom_image_up.reshape(-1, n, image_size, n).mean((-1,-3))
    # smoothing
    phantom_image_sm = gaussian_filter(phantom_image, sigma)

    # Apply Gaussian noise
    SD_noise = phantom_mean/SNR
    noise_values = np.random.normal(loc=0, scale=SD_noise, size=phantom_image_sm.shape)
    phantom_image_noise = np.sqrt((phantom_image_sm+noise_values)**2).astype(np.uint8)
    if np.max(phantom_image_noise)>255:
        Warning("Values within phantom exceed 255, values are clipped")
        np.clip(phantom_mean,0, 255).astype(np.uint8)
        

    # Save as PNG
    output_filename = f"GeomZ_{rect_size[0]}x{rect_size[1]}_Shift_{shift[0]}x{shift[1]}_SNR_{SNR}_Angle_{angle}_Sigma_{sigma}"
    png_filename = f"{output_filename}.png"
    Image.fromarray(phantom_image_noise).save(png_filename)
    
    # Open PNG file and add text to pixeldata
    image_text = Image.open(png_filename)
    draw = ImageDraw.Draw(image_text)
    text = f"X size = {rect_size[0]} mm\nY size = {rect_size[1]} mm\nOffc: x = {shift[0]} mm, y = {shift[1]} mm\nAngle = {angle}°\nSNR = {SNR}\nsigma = {sigma}"
    font = ImageFont.load_default()
    text_position = (160, 240)
    draw.text(text_position, text, font=font)  # White text on black background

    # Save the images with the text added
    image_text.save(png_filename)

    print(f"Image saved as {png_filename}")
    return png_filename

def generate_mr_geometryZ_noUpsamping(image_size=512, rect_size=[190, 147.5], pixel_size=[0.488, 0.488], shift=[0, 0], angle=0, SNR=100, sigma = 1):
    """
    Generates a synthetic MR image (phantom) with a rectangular structure, 
    intended as a digital reference object for validating the geometry Z module 
    of the WAD-QC MR module. The image is saved as a PNG file and includes 
    text annotation with simulation parameters embedded in the pixel data.
    The resulting image can be used to verify slice geometry measurements after
    installation or updates of the WAD-QC module.

    Parameters:
    -----------
    image_size : int, optional
        Size (in pixels) of the square image. Default is 512x512.
    rect_size : list of float, optional
        Size of the rectangle [width, height] in mm. Default is [190, 147.5].
    pixel_size : list of float, optional
        Pixel dimensions in [x, y] (in mm). Default is [0.488, 0.488].
    shift : list of float, optional
        Offset of the rectangle center from the image center in [x, y] (in mm). Default is [0, 0].
    angle : float, optional
        Rotation angle (in degrees) applied to the rectangle. Default is 0 (no rotation).
    SNR : float, optional
        Signal-to-noise ratio for simulating Gaussian noise. Default is 100.
    sigma : float, optional
        Standard deviation for Gaussian blurring. Default is 1.

    Output:
    --------
    png_filename : str
        Filename of the generated PNG image containing the phantom.

    Notes:
    ------
    - Rectangle is added via PIL and can be rotated as needed.
    - Gaussian noise is applied based on the given SNR value.
    - Pixel intensity values above 255 may be clipped; a warning is issued.
    """
    
    # Define parameters
    rect_width_pixels = round(rect_size[0]/pixel_size[0])  # Width of the rectangle (in pixels)
    rect_height_pixels = round(rect_size[1]/pixel_size[1])  # Height of the rectangle (in pixels)
    center = image_size // 2

    # Create a blank image with zeros
    phantom_image = np.zeros((image_size, image_size), dtype=np.uint8)

    # Create a grid of coordinates
    y, x = np.ogrid[:image_size, :image_size]

    # Create a rectangle shape and rotate it
    rect = Image.new('L', (rect_width_pixels, rect_height_pixels), color=240)  # White rectangle
    rotated_rect = rect.rotate(angle, expand=True, resample=Image.BICUBIC)
    
    # Get the rotated rectangle's position on the phantom image
    rect_x = center + round(shift[0]/pixel_size[0]) - rotated_rect.width // 2
    rect_y = center + round(shift[1]/pixel_size[1]) - rotated_rect.height // 2
    
    # Convert to PIL, paste rotated rectangle, convert back to numpy array
    pil_image = Image.fromarray(phantom_image)
    pil_image.paste(rotated_rect, (rect_x, rect_y), rotated_rect)
    
    phantom_image = np.array(pil_image)
    phantom_mean = 200
    phantom_image[phantom_image > 0] = phantom_mean
    phantom_image_sm = gaussian_filter(phantom_image, sigma)

    # Apply Gaussian noise
    SD_noise = phantom_mean/SNR
    noise_values = np.random.normal(loc=0, scale=SD_noise, size=phantom_image_sm.shape)
    phantom_image_noise = np.sqrt((phantom_image_sm+noise_values)**2).astype(np.uint8)
    if np.max(phantom_image_noise)>255:
        Warning("Values within phantom exceed 255, values are clipped")
        np.clip(phantom_mean,0, 255).astype(np.uint8)
        

    # Save as PNG
    output_filename = f"GeomZ_{rect_size[0]}x{rect_size[1]}_Shift_{shift[0]}x{shift[1]}_SNR_{SNR}_Angle_{angle}"
    png_filename = f"{output_filename}.png"
    Image.fromarray(phantom_image_noise).save(png_filename)
    
    # Open PNG file and add text to pixeldata
    image_text = Image.open(png_filename)
    draw = ImageDraw.Draw(image_text)
    text = f"X size = {rect_size[0]} mm\nY size = {rect_size[1]} mm\nOffc: x = {shift[0]} mm, y = {shift[1]} mm\nAngle = {angle}°\nSNR = {SNR}\nSigma = {sigma}"
    font = ImageFont.load_default()
    text_position = (160, 240)
    draw.text(text_position, text, font=font)  # White text on black background

    # Save the images with the text added
    image_text.save(png_filename)

    print(f"Image saved as {png_filename}")
    return png_filename

def generate_mr_SNR_IU_GP_phantom(image_size=512, diam_x=190, diam_y=190, pixel_size=[1,1], SNR=195, IU = 68, GP = 0.5):
    """
    Generates a synthetic MR image (phantom) with an elliptical structure, 
    intended as a digital reference object for validating SNR, Image Uniformity (IU),
    and Ghosting Percentage (GP) in the WAD-QC MR module. The image is saved as a
    PNG file and includes text annotation with simulation parameters embedded in the pixel data.
    The resulting image can be used to verify SNR, IU and GP measurements after
    installation or updates of the WAD-QC module.
    
    Parameters:
    -----------
    image_size : int, optional
        Size (in pixels) of the square image. Default is 512x512.
    diam_x : float, optional
        Diameter of the ellipse in the x-direction (in mm). Default is 190 mm.
    diam_y : float, optional
        Diameter of the ellipse in the y-direction (in mm). Default is 190 mm.
    pixel_size : list of float, optional
        Pixel dimensions in [x, y] (in mm). Default is [1, 1].
    SNR : float, optional
        Nominal signal-to-noise ratio used to simulate Gaussian noise. Default is 195.
    IU : float, optional
        Image Uniformity percentage used to simulate variation in signal intensity. Default is 68.
    GP : float, optional
        Ghosting Percentage used to add synthetic ghost signal in background ROIs. Default is 0.5.
    
    Output:
    --------
    png_filename : str
        Filename of the generated PNG image containing the phantom.
    
    Notes:
    ------
    - Simulated signal intensities are adjusted based on SNR, IU, and GP inputs.
    - Additional ROIs are included for automated SNR, IU, and ghosting analysis.
    - Image is saved with 16-bit pixel depth for sufficient dynamic range.
    """
    # Define parameters
    radius_x_pixels = (diam_x / pixel_size[0]) / 2  # Radius in the x-direction (in pixels)
    radius_y_pixels = (diam_y / pixel_size[1]) / 2  # Radius in the y-direction (in pixels)
    center = image_size // 2
    
    # Create a blank image with zeros
    phantom_image = np.zeros((image_size, image_size), dtype=np.uint16)

    # Create a grid of coordinates
    y, x = np.ogrid[:image_size, :image_size]
    
    # Ellipse equation: ((x - center_x) / radius_x)^2 + ((y - center_y) / radius_y)^2 <= 1
    mask = (((x - center) / radius_x_pixels) ** 2 + 
            ((y - center) / radius_y_pixels) ** 2) <= 1  # Ellipse mask
    
    phantom_mean = 2000
    SD_noise =  (phantom_mean)/SNR
    
    SNR_masks = np.zeros((image_size, image_size), dtype=np.uint16)
    SNR_masks[(y >= 9) & (y <= 23) & (x >= 9) & (x <= 23)] = 1
    SNR_masks[(y >= 233) & (y <= 247) & (x >= 233) & (x <= 247)] = 1
    SNR_masks[(y >= 9) & (y <= 23) & (x >= 233) & (x <= 247)] = 1
    SNR_masks[(y >= 233) & (y <= 247) & (x >= 9) & (x <= 23)] = 1
    
    noise_image = np.random.normal(0, SD_noise, phantom_image.shape)
    phantom_image = phantom_image + phantom_mean*mask
    phantom_image = np.sqrt((phantom_image + noise_image)**2).astype(np.uint16)
    
    SNR_act = np.round(0.655 * np.mean(phantom_image[mask == 1])/np.std(phantom_image[SNR_masks == 1]))
    
    #Add two circles with higher and lower values for image uniformity test
    k = 1 - IU / 100
    min_value = phantom_mean - 500
    max_value = min_value * (1 + k) / (1 - k)
    subtract_value = np.uint16(phantom_mean - min_value)
    add_value = np.uint16(max_value - phantom_mean)
    
    y, x = np.ogrid[:image_size, :image_size]
    
    # Circle parameters
    cm2_area = 2.5  # 1 cm²
    radius_pixels = int(np.sqrt((cm2_area * (10**2)) / np.pi))  # Convert cm² to pixels²

    # Circle 1: Brighter area (centered at center-30, center)
    circle1_mask = ((x - (center - 30))**2 + (y - center)**2) <= radius_pixels**2
    phantom_image[circle1_mask] += add_value

    # Circle 2: Darker area (centered at center+30, center)
    circle2_mask = ((x - (center + 30))**2 + (y - center)**2) <= radius_pixels**2
    phantom_image[circle2_mask] -= subtract_value
    
    #Add ROIs that contain ghosting signal
    Ghosting_signal = GP/100*2*phantom_mean+np.mean(phantom_image[SNR_masks == 1])
    rect_masks = np.zeros((image_size, image_size), dtype=np.uint16)
    rect_masks[(y >= 100) & (y <= 156) & (x >= 9) & (x <= 23)] = 1
    rect_masks[(y >= 100) & (y <= 156) & (x >= 233) & (x <= 247)] = 1
    rect_masks[(y >= 9) & (y <= 23) & (x >= 100) & (x <= 156)] = 1
    rect_masks[(y >= 233) & (y <= 247) & (x >= 100) & (x <= 156)] = 1
    phantom_image[rect_masks == 1] = Ghosting_signal
    
    # Save as PNG
    output_filename = f"SNR_IU_phantom_SNR{SNR_act}_IU{IU}_GP{GP}"
    png_filename = f"{output_filename}.png"
    Image.fromarray(phantom_image).save(png_filename, format="PNG", bitdepth=16)
           
    # Open PNG file and add text to pixeldata
    image_text = Image.open(png_filename)
    draw = ImageDraw.Draw(image_text)
    text = f"SNR = {SNR_act}\nIU = {IU}"
    text2 = f"Ghosting = {GP}%"
    font = ImageFont.truetype("DejaVuSans.ttf", 8)
    draw.text((0.25*center - 20, 30), text, font=font, fill=600)
    draw.text((1.6*center - 20, 30), text2, font=font, fill=600)
    image_text.save(png_filename)

    print(f"Images saved as {png_filename}")
    return png_filename

def generate_mr_B0_map_phantom(image_size=128, diam_x=80, diam_y=80, pixel_size=[2,2], B0_uniformity_ppm = 1.2, dTE = 3.04, B0 = 1.5):
    """
    Generates a simulated MR phantom for B0 field inhomogeneity analysis using synthetic phase and magnitude images.
    
    Parameters:
    -----------
    image_size : int, optional
        Size (in pixels) of the square image. Default is 128x128
    diam_x : float, optional
        Diameter of the ellipse in the x-direction (in mm). Default is 80
    diam_y : float, optional
        Diameter of the ellipse in the y-direction (in mm). Default is 80
    pixel_size : list of float, optional
        Pixel dimensions in [x, y] (in mm). Default is [2, 2].
    B0_uniformity_ppm : float, optional
        Desired peak-to-peak B0 inhomogeneity in ppm. Default is 1.2.  
    dTE : float, optional
        Echo time difference in milliseconds. Default is 3.04.
    B0 : float, optional
        Main magnetic field strength in Tesla. Default is 1.5.
    
   Output:
   --------
  png_filename_ph1 : str
      Filename of the generated PNG image containing the phase image of the first TE.
  png_filename_ph2 : str
      Filename of the generated PNG image containing the phase image of the seconde TE.
  png_filename_magg : str
      Filename of the generated PNG image containing the magnitude image
      
    Notes:
    ------
    - The simulated phase images reflect spatial variations in magnetic field (B0) using incremental phase shifts in distinct circular ROIs.
    - The B0 field deviation is computed based on the user-defined peak-to-peak B0 inhomogeneity in ppm, echo time difference (dTE), and main field strength (B0).
    - The magnitude image contains a constant value but shares the same spatial mask as the phase images.
    - All images are saved in 16-bit PNG format for sufficient dynamic range.
    """
    
    # Define parameters
    radius_x_pixels = (diam_x / pixel_size[0]) / 2  # Radius in the x-direction (in pixels)
    radius_y_pixels = (diam_y / pixel_size[1]) / 2  # Radius in the y-direction (in pixels)
    center = image_size // 2
    
    # Create a blank image with zeros
    phase_image1 = np.zeros((image_size, image_size), dtype=np.uint16)
    phase_image2 = np.zeros((image_size, image_size), dtype=np.uint16)
    mag_image1 = np.zeros((image_size, image_size), dtype=np.uint16)


    # Create a grid of coordinates
    y, x = np.ogrid[:image_size, :image_size]
    
    # Ellipse equation: ((x - center_x) / radius_x)^2 + ((y - center_y) / radius_y)^2 <= 1
    mask = (((x - center) / radius_x_pixels) ** 2 + 
            ((y - center) / radius_y_pixels) ** 2) <= 1  # Ellipse mask
    
    mean_phase_image1 = 3000
    mean_phase_image2 = 2900
    mean_mag_image1 = 50
    
    noise_phase = np.zeros((image_size, image_size), dtype=np.uint16).astype(np.uint16)
    noise_mag = np.zeros((image_size, image_size), dtype=np.uint16).astype(np.uint16)
    
    phase_image1 = (mean_phase_image1*mask + noise_phase).astype(np.uint16)
    phase_image2 = (mean_phase_image2*mask + noise_phase).astype(np.uint16)
    mag_image1 = (mean_mag_image1*mask + noise_mag).astype(np.uint16)   

    
    # Circle parameters
    cm2_area = 1  # 1 cm²
    radius_pixels = int(np.sqrt((cm2_area * (5**2)) / np.pi))  # Convert cm² to pixels²

    # Circles
    circle1_mask = ((x - (center + 10))**2 + (y - center + 10)**2) <= radius_pixels**2
    circle2_mask = ((x - (center + 15))**2 + (y - center)**2) <= radius_pixels**2
    circle3_mask = ((x - (center + 10))**2 + (y - center - 10)**2) <= radius_pixels**2
    circle4_mask = ((x - (center))**2 + (y - center - 17)**2) <= radius_pixels**2
    circle5_mask = ((x - (center - 10))**2 + (y - center -10)**2) <= radius_pixels**2
    circle6_mask = ((x - (center - 15))**2 + (y - center)**2) <= radius_pixels**2
    circle7_mask = ((x - (center - 10))**2 + (y - center + 10)**2) <= radius_pixels**2
    
    B0_diff_rad = (B0_uniformity_ppm*267513*dTE*B0)/1.0e6
    B0_diff_im = (B0_diff_rad*4096)/(2*np.pi)
    Increment = np.uint16(B0_diff_im / 7)
    #Increment = 0
    phase_image1[circle1_mask] -= Increment*4
    phase_image1[circle2_mask] -= Increment*3
    phase_image1[circle3_mask] -= Increment*2
    phase_image1[circle4_mask] -= Increment*1
    phase_image1[circle5_mask] += Increment*1
    phase_image1[circle6_mask] += Increment*2
    phase_image1[circle7_mask] += Increment*3
    
    # Save as PNG
    output_filename = f"B0_map_phantom_p2p{B0_uniformity_ppm}_phase_image1"
    png_filename_ph1 = f"{output_filename}.png"
    Image.fromarray(phase_image1).save(png_filename_ph1, format="PNG", bitdepth=16)

    output_filename = f"B0_map_phantom_p2p{B0_uniformity_ppm}_phase_image2"
    png_filename_ph2 = f"{output_filename}.png"
    Image.fromarray(phase_image2).save(png_filename_ph2, format="PNG", bitdepth=16)
    
    output_filename = f"B0_map_phantom_p2p_{B0_uniformity_ppm}_magnitude_image"
    png_filename_magn = f"{output_filename}.png"
    Image.fromarray(mag_image1).save(png_filename_magn, format="PNG", bitdepth=16)
           
    # Open PNG file and add text to pixeldata
    image_text = Image.open(png_filename_ph1)
    draw = ImageDraw.Draw(image_text)
    text = f"B0 uniformity = {B0_uniformity_ppm} ppm"
    font = ImageFont.truetype("DejaVuSans.ttf", 8)
    draw.text((10, 10), text, font=font, fill=int(mean_phase_image1 + Increment))
    image_text.save(png_filename_ph1)

    print(f"Images saved as {png_filename_ph1}, {png_filename_ph2} and {png_filename_magn}")
    return [png_filename_ph1, png_filename_ph2, png_filename_magn]


def replace_dicom_pixel_data(dicom_path: str, png_path: str, output_path: str, set_rescale_slope: bool):
    """
    Replaces the pixel data in a DICOM file with pixel values from a provided PNG image. 
    
    Parameters:
    -----------
    dicom_path : str
       Path to the input DICOM file whose pixel data is to be replaced.
    png_path : str
       Path to the PNG image file (typically 16-bit grayscale) used to replace the DICOM pixel data.
    output_path : str
       Path where the modified DICOM file will be saved.
    set_rescale_slope : bool
       If True, sets the DICOM tag RescaleSlope to 1 to standardize intensity scaling.

    Output:
    -------
    None
       A new DICOM file is saved at the specified output path with updated pixel data.

    Notes:
    ------
    - The PNG image is expected to match the bit depth defined in the original DICOM's `BitsAllocated` tag (typically 8 or 16 bits).
    - The image is automatically converted to 16-bit mode ("I;16") if it is not already in that format.
    - If the PNG data range is ≤ 255 and the DICOM expects 16-bit data, the image values are scaled accordingly to preserve intensity fidelity.
    - Only pixel data is replaced; all other DICOM metadata is retained unless explicitly modified.
    - Setting `RescaleSlope` to 1 is useful for preserving absolute pixel intensity interpretation in some viewing software or pipelines.
    """
    
    # Load the DICOM file
    dicom = pydicom.dcmread(dicom_path)
    
    # Extract metadata
    bits_allocated = dicom.BitsAllocated  # Typically 8 or 16
    
    # Ensure bit depth compatibility
    dtype_map = {8: np.uint8, 16: np.uint16}
    if bits_allocated not in dtype_map:
        raise ValueError(f"Unsupported BitsAllocated: {bits_allocated}")
    
    expected_dtype = dtype_map[bits_allocated]

    # Load the PNG image correctly as 16-bit
    img = Image.open(png_path)
    if img.mode != "I;16":
        img = img.convert("I;16")  # Convert to 16-bit mode

    # Convert to NumPy array and ensure correct dtype
    img_array = np.array(img, dtype=np.uint16)  # Explicitly set as uint16

    # Normalize values if necessary
    if img_array.max() <= 255 and expected_dtype == np.uint16:
        # Scale up if the PNG was stored in 8-bit range but DICOM expects 16-bit
        img_array = (img_array.astype(np.float32) * (65535.0 / 255.0)).astype(np.uint16)

    # Update DICOM pixel data
    dicom.PixelData = img_array.tobytes()  # Use tobytes() to store raw data
    
    # Set RescaleSlope to 1 if requested
    if set_rescale_slope:
        dicom.RescaleSlope = 1
    
    # Save the modified DICOM file
    dicom.save_as(output_path)
    
    print(f"Updated DICOM pixel data: {output_path}")


def replace_uids(
    directory,
    output_dir=None,
    new_patID=None,
    new_patName=None,
    new_station_name=None,
    new_study_uid=None,
    new_study_date=None,   # Format: 'YYYYMMDD'
    new_study_time=None,   # Format: 'HHMMSS'
    new_image_comment=None
):
    """
    Replaces the UIDs, study date/time and patient properties in a DICOM file by desired content
    
    Parameters:
    -----------
    directory : str
       Path to the input DICOM files whose contents is to be replaced.
    output_dir : str
       Path where the modified DICOM file will be saved.
    new_patID : str
    new_patName : str
       Optional new patient ID/name info.
    new_station_name : str
       Optional new patient ID/name info.
    new_study_uid : uid
       Optional new study UID. Note: if none is given, a new study UID is generated.
    new_study_date : str   # Format: 'YYYYMMDD'
    new_study_time : str   # Format: 'HHMMSS'
       Optional new study date and time
    new_image_comment : str
       Optional image comment.

    Output:
    -------
    new_study_uid : uid
       The study UID of output data. May be used in consecutive calls of this function.
       A new DICOM file is saved at the specified output path with updated header data.

    Notes:
    ------
    - Series UID and instance UID are always newly generated.
    - Study UID is generated only if no study UID is proveded as parameter.
      Typical use is to generate a new study UID for the first series and pass
      this new UID on to the next series so they will have the same study UID.
    """

    # Generate shared Study UID if needed
    if not new_study_uid:
        new_study_uid = generate_uid()
    # Generate shared Series UIDs
    new_series_uid = generate_uid()

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        # Skip non-files
        if not os.path.isfile(filepath):
            continue

        try:
            ds = pydicom.dcmread(filepath)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        # Replace UIDs
        ds.StudyInstanceUID = new_study_uid
        ds.SeriesInstanceUID = new_series_uid
        ds.SOPInstanceUID = generate_uid()

        # Optionally replace patientID
        if new_patID:
            ds.PatientID = new_patID
        # Optionally replace patientName
        if new_patName:
            ds.PatientName = new_patName
        # Optionally replace StationName
        if new_station_name:
            ds.StationName = new_station_name

        # Optionally update StudyDate and StudyTime
        if new_study_date:
            ds.StudyDate = new_study_date
            ds.SeriesDate = new_study_date
            ds.AcquisitionDate = new_study_date
            ds.ContentDate = new_study_date
        if new_study_time:
            ds.StudyTime = new_study_time
            ds.SeriesTime = new_study_time
            ds.AcquisitionTime = new_study_time
            ds.ContentTime = new_study_time
        if new_study_date or new_study_time:
            ds.AcquisitionDateTime = ds.AcquisitionDate + ds.AcquisitionTime

        # Optionally update ImageComments
        if new_image_comment is not None:
            ds.ImageComments = new_image_comment

        # Save the modified file
        if output_dir:
            save_path = os.path.join(output_dir, filename)
        else:
            save_path = filepath  # overwrite original

        ds.save_as(save_path)
        print(f"Updated DICOM header: {save_path}")
        
    # use for other series in a study
    return new_study_uid
