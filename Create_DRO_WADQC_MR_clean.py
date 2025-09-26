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
Created on Mon Jan 6 2025

@author: Koen Baas, Joost Kuijer

Description:
This script automates the generation and integration of simulated MR phantom images 
into existing DICOM files for WAD-QC validation. Paths are centralized at the top 
to simplify reuse across different environments.
Explanation of parameters is given at the called funtion in DRO_WADQC_MR_functions

Dependencies:
- Python 3
- os, sys, datetime
- DRO_WADQC_MR_functions module (local)
  --> numpy, matplotlib, PIL, pydicom, scipy


TODO: read obligatory parameter settings from source DICOM files.
"""

__version__ = '20250926'
__author__ = 'kbaas, jkuijer'

import sys
import os
from datetime import datetime

#%% ========================= USER-DEFINED PATHS =============================
# Base directory (set this only)
BASE_DRO = "<your path>/WADQC_MR_DRO-main"
BASE_DRO = "/home/wadqc/DRO_Koen/WADQC_MR_DRO-main"
# Derived directories (change if necessary) 
DICOM_ORIG = os.path.join(BASE_DRO, "DICOM_orig") #original dicom files
DICOM_SIM = os.path.join(BASE_DRO, "DICOM_sim_spec_sim6") #folder where simulated data will be saved
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) #folder where this script is saved

# Add local function module path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import DRO_WADQC_MR_functions

#%% ====================== DICOM HEADER PROPERTIES ===========================
# Optionally set a new patient ID and Name
patID = '9999DRO'
patName = 'ACR_DRO'
# optionally set a new StationName (for selector rules)
stationName = "ACR_DRO"
# Set study date and time to relate simulation with results in WAD-QC
# can also be set manually in format YYYYMMDD and HHMMSS
study_date = datetime.now().strftime('%Y%m%d')
study_time = datetime.now().strftime('%H%M%S')
# Generate new study UID
study_uid = None

#%% ======================= XY CIRCULAR PHANTOM ==============================
filename_circular = DRO_WADQC_MR_functions.generate_mr_geometryXY(
    image_size=256, pixel_size=[1, 1], # MUST match source image properties
    diam_x=190, diam_y=190, shift=[0, 0], 
    SNR=40, sigma=0.5
)

dicom_src = os.path.join(DICOM_ORIG, "series010")
dicom_tgt = os.path.join(DICOM_SIM,  "series010")
dicom_file = os.path.join(dicom_tgt, "MR000006.dcm")
png_path = os.path.join(SCRIPT_DIR, filename_circular)
study_uid = DRO_WADQC_MR_functions.replace_uids(
    dicom_src, output_dir=dicom_tgt,
    new_patID=patID, new_patName=patName, new_station_name=stationName,
    new_study_uid=study_uid, new_study_date=study_date, new_study_time=study_time,
    new_image_comment=filename_circular
)
DRO_WADQC_MR_functions.replace_dicom_pixel_data(dicom_file, png_path, dicom_file, False)

#%% ======================= Z RECTANGULAR PHANTOM ============================
filename_rect = DRO_WADQC_MR_functions.generate_mr_geometryZ(
    image_size=512, pixel_size=[0.488, 0.488], # MUST match source image properties
    rect_size=[190, 147.5], shift=[0, 0], angle=2,
    SNR=40, sigma=1
)

dicom_src = os.path.join(DICOM_ORIG, "series012")
dicom_tgt = os.path.join(DICOM_SIM,  "series012")
dicom_file = os.path.join(dicom_tgt, "MR000000.dcm")
png_path = os.path.join(SCRIPT_DIR, filename_rect)
study_uid = DRO_WADQC_MR_functions.replace_uids(
    dicom_src, output_dir=dicom_tgt,
    new_patID=patID, new_patName=patName, new_station_name=stationName,
    new_study_uid=study_uid, new_study_date=study_date, new_study_time=study_time,
    new_image_comment=filename_rect
)
DRO_WADQC_MR_functions.replace_dicom_pixel_data(dicom_file, png_path, dicom_file, False)

#%% =================== SNR + Image Uniformity PHANTOM =======================
filename_SNR = DRO_WADQC_MR_functions.generate_mr_SNR_IU_GP_phantom(
    image_size=256, pixel_size=[1, 1], # MUST match source image properties
    diam_x=190, diam_y=190,
    SNR=185, IU=68, GP=0.5
)

dicom_src = os.path.join(DICOM_ORIG, "series011")
dicom_tgt = os.path.join(DICOM_SIM,  "series011")
dicom_file = os.path.join(dicom_tgt, "MR000000.dcm")
png_path = os.path.join(SCRIPT_DIR, filename_SNR)
study_uid = DRO_WADQC_MR_functions.replace_uids(
    dicom_src, output_dir=dicom_tgt,
    new_patID=patID, new_patName=patName, new_station_name=stationName,
    new_study_uid=study_uid, new_study_date=study_date, new_study_time=study_time,
    new_image_comment=filename_SNR
)
DRO_WADQC_MR_functions.replace_dicom_pixel_data(dicom_file, png_path, dicom_file, False)

#%% ======================= B0 MAP PHANTOM ===================================
filename_phase1, filename_phase2, filename_mag = DRO_WADQC_MR_functions.generate_mr_B0_map_phantom(
    image_size=128, pixel_size=[2, 2], # MUST match source image properties
    diam_x=190, diam_y=190,
    B0_uniformity_ppm=1.2,
    dTE=2.66, B0=1.5 # MUST match source image properties
)

# phase images
dicom_src = os.path.join(DICOM_ORIG, "series008")
dicom_tgt = os.path.join(DICOM_SIM,  "series008")
dicom_file1 = os.path.join(dicom_tgt, "MR000000.dcm")
dicom_file2 = os.path.join(dicom_tgt, "MR000001.dcm")
png_path1 = os.path.join(SCRIPT_DIR, filename_phase1)
png_path2 = os.path.join(SCRIPT_DIR, filename_phase2)
study_uid = DRO_WADQC_MR_functions.replace_uids(
    dicom_src, output_dir=dicom_tgt,
    new_patID=patID, new_patName=patName, new_station_name=stationName,
    new_study_uid=study_uid, new_study_date=study_date, new_study_time=study_time,
    new_image_comment=filename_phase1
)
DRO_WADQC_MR_functions.replace_dicom_pixel_data(dicom_file1, png_path1, dicom_file1, True)
DRO_WADQC_MR_functions.replace_dicom_pixel_data(dicom_file2, png_path2, dicom_file2, True)

# magnitude images
dicom_src = os.path.join(DICOM_ORIG, "series009")
dicom_tgt = os.path.join(DICOM_SIM,  "series009")
dicom_file1 = os.path.join(dicom_tgt, "MR000000.dcm")
dicom_file2 = os.path.join(dicom_tgt, "MR000001.dcm")
png_path = os.path.join(SCRIPT_DIR, filename_mag)
study_uid = DRO_WADQC_MR_functions.replace_uids(
    dicom_src, output_dir=dicom_tgt,
    new_patID=patID, new_patName=patName, new_station_name=stationName,
    new_study_uid=study_uid, new_study_date=study_date, new_study_time=study_time,
    new_image_comment=filename_SNR
)
# both magnitude images are identical
DRO_WADQC_MR_functions.replace_dicom_pixel_data(dicom_file1, png_path, dicom_file1, False)
DRO_WADQC_MR_functions.replace_dicom_pixel_data(dicom_file2, png_path, dicom_file2, False)

#%% ======== COPY SOME OTHER SERIES WITHOUT MODIFICATION OF PIXEL DATA =======
dicom_src = os.path.join(DICOM_ORIG, "series007")
dicom_tgt = os.path.join(DICOM_SIM,  "series007")
study_uid = DRO_WADQC_MR_functions.replace_uids(
    dicom_src, output_dir=dicom_tgt,
    new_patID=patID, new_patName=patName, new_station_name=stationName,
    new_study_uid=study_uid, new_study_date=study_date, new_study_time=study_time,
    new_image_comment="Unmodified"
)
