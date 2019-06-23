# Rishabh Jain
# 06/14/19

# import the necessary packages
import numpy as np
import imutils
import cv2

import sys
from PIL import Image

import glob

from gelquant import gelquant
from matplotlib import pyplot as plt

import csv
import pandas as pd

# reading a bunch of RAW images into a directory ...
image_list = []
for filename in sorted(glob.glob('cleaned_data/*.png')):
    print(filename)
    im= Image.open(filename)
    image_list.append(im)

# Set-up to do a PIL image transformation
def find_coeffs(source_coords, target_coords):
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
    A = np.matrix(matrix, dtype=np.float)
    B = np.array(source_coords).reshape(8)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


# "hard code" the image parameters
height= 427
width=  945
margin_width=  30
margin_height= 15

# other parameters
number_lanes= 26
number_expts= 1

# coeffs for perspective transformation
coeffs = find_coeffs(
    [(0, 0), (width- margin_width, 0), (width, height- margin_height), (0, height- margin_height)],
    [(0, 0), (width, 0), (width, height), (0, height)])


# Now, we can do serious pre-processing ...
# 1. Perspective transformation on the gel-images
# 2. Find individual gel-lanes 

df = pd.DataFrame([])
imgcounter=0
for img in image_list:

	img_transform= img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
	plt.imshow(np.asarray(img_transform))
	data, bounds = gelquant.lane_parser(img_transform, number_lanes, number_expts, 0, 100)
	
	# make a dataframe
	data = pd.DataFrame(np.array(data))
	df = df.append(data)
	print('processed image #', imgcounter)
	imgcounter+=1


# save 
df.to_csv('processed_gels.csv', encoding='utf-8')
