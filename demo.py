# Rishabh Jain
# 06/20/19

# import the necessary packages
import numpy as np
import sys

from PIL import Image, ImageDraw, ImageFont
from gelquant import gelquant
from matplotlib import pyplot as plt

import pickle

def find_coeffs(source_coords, target_coords):
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
    A = np.matrix(matrix, dtype=np.float)
    B = np.array(source_coords).reshape(8)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

# Step 1. Display a sample-image
img = Image.open('cleaned_data/Gel14_2019-03-25_Hts97_6_8bit.png')
#plt.imshow(np.asarray(img))
#plt.show()

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

# Step 2: Perspetive transform
img_transform= img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
#plt.imshow(np.asarray(img_transform))


# Step 3. Strip out the individual lanes from the 26 lane gel-image
number_lanes= 26
number_expts= 1

data, bounds = gelquant.lane_parser(img_transform, number_lanes, number_expts, 0, 100)


# drop the EARLY part of the waveform
data = np.array(data)
data= data[:, -177:]
fig = plt.figure()
#plt.plot(data[1, :])

# apply scaler to TEST data
scaler =pickle.load(open('scaler.pkl', 'rb'))
xtest= scaler.transform(data)

# Step 4. PREDICT
load_rf_model =pickle.load(open('random_forest_model.pkl', 'rb'))
y_predict= load_rf_model.predict(xtest)

# draw the message on the background
draw = ImageDraw.Draw(img)

for i in range(0, 26):

	(x, y) = (20+ 36*i, 103)
	color = 'rgb(100, 255, 255)' # magenta
	lane= 'L' + str(i+1)
	draw.text((x, y), lane, fill=color, fontsize=20)

	(x, y) = (20+ 36*i, 120)
	color = 'rgb(255, 100, 255)' # green
	draw.text((x, y), str(y_predict[i]), fill=color, fontsize=20)


 	
# save the edited image
img.save('ML_inspector_RJ.png')
#print(y_predict)

