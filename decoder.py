"""
1st iteration:
Read original image
Convert into array of pixel values
Apply dct
Read encoding file
for every row in encoding file:
    Find corresponding Domain block
    Resize to 4x4
    Multiply with contrast(s) and add to brightness (o)
    Replace domain with range
apply inverse dct on transformed image 1
store transformed image 1

2nd iteration:
Read transformed image 1
Convert into array of pixel values
Read encoding file
for every row in encoding file:
    replace 4x4 range block with corresponding domain 4x4 domain block
Store transformed image 2

"""

import cv2
import numpy as np
import pandas as pd
from scipy.fft import fft, dct, idct
from PIL import Image as im
import math
import os
import logging as log

# from QuadTree import create_cosine_matrix, perform_DCT, quantize, convert_domain_to_4x4

PI = 3.14
QUALITY = 2
RANGE_SIZE = 4
DOMAIN_SIZE = 8

####### TO BE CHANGED #########

IMAGE_ROWS = 200
IMAGE_COLS = 200
INPUT_IMAGE = os.path.join("Output_Images", "decoded_koala_200_org.png")
ENCODING_FILE_NAME = os.path.join("Encoding_files", "koala_200_org_itr2.csv")
SAVE_IMAGE_AS = "decoded_koala_200_org_itr2"

###############################

log.basicConfig(
    filename="fractal.log",
    filemode="a",
    format="%(asctime)s - %(message)s",
    level=log.INFO,
)

##########################################
# Read Data
##########################################

img = cv2.imread(INPUT_IMAGE)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
edges = cv2.Canny(
    image=img_blur, threshold1=100, threshold2=200
)  # Canny Edge Detection

org_input_arr = np.array(img_gray[0:IMAGE_ROWS, 0:IMAGE_COLS])
input_arr = np.array(img_gray[0:IMAGE_ROWS, 0:IMAGE_COLS])
print(f"Printing input array: \n {input_arr}")

edges = edges[0:IMAGE_ROWS, 0:IMAGE_COLS]
print(f"Printing edges array: \n {edges}")

print(input_arr.shape)
print(edges.shape)

#################################################################
# Creating Cosine matrix, performing DT and Quantization
#################################################################
log.info("Decoding started")


def create_cosine_matrix(C, Ct):
    matrix_dict = {"C": [], "Ct": []}
    for j in range(0, IMAGE_ROWS):
        C[0][j] = 1.0 / math.sqrt(IMAGE_ROWS)
        Ct[j][0] = C[0][j]

    for i in range(1, IMAGE_ROWS):
        for j in range(0, IMAGE_COLS):
            C[i][j] = math.sqrt(2.0 / IMAGE_ROWS) * math.cos(
                ((2 * j + 1) * i * PI) / (2.0 * IMAGE_ROWS)
            )
            Ct[j][i] = C[i][j]

    matrix_dict["C"] = C
    matrix_dict["Ct"] = Ct
    return matrix_dict


##################################


def perform_DCT(C, Ct, input_arr):
    for i in range(IMAGE_ROWS):
        for j in range(IMAGE_COLS):
            for k in range(IMAGE_ROWS):  # either IMAGE_ROWS or IMAGE_COLS
                intermediate[i][j] += Ct[i][k] * input_arr[k][j]

    for i in range(IMAGE_ROWS):
        for j in range(IMAGE_COLS):
            for k in range(IMAGE_ROWS):
                DCT[i][j] += intermediate[i][k] * C[k][j]
    return DCT


#################################


def quantize(quantum, DCT, quantizedDCT, QUALITY):
    # Defining the quantization matrix
    for i in range(0, IMAGE_ROWS):
        for j in range(0, IMAGE_COLS):
            quantum[i][j] = 1 + ((1 + i + j) * QUALITY)

    # Final updated DCT matrix with quantization
    for i in range(0, IMAGE_ROWS):
        for j in range(0, IMAGE_COLS):
            quantizedDCT[i][j] = DCT[i][j] / quantum[i][j]

    # Converting elements to integers
    for i in range(0, IMAGE_ROWS):
        for j in range(0, IMAGE_COLS):
            quantizedDCT[i][j] = int(quantizedDCT[i][j])
    return quantizedDCT


###################################


def convert_domain_to_4x4(domain_block):
    rows = 0
    cols = 0
    fourx4 = []
    for i in range(4):
        for j in range(4):
            local_pixels = domain_block[cols : cols + 2, rows : rows + 2]
            sum = (
                np.int64(local_pixels[0][0])
                + np.int64(local_pixels[0][1])
                + np.int64(local_pixels[1][0])
                + np.int64(local_pixels[1][1])
            )
            sum /= 4
            fourx4.append(int(sum))
            rows += 2
        cols += 2
        rows = 0
    fourx4 = np.array(fourx4)
    fourx4 = fourx4.reshape(4, 4)
    return fourx4


###################################


def find_PSNR(original, compressed):
    # mse = 0
    # for i in range(original.shape[0]):
    #     for j in range(original.shape[1]):
    #         print(original[i][j])
    #         print(compressed[i][j])
    #         print(original[i][j] - compressed[i][j])
    #         mse += ((original[i][j] - compressed[i][j]) ** 2)
    #         print(mse)
    #         break
    #     break
    # print(mse)
    # mse /= (IMAGE_ROWS * IMAGE_COLS)
    mse = np.mean((original - compressed) ** 2)
    print(mse)
    if mse == 0:
        return 100
    max_pixel = 254
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


###################################


def save_image(image_name, input_arr):
    image = im.fromarray(input_arr)
    image.save(os.path.join("Output_Images", str(image_name + ".png")))


#################################################################
# Creating Cosine matrix, performing DT and Quantization
#################################################################
# C = [[0.0] * IMAGE_COLS for i in range(IMAGE_ROWS)]
# Ct = [[0.0] * IMAGE_COLS for i in range(IMAGE_ROWS)]

# intermediate = [[0.0] * IMAGE_COLS for i in range(IMAGE_ROWS)]
# DCT = [[0.0] * IMAGE_COLS for i in range(IMAGE_ROWS)]

# quantum = [[0] * IMAGE_COLS for i in range(IMAGE_ROWS)]
# quantizedDCT = [[0] * IMAGE_COLS for i in range(IMAGE_ROWS)]

# matrix_dict = create_cosine_matrix(C, Ct)
# C = matrix_dict["C"]
# Ct = matrix_dict["Ct"]
# DCT = perform_DCT(C, Ct, input_arr)
# quantizedDCT = quantize(quantum, DCT, quantizedDCT, QUALITY)
# input_arr = np.array(quantizedDCT)

# print(f"\n Updated input arrray: \n {input_arr}")

#################################################################
# Read Encoding File
#################################################################

df = pd.read_csv(ENCODING_FILE_NAME, delimiter="\t")
print(df.head())

for i in range(df.shape[0]):
    range_start_row = df.iloc[i].Range_index_row
    range_start_col = df.iloc[i].Range_index_col
    domain_start_row = df.iloc[i].Domain_index_row
    domain_start_col = df.iloc[i].Domain_index_col
    range_block = input_arr[
        range_start_row : range_start_row + 4, range_start_col : range_start_col + 4
    ]
    domain_block = input_arr[
        domain_start_row : domain_start_row + 8, domain_start_col : domain_start_col + 8
    ]
    domain_block = convert_domain_to_4x4(domain_block)
    for x in range(domain_block.shape[0]):
        for y in range(domain_block.shape[1]):
            domain_block[x][y] = int(
                (domain_block[x][y] * df.iloc[i].contrast) + df.iloc[i].brightness
            )
    for x in range(range_block.shape[0]):
        for y in range(range_block.shape[1]):
            range_block[x][y] = domain_block[x][y]

# print(f"Decoded Array")
# print(input_arr)


###################################################
# If applying DCT
###################################################

# final_dct_128 = idct(input_arr)
# print(f"\n Inverse DCT using scipy: \n {final_dct_128}")

# # Converting elements to integers
# for i in range (0, IMAGE_ROWS):
#     for j in range(0, IMAGE_COLS):
#         final_dct_128[i][j] = int(final_dct_128[i][j])

###################################################
# Find PSNR Value
###################################################

# print(org_input_arr)
# print(input_arr)
PSNR = find_PSNR(org_input_arr, input_arr)
print(f"PSNR value = {PSNR}")

###################################################
# Save Image
###################################################

save_image(SAVE_IMAGE_AS, input_arr)
log.info("Decoding ended")
