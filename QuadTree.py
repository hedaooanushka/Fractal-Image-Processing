import numpy as np
import pandas as pd
from scipy.fft import fft, dct, idct
import cv2
import math

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


N = 8  # rows
M = 8  # columns
#  have to delete M and N variables once we figure out what to replace them with (either IMAGE_ROWS or IMAGE_COLS) on lines 110, 115
PI = 3.14
IMAGE_ROWS = 16
IMAGE_COLS = 16
QUALITY = 2
RANGE_SIZE = 4
DOMAIN_SIZE = 8
DEPTH_MIN = 1
DEPTH_MAX = 3
DEPTH = 0
THRESHOLD = 8
C = []
Ct = []
all_range = []
all_ranges_indexes = []
edge_ranges = []
edge_domains = []
not_edge_domains = []

hex_squares_case1 = {
    1: [(0, 0), (0, 1)],
    2: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
    3: [(2, 0), (2, 1), (3, 0), (3, 1)],
    4: [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2)],
    5: [(1, 1) ,(1, 2), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2)],
    6: [(3, 1), (3, 2)],
    7: [(0, 3)],
    8: [(0, 3), (1, 2), (1, 3), (2, 3)],
    9: [(2, 3), (3, 2), (3, 3)]
}

hex_squares_case2 = {
    1: [(0, 0)],
    2: [(0, 0), (1, 0), (2, 0)],
    3: [(2, 0), (3, 0)],
    4: [(0, 0), (0, 1), (0, 2), (1, 0, (1, 1))],
    5: [(1, 0), (1, 1) ,(2, 0), (2, 1), (2, 2), (3, 0), (3, 1)],
    6: [(3, 0), (3, 1)],
    7: [(0, 2), (0, 3)],
    8: [(0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3)],
    9: [(2, 2), (2, 3), (3, 1), (3, 2), (3, 3)],
    10:[(0, 3), (1, 3)],
    11: [(1, 3), (2, 3)],
    12: [(3, 3)]   
}

hex_squares_case3 = {
    1: [(0, 0),(0, 1),(1, 0)],
    2: [(1, 0),(2, 0),(2, 1),(3, 0)],
    3: [(3, 0)],
    4: [(0, 1),(0, 2)],
    5: [(0, 1),(0, 2),(1, 0),(1, 1),(1, 2),(2, 1),(2, 2)],
    6: [(2, 1),(2, 2),(3, 0),(3, 1),(3, 2)],
    7: [(0, 2),(0, 3),(1, 2),(1, 3)],
    8: [(1, 2),(1, 3),(2, 2),(2, 3),(3, 2),(3, 3)],
    9: [(3, 2),(3, 3)]
}

##########################################
# Read Data
##########################################

img = cv2.imread('Lenna.png') 
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection


input_arr = img_gray[0:IMAGE_ROWS, 0:IMAGE_COLS]
print(f"Printing input array: \n {input_arr}")

edges = edges[0:IMAGE_ROWS, 0:IMAGE_COLS]
print(f"Printing edges array: \n {edges}")

#################################################################
# Creating the cosine matrix and the transposed cosine matrix
#################################################################
C = [[0.0]*IMAGE_COLS for i in range(IMAGE_ROWS)]
Ct = [[0.0]*IMAGE_COLS for i in range(IMAGE_ROWS)]

for j in range (0, IMAGE_ROWS):
    C[0][j] = 1.0 / math.sqrt(IMAGE_ROWS)
    Ct[j][0] = C[0][j]
    
for i in range (1, IMAGE_ROWS):
    for j in range(0, IMAGE_COLS):
        C[ i ][ j ] = math.sqrt( 2.0 / IMAGE_ROWS ) * math.cos( (( 2 * j + 1 ) * i * PI )/ ( 2.0 * IMAGE_ROWS ) )
        Ct[ j ][ i ] = C[ i ][ j ]

print(f"\n Cosine matrix: \n {C}")

##########################################################################
# Performing DCT on 8x8 block (input --> intermediate --> DCT) Ct*input*C
##########################################################################

intermediate  = [ [0.0]*IMAGE_COLS for i in range(IMAGE_ROWS)]
DCT  = [ [0.0]*IMAGE_COLS for i in range(IMAGE_ROWS)]

for i in range(IMAGE_ROWS):
    for j in range(IMAGE_COLS):
        for k in range(N):   # either IMAGE_ROWS or IMAGE_COLS
            intermediate[i][j] += Ct[i][k] * input_arr[k][j]

for i in range(IMAGE_ROWS):
    for j in range(IMAGE_COLS):
        for k in range(N):
            DCT[i][j] += intermediate[i][k] * C[k][j]

print(f"\n DCT without quantization: \n {DCT}")

original_scipy = idct(DCT)
print(f"\n Inverse DCT using scipy: \n {original_scipy}")

#################################################################
# Quantization according to the QUALITY factor = 2
#################################################################

quantum = [[0]*IMAGE_COLS for i in range(IMAGE_ROWS)]
quantizedDCT = [[0]*IMAGE_COLS for i in range(IMAGE_ROWS)]

# Defining the quantization matrix
for i in range (0, IMAGE_ROWS):
    for j in range(0, IMAGE_COLS):
        quantum[i][j] = 1 + ((1 + i + j) * QUALITY)

# Final updated DCT matrix with quantization
for i in range (0, IMAGE_ROWS):
    for j in range(0, IMAGE_COLS):
        quantizedDCT[i][j] = DCT[i][j] / quantum[i][j]

# Converting elements to integers
for i in range (0, IMAGE_ROWS):
    for j in range(0, IMAGE_COLS):
        quantizedDCT[i][j] = int(quantizedDCT[i][j])

print(f"\n Qunatized DCT: \n {quantizedDCT}")

# Updating input array with quantized DCT array
# input_arr = np.array(quantizedDCT)
print(f"\n Updated input arrray: \n {input_arr}")

################### DOUBT ##############################

# Do we have to find edges on the original input array or on the transformed quantized DCT array?

#  if find edge on quantized array
    # img_gray = cv2.cvtColor(input_arr, cv2.COLOR_BGR2GRAY)
    # img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    # edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
    # print(f"Printing UPDATED edges array: \n {edges}")

#  But here, cv2.cvtColor(input_arr, cv2.COLOR_BGR2GRAY), input array must be 3d, i.e. color image with 3 channels RGB

#################################################################
# FUNCTIONS FOR CLASSIFICATION
#################################################################

def is_edge(block):
    if np.sum(block) > 0:
            return True
    return False

##################################

def check_features(block):
    features = {
        "edge": False
    }
    if (is_edge(block)):
        features["edge"] = True
    return features

################################## 

def convert_domain_to_4x4(domain_block):
    rows = 0
    cols = 0
    fourx4 = []
    for i in range(4):
        for j in range(4):
            local_pixels = domain_block[cols:cols+2, rows:rows+2]
            sum = local_pixels[0][0] + local_pixels[0][1] + local_pixels[1][0] + local_pixels[1][1]
            sum /= 4
            fourx4.append(int(sum))
            rows += 2
        cols += 2   
        rows = 0 
    fourx4 = np.array(fourx4)
    fourx4 = fourx4.reshape(4, 4)
    return fourx4

##################################

def find_case(i):
    case_info = {
        "case_no": {},
        "hex_count": 0
    }
    if (i % 3 == 0):
        case_info["case_no"] = hex_squares_case1
        case_info["hex_count"] = 9
    elif (i % 3 == 1):
        case_info["case_no"] = hex_squares_case2
        case_info["hex_count"] = 12
    else:
        case_info["case_no"] = hex_squares_case3
        case_info["hex_count"] = 9
    return case_info

##################################

def find_hex_intensity(block, case):
    intensity = {
        1:0,
        2:0,
        3:0,
        4:0,
        5:0,
        6:0, 
        7:0,
        8:0,
        9:0,
        10:0,
        11:0,
        12:0
    }
    for key in case:
        sq_pixel_sum = 0
        for co_ordinate in case[key]:
            sq_pixel_sum += block[co_ordinate[0]][co_ordinate[1]]
        # print(sq_pixel_sum)
        sq_pixel_sum /= len(case[key])
        intensity[key] = (sq_pixel_sum)
    return intensity

##################################

def find_contrast(range_intensity, domain_intensity, case_info):
    sum_prod = 0
    sum_r = 0
    sum_d = 0
    sum_d2 = 0
    for key in domain_intensity:
        sum_prod += (range_intensity[key] * domain_intensity[key])
        sum_r += range_intensity[key]
        sum_d += domain_intensity[key]
        sum_d2 += (domain_intensity[key] * domain_intensity[key])
    s = ((case_info["hex_count"] * sum_prod) - (sum_d* sum_r)) / ((case_info["hex_count"] * (sum_d2)) - sum_d2)
    # final rms_error results in NaN if denominator becomes 0, have to handle this
    return s

##################################

def find_brightness(range_intensity, domain_intensity, case_info, contrast):
    sum_r = 0
    sum_d = 0
    for key in domain_intensity:
        sum_r += range_intensity[key]
        sum_d += domain_intensity[key]
    o = (sum_r - (contrast * sum_d)) / case_info["hex_count"]
    return o

##################################

def find_rms_error(range_intensity, domain_intensity, case_info):
    R = 0
    contrast = find_contrast(range_intensity, domain_intensity, case_info)
    brightness = find_brightness(range_intensity, domain_intensity, case_info, contrast)
    for key in range_intensity:
        x = contrast * domain_intensity[key] + brightness - range_intensity[key]
        R += pow(x, 2)
    return math.sqrt(R)


# #################################################################
# CREATING RANGE POOL AND CLASSIFY
# #################################################################

no_of_range_in_row = (int)(IMAGE_COLS / 4)
row = 0
col = 0
for i in range(no_of_range_in_row):
    for j in range(no_of_range_in_row):
        temp_range = input_arr[row:row+4, col:col+4]
        all_range.append(temp_range)
        all_ranges_indexes.append((row, col))
        # classifying ranges
        features_range = check_features(edges[row:row+4, col:col+4])
        if (features_range["edge"]):
            edge_ranges.append(temp_range)
        col += 4
    row += 4
    col = 0

print(f"\n Printing all_range: \n {all_range}")
print(f"\n Printing length of all_range: {len(all_range)}")
print(f"\n Printing all_range_indexes: \n {all_ranges_indexes}")
print(f"\n Length of edge_ranges list = {len(edge_ranges)}")



# #################################################################
# CREATING DOMAIN POOL AND CLASSIFY
# #################################################################

no_of_domain_in_row = IMAGE_COLS - DOMAIN_SIZE - 1
temp_input_arr = input_arr
temp_domain = [[0]*DOMAIN_SIZE for i in range(DOMAIN_SIZE)]

row_overlaps = 0
col_overlaps = 0
print(len(input_arr[0]))
while(col_overlaps < len(input_arr[0])-7):
    while(row_overlaps < len(input_arr[0])-7):
        temp_domain = [[0]*DOMAIN_SIZE for i in range(DOMAIN_SIZE)]
        for i in range (DOMAIN_SIZE):
            for j in range (DOMAIN_SIZE):
                temp_domain[i][j] = input_arr[i + col_overlaps][j+row_overlaps]
        # classifying domains
        features = check_features(edges[col_overlaps:col_overlaps+8, row_overlaps:row_overlaps+8])
        if (features_range["edge"]):
            edge_domains.append(temp_domain)
        else:
            not_edge_domains.append(temp_domain)
        row_overlaps += 1
    col_overlaps += 1
    row_overlaps = 0

print(f"\n Printing length of edge_domains: {len(edge_domains)}")
print(f"\n Printing length of not_edge_domains: {len(not_edge_domains)}")

# #################################################################
# MAIN ALGORITHM
# #################################################################

for i in range(len(all_range)):
    case_info = find_case(i)
    best_rms_error = 1000000000
    range_intensity = find_hex_intensity(all_range[i], case_info["case_no"])
    if all_range[i] not in edge_ranges:
        for domain_block in not_edge_domains:
            domain_block = np.array(domain_block)
            domain_block = convert_domain_to_4x4(domain_block)
            domain_intensity = find_hex_intensity(domain_block, case_info["case_no"])
            rms_error = find_rms_error(range_intensity, domain_intensity, case_info)
            print(rms_error)
            if(rms_error < best_rms_error):
                best_rms_error = rms_error       
    else:
        for domain_block in edge_domains:
            domain_block = np.array(domain_block)
            domain_block = convert_domain_to_4x4(domain_block)
            domain_intensity = find_hex_intensity(domain_block, case_info["case_no"])
            rms_error = find_rms_error(range_intensity, domain_intensity, case_info)
            print(rms_error)
            if(rms_error < best_rms_error):
                best_rms_error = rms_error 
    print(f"Range {i} error = {best_rms_error}")

# if all above is correct, we need to create a dictionary with all the required information to decode and add it to the encoding.py file 








