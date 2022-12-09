import math
import numpy as np
import pandas as pd
import cv2

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


N = 8  # rows
M = 8  # columns
PI = 3.14
IMAGE_SIZE = 32
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
all_domain = []
vertical_edge_domain = []
horizontal_edge_domain = []
diagonal_edge_domain = []

##########################################
# Read Data
##########################################
img = cv2.imread('Lenna.png') 
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection

input_arr = img_gray[0:32, 0:32]
print(f"Printing input array = {input_arr}")
edges = edges[0:32, 0:32]



#################################################################
# Creating the cosine matrix and the transposed cosine matrix
#################################################################
C = [[0.0]*M for i in range(N)]
Ct = [[0.0]*M for i in range(N)]

for j in range (0, N):
    C[0][j] = 1.0 / math.sqrt(N)
    Ct[j][0] = C[0][j]
    
for i in range (1, N):
    for j in range(0, M):
        C[ i ][ j ] = math.sqrt( 2.0 / N ) * math.cos( (( 2 * j + 1 ) * i * PI )/ ( 2.0 * N ) )
        Ct[ j ][ i ] = C[ i ][ j ]


##########################################################################
# Performing DCT on 8x8 block (input --> intermediate --> DCT) Ct*input*C
##########################################################################

intermediate  = [ [0.0]*8 for i in range(8)]
DCT  = [ [0.0]*8 for i in range(8)]

for i in range(N):
    for j in range(N):
        for k in range(N):
            intermediate[i][j] += Ct[i][k] * input_arr[k][j]

for i in range(N):
    for j in range(N):
        for k in range(N):
            DCT[i][j] += intermediate[i][k] * C[k][j]


#################################################################
# Quantization according to the QUALITY factor = 2
#################################################################

quantum = [[0]*M for i in range(N)]
quantizedDCT = [[0]*M for i in range(N)]

# Defining the quantization matrix
for i in range (0, N):
    for j in range(0, M):
        quantum[ i ][ j ] = 1 + ( ( 1 + i + j ) * QUALITY )

# Final updated DCT matrix with quantization
for i in range (0, N):
    for j in range(0, M):
        quantizedDCT[ i ][ j ] = DCT[ i ][ j ] / quantum[ i ][ j ]

# print(quantizedDCT)

#################################################################
# FUNCTIONS FOR CLASSIFICATION
#################################################################
def is_vertical_edge(block):
    for i in range(len(block)): 
        flag = 1 
        for j in range(len(block)): 
            if block[j][i] != 0: 
                flag = 0 
        if flag == 1: 
            return True 
    return False

def is_horizontal_edge(block):
    for i in range(len(block)): 
        flag = 1 
        for j in range(len(block)): 
            if block[i][j] != 0: 
                flag = 0 
        if flag == 1: 
            return True 
    return False

def is_diagonal_edge(block):
    for i in range(len(block)): 
        flag = 1 
        for j in range(len(block)): 
            if block[i][j] != 0: 
                flag = 0 
        if flag == 1: 
            return True 
    return False

def check_features(block):
    features = {
        "vertical_edge": False,
        "horizontal_edge": False,
        "diagonal_edge": False
    }
    if (is_vertical_edge(block)):
        features["vertical_edge"] = True
    if (is_horizontal_edge(block)):
        features["horizontal_edge"] = True
    if (is_diagonal_edge(block)):
        features["diagonal_edge"] = True
    return features

#################################################################
# CREATING RANGE POOL AND CLASSIFY
#################################################################
no_of_range_in_row = (int)(IMAGE_SIZE / 4)
row = 0

for i in range(0, no_of_range_in_row):
    fourx32 = []
    for j in range(0, 4):
        t1 = np.array_split(input_arr[row], no_of_range_in_row)
        fourx32.append(t1)
        row += 1
    fourxfour = np.hsplit(np.array(fourx32),8)
    for each_element in fourxfour:
        fourxfour_trial = np.reshape(each_element,(4,4))
        all_range.append(fourxfour_trial)

#  write the code for classification
# figure out the indexes of the range_block

# print(f"Printing all_range = {all_range}")

#################################################################
# CREATING DOMAIN POOL AND CLASSIFY
#################################################################

no_of_domain_in_row = IMAGE_SIZE - M - 1
temp_input_arr = input_arr
temp_domain = [[0]*M for i in range(N)]

row_overlaps = 0
col_overlaps = 0
print(len(input_arr[0]))
while(col_overlaps < len(input_arr[0])-7):
    while(row_overlaps < len(input_arr[0])-7):
        temp_domain = [[0]*M for i in range(N)]
        for i in range (8):
            for j in range (8):
                temp_domain[i][j] = input_arr[i + col_overlaps][j+row_overlaps]
        all_domain.append(temp_domain)
        features = check_features(edges[col_overlaps:col_overlaps+8, row_overlaps:row_overlaps+8])
        if (features["vertical_edge"]):
            vertical_edge_domain.append(temp_domain)
        if (features["horizontal_edge"]):
            horizontal_edge_domain.append(temp_domain)
        if (features["diagonal_edge"]):
            diagonal_edge_domain.append(temp_domain)

        #  write the code for classification and figure out indexes of domain_block
        row_overlaps += 1
    col_overlaps += 1
    row_overlaps = 0

print(len(vertical_edge_domain))
print(len(horizontal_edge_domain))
print(len(diagonal_edge_domain))






