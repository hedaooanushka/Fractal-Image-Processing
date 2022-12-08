import math
import numpy as np

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

input_arr = [
[2, 5, 7, 0, 6, 2, 7, 7, 7, 3, 7, 0, 7, 1, 7, 7, 3, 0, 4, 3, 6, 4, 8, 1, 6, 1, 2, 3, 7, 3, 6, 6],
[2, 4, 3, 5, 3, 6, 3, 5, 5, 7, 1, 6, 4, 5, 1, 8, 0, 2, 7, 2, 4, 5, 0, 3, 3, 6, 3, 3, 1, 0, 4, 4],
[3, 4, 3, 7, 4, 7, 4, 2, 4, 6, 0, 5, 3, 6, 7, 1, 1, 7, 4, 4, 7, 2, 4, 3, 6, 6, 6, 7, 3, 8, 2, 3],
[2, 3, 2, 4, 4, 1, 6, 3, 3, 4, 6, 5, 2, 4, 0, 3, 7, 1, 4, 6, 5, 8, 0, 6, 7, 0, 7, 8, 7, 0, 7, 0],
[3, 1, 1, 6, 0, 7, 3, 5, 1, 4, 3, 6, 4, 3, 3, 7, 1, 7, 7, 4, 3, 5, 2, 7, 4, 7, 8, 0, 4, 2, 6, 3],
[5, 0, 3, 7, 8, 2, 1, 2, 0, 2, 4, 5, 7, 8, 3, 0, 7, 2, 6, 1, 6, 8, 4, 1, 0, 5, 6, 3, 1, 3, 6, 2],
[4, 6, 5, 1, 7, 0, 4, 0, 4, 7, 3, 1, 3, 5, 2, 8, 3, 4, 0, 0, 1, 1, 4, 0, 2, 1, 6, 2, 2, 4, 0, 4],
[1, 4, 0, 0, 2, 5, 3, 7, 7, 7, 0, 1, 5, 8, 8, 6, 1, 1, 4, 5, 8, 0, 4, 3, 8, 7, 8, 7, 1, 8, 3, 2],
[6, 7, 7, 8, 4, 1, 4, 2, 6, 6, 5, 6, 1, 2, 8, 0, 3, 4, 1, 1, 6, 3, 0, 1, 3, 3, 5, 5, 5, 1, 4, 7],
[3, 6, 1, 2, 8, 8, 4, 8, 1, 2, 4, 3, 0, 7, 1, 0, 8, 4, 7, 4, 2, 7, 3, 6, 7, 3, 0, 7, 4, 1, 2, 6],
[4, 3, 2, 1, 8, 7, 2, 2, 0, 8, 8, 6, 5, 0, 5, 0, 0, 8, 2, 4, 0, 8, 3, 3, 5, 2, 8, 0, 1, 7, 3, 5],
[6, 2, 8, 8, 6, 0, 3, 4, 4, 8, 6, 2, 0, 3, 1, 5, 1, 3, 7, 4, 0, 6, 6, 6, 2, 5, 5, 8, 1, 1, 2, 5],
[6, 5, 4, 1, 3, 6, 0, 0, 5, 8, 6, 0, 8, 5, 6, 3, 5, 5, 4, 8, 3, 3, 0, 5, 8, 8, 1, 4, 4, 4, 0, 6],
[0, 7, 7, 8, 6, 7, 3, 3, 1, 6, 3, 2, 8, 3, 2, 1, 2, 4, 5, 4, 4, 8, 1, 8, 3, 4, 2, 4, 8, 7, 2, 8],
[7, 3, 5, 5, 7, 3, 2, 0, 8, 6, 3, 8, 4, 3, 0, 4, 4, 8, 5, 3, 5, 6, 8, 7, 7, 5, 3, 7, 6, 2, 3, 7],
[7, 3, 5, 6, 4, 5, 0, 7, 1, 0, 5, 1, 5, 1, 8, 4, 2, 6, 0, 4, 1, 2, 5, 3, 7, 1, 1, 4, 3, 7, 3, 1],
[2, 8, 1, 8, 0, 8, 7, 4, 8, 8, 6, 3, 0, 0, 5, 1, 1, 8, 5, 3, 3, 3, 5, 2, 0, 5, 2, 6, 8, 5, 0, 2],
[4, 7, 5, 3, 1, 7, 0, 5, 2, 7, 6, 4, 4, 1, 1, 5, 4, 0, 8, 0, 8, 0, 5, 3, 4, 2, 1, 5, 2, 5, 3, 5],
[7, 3, 8, 3, 2, 0, 2, 1, 0, 2, 7, 0, 5, 2, 4, 6, 4, 8, 8, 3, 1, 3, 7, 0, 6, 6, 6, 8, 4, 8, 7, 8],
[2, 1, 5, 5, 5, 1, 7, 4, 8, 1, 7, 8, 4, 7, 8, 8, 7, 7, 3, 6, 5, 6, 7, 1, 8, 6, 6, 0, 3, 3, 8, 1],
[3, 1, 0, 0, 0, 8, 5, 4, 6, 3, 5, 5, 4, 5, 6, 2, 3, 5, 1, 4, 4, 5, 5, 8, 0, 6, 8, 1, 8, 7, 2, 0],
[5, 4, 7, 2, 3, 2, 1, 7, 0, 8, 5, 5, 3, 0, 8, 2, 1, 0, 3, 4, 3, 0, 8, 0, 0, 0, 7, 6, 5, 6, 6, 5],
[4, 1, 0, 8, 3, 3, 7, 7, 3, 1, 6, 3, 4, 0, 8, 2, 8, 1, 1, 3, 7, 3, 4, 2, 8, 2, 6, 3, 4, 3, 5, 7],
[5, 1, 6, 6, 0, 0, 7, 0, 5, 7, 5, 6, 1, 4, 4, 4, 5, 5, 4, 2, 3, 4, 4, 0, 3, 3, 3, 3, 3, 0, 4, 1],
[8, 7, 7, 7, 8, 0, 2, 4, 6, 8, 4, 1, 6, 2, 4, 6, 5, 8, 2, 0, 2, 2, 0, 1, 4, 3, 5, 0, 1, 6, 2, 0],
[8, 2, 0, 6, 2, 7, 4, 1, 3, 8, 4, 2, 0, 1, 8, 2, 6, 7, 0, 5, 7, 0, 6, 0, 3, 6, 8, 7, 7, 7, 0, 6],
[7, 1, 1, 5, 1, 3, 0, 5, 5, 4, 5, 8, 3, 2, 8, 8, 2, 2, 1, 2, 2, 5, 3, 8, 0, 3, 7, 2, 8, 0, 7, 0],
[4, 5, 2, 3, 2, 2, 2, 4, 2, 5, 6, 5, 0, 2, 7, 3, 4, 5, 7, 4, 2, 8, 4, 7, 0, 6, 7, 5, 5, 7, 2, 2],
[8, 7, 3, 6, 0, 7, 7, 7, 1, 5, 2, 5, 2, 7, 5, 0, 4, 0, 7, 0, 6, 2, 2, 8, 8, 3, 7, 6, 0, 2, 0, 3],
[7, 0, 3, 7, 7, 1, 0, 7, 6, 6, 0, 5, 5, 3, 7, 2, 2, 3, 7, 0, 4, 1, 3, 2, 5, 1, 4, 1, 0, 5, 0, 4],
[2, 5, 1, 3, 5, 7, 5, 2, 0, 5, 2, 5, 1, 8, 5, 0, 2, 2, 4, 0, 5, 6, 0, 0, 1, 2, 1, 5, 2, 4, 7, 3],
[8, 4, 3, 2, 4, 6, 3, 6, 0, 2, 1, 6, 4, 4, 8, 4, 1, 8, 7, 1, 0, 5, 8, 2, 5, 5, 5, 5, 4, 1, 7, 3]]


C = [[0.0]*M for i in range(N)]
Ct = [[0.0]*M for i in range(N)]

#################################################################
# Creating the cosine matrix and the transposed cosine matrix
#################################################################
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
# CREATING RANGE POOL
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

#################################################################
# CREATING DOMAIN POOL
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
        row_overlaps += 1
    col_overlaps += 1
    row_overlaps = 0


