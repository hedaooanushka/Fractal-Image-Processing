from Fractal_Image_Processing import (
    DCT,
    Edge_Detection,
    Range_Domain_Processing,
    File_Operations,
)
import numpy as np
import cv2
import os
import logging as log

#############################
# Constants initialization
#############################
IMAGE_ROWS = 32
IMAGE_COLS = 32
RANGE_SIZE = 4
DOMAIN_SIZE = 8
IS_DCT = False
INPUT_IMAGE = os.path.join("Input_Images", "test4.png")
ENCODING_FILE_NAME = os.path.join("Encoding_files", "test4_weighted_32x32_trial.csv")
PI = 3.14
QUALITY = 2
log.basicConfig(
    filename="fractal.log",
    filemode="a",
    format="%(asctime)s - %(message)s",
    level=log.INFO,
)
C = []
Ct = []
all_range = []
all_ranges_indexes = []
edge_ranges = []
edge_ranges_indexes = []
edge_domains = []
edge_domains_indexes = []
not_edge_domains = []
not_edge_domains_indexes = []

hex_squares_case1 = {
    1: [(0, 0), 0.25],
    2: [(0, 2), 0.25],
    3: [(0, 0), 0.375, (0, 1), 0.25],
    4: [
        (0, 0),
        0.375,
        (0, 1),
        0.25,
        (1, 0),
        1,
        (1, 1),
        0.5,
        (2, 0),
        0.375,
        (2, 1),
        0.25,
    ],
    5: [(2, 0), 0.375, (2, 1), 0.25, (3, 0), 1, (3, 1), 0.5],
    6: [(0, 1), 0.5, (0, 2), 1, (0, 3), 0.25, (1, 1), 0.25, (1, 2), 0.375],
    7: [
        (1, 1),
        0.25,
        (1, 2),
        0.375,
        (2, 1),
        0.5,
        (2, 2),
        1,
        (2, 3),
        0.25,
        (3, 1),
        0.25,
        (3, 2),
        0.375,
    ],
    8: [(3, 1), 0.25, (3, 2), 0.375],
    9: [(0, 3), 0.375],
    10: [(0, 3), 0.375, (1, 2), 0.5, (1, 3), 1, (2, 3), 0.375],
    11: [(2, 3), 0.375, (3, 2), 0.25, (3, 3), 1],
}
hex_squares_case2 = {
    1: [(0, 0), 0.25],
    2: [(0, 0), 0.25, (1, 0), 0.5, (2, 0), 0.25],
    3: [(2, 0), 0.25, (3, 0), 0.5],
    4: [(0, 0), 0.5, (0, 1), 1, (0, 2), 0.25, (1, 0), 0.25, (1, 1), 0.375],
    5: [
        (1, 0),
        0.25,
        (1, 1),
        0.375,
        (2, 0),
        0.5,
        (2, 1),
        1,
        (2, 2),
        0.25,
        (3, 0),
        0.25,
        (3, 1),
        0.375,
    ],
    6: [(3, 0), 0.25, (3, 1), 0.375],
    7: [(0, 2), 0.375, (0, 3), 0.25],
    8: [
        (0, 2),
        0.375,
        (0, 3),
        0.25,
        (1, 1),
        0.25,
        (1, 2),
        1,
        (1, 3),
        0.5,
        (2, 2),
        0.375,
        (2, 3),
        0.25,
    ],
    9: [(2, 2), 0.375, (2, 3), 0.25, (3, 1), 0.25, (3, 2), 1, (3, 3), 0.5],
    10: [(0, 3), 0.5, (1, 3), 0.25],
    11: [(1, 3), 0.25, (2, 3), 0.5],
    12: [(3, 3), 0.25],
}
hex_squares_case3 = {
    1: [(0, 0), 1, (0, 1), 0.25, (1, 0), 0.375],
    2: [(1, 0), 0.375, (2, 0), 1, (2, 1), 0.25, (3, 0), 0.375],
    3: [(3, 0), 0.375],
    4: [(0, 1), 0.375, (0, 2), 0.25],
    5: [
        (0, 1),
        0.375,
        (0, 2),
        0.25,
        (1, 0),
        0.25,
        (1, 1),
        1,
        (1, 2),
        0.5,
        (2, 1),
        0.375,
        (2, 2),
        0.25,
    ],
    6: [(2, 1), 0.375, (2, 2), 0.25, (3, 0), 0.25, (3, 1), 1, (3, 2), 0.5],
    7: [(0, 2), 0.5, (0, 3), 1, (1, 2), 0.5, (1, 3), 0.375],
    8: [
        (1, 2),
        0.25,
        (1, 3),
        0.375,
        (2, 2),
        0.5,
        (2, 3),
        1,
        (3, 2),
        0.25,
        (3, 3),
        0.375,
    ],
    9: [(3, 2), 0.25, (3, 3), 0.375],
    10: [(1, 3), 0.25],
    11: [(3, 3), 0.25],
}

img = cv2.imread(INPUT_IMAGE)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ED = Edge_Detection(IMAGE_ROWS, IMAGE_COLS)

input_arr, edges = ED.read_image(INPUT_IMAGE)

# print(f"Printing input array: \n {input_arr}")

# print(f"Printing edges array: \n {edges}")

## DCT
if IS_DCT:
    dct = DCT(IMAGE_ROWS, IMAGE_COLS)
    matrix_dict = dct.create_cosine_matrix()
    dct_matrix = dct.perform_DCT(input_arr)
    quantized_DCT = dct.quantize(QUALITY)
    input_arr = np.array(quantized_DCT)
# print(quantized_DCT)
## Range and domain
rd = Range_Domain_Processing(
    IMAGE_ROWS,
    IMAGE_COLS,
    hex_squares_case1,
    hex_squares_case2,
    hex_squares_case3,
    input_arr,
    DOMAIN_SIZE,
    RANGE_SIZE,
    edges,
)

all_range, all_ranges_indexes, edge_ranges, edge_ranges_indexes = rd.create_range_pool()
# print(edge_ranges_indexes)

(
    edge_domains,
    edge_domains_indexes,
    not_edge_domains,
    not_edge_domains_indexes,
) = rd.create_domain_pool()
# print(f"\n Printing length of edge_domains: {len(edge_domains)}")
# print(f"\n Printing length of not_edge_domains: {len(not_edge_domains)}")

## Setting encoding data

file = File_Operations(
    IMAGE_ROWS,
    IMAGE_COLS,
    hex_squares_case1,
    hex_squares_case2,
    hex_squares_case3,
    input_arr,
    DOMAIN_SIZE,
    RANGE_SIZE,
    edges,
)
log.info("Encoding started")
encoding_file_data = file.create_dataframe(
    all_range,
    all_ranges_indexes,
    edge_ranges_indexes,
    not_edge_domains,
    not_edge_domains_indexes,
    edge_domains_indexes,
    edge_domains,
)
# print(edge_ranges_indexes)
# print(not_edge_domains_indexes)
file.save_encoding_file(encoding_file_data, ENCODING_FILE_NAME)
log.info("Encoding ended")
