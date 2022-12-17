import numpy as np
import pandas as pd
from scipy.fft import fft, dct, idct
import cv2
import math
from PIL import Image as im

# DCT
# -> create_cosine_matrix
# -> perform_DCT
# -> quantize
PI = 3.14


class DCT:
    def __init__(self, image_rows, image_cols):
        self.image_rows = image_rows
        self.image_cols = image_cols
        self.C = [[0.0] * self.image_cols for i in range(self.image_rows)]
        self.Ct = [[0.0] * self.image_cols for i in range(self.image_rows)]
        self.intermediate = [[0.0] * self.image_cols for i in range(self.image_rows)]
        self.DCT = [[0.0] * self.image_cols for i in range(self.image_rows)]
        self.quantum = [[0] * self.image_cols for i in range(self.image_rows)]
        self.quantizedDCT = [[0] * self.image_cols for i in range(self.image_rows)]

    def create_cosine_matrix(self):
        matrix_dict = {"C": [], "Ct": []}
        for j in range(0, self.image_rows):
            self.C[0][j] = 1.0 / math.sqrt(self.image_rows)
            self.Ct[j][0] = self.C[0][j]

        for i in range(1, self.image_rows):
            for j in range(0, self.image_cols):
                self.C[i][j] = math.sqrt(2.0 / self.image_rows) * math.cos(
                    ((2 * j + 1) * i * PI) / (2.0 * self.image_rows)
                )
                self.Ct[j][i] = self.C[i][j]

        matrix_dict["C"] = self.C
        matrix_dict["Ct"] = self.Ct
        return matrix_dict

    def perform_DCT(self, input_arr):
        for i in range(self.image_rows):
            for j in range(self.image_cols):
                for k in range(
                    self.image_rows
                ):  # either self.image_rows or self.image_cols
                    self.intermediate[i][j] += self.Ct[i][k] * input_arr[k][j]

        for i in range(self.image_rows):
            for j in range(self.image_cols):
                for k in range(self.image_rows):
                    self.DCT[i][j] += self.intermediate[i][k] * self.C[k][j]
        return self.DCT

    def quantize(self, QUALITY):
        # Defining the quantization matrix
        for i in range(0, self.image_rows):
            for j in range(0, self.image_cols):
                self.quantum[i][j] = 1 + ((1 + i + j) * QUALITY)

        # Final updated DCT matrix with quantization
        for i in range(0, self.image_rows):
            for j in range(0, self.image_cols):
                self.quantizedDCT[i][j] = self.DCT[i][j] / self.quantum[i][j]

        # Converting elements to integers
        for i in range(0, self.image_rows):
            for j in range(0, self.image_cols):
                self.quantizedDCT[i][j] = int(self.quantizedDCT[i][j])
        return self.quantizedDCT


# Edge detection
# ->is_edge
# ->check_features


class Edge_Detection:
    def __init__(self, image_rows, image_cols) -> None:
        self.image_rows = image_rows
        self.image_cols = image_cols

    def read_image(self, IMAGE):
        img = cv2.imread(IMAGE)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        edges = cv2.Canny(
            image=img_blur, threshold1=100, threshold2=200
        )  # Canny Edge Detection
        input_arr = img_gray[0 : self.image_rows, 0 : self.image_cols]
        edges = edges[0 : self.image_rows, 0 : self.image_cols]
        return input_arr, edges

    def canny_edge_detection(self, img_gray):
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        edges = cv2.Canny(
            image=img_blur, threshold1=100, threshold2=200
        )  # Canny Edge Detection
        input_arr = img_gray[0 : self.image_rows, 0 : self.image_cols]
        edges = edges[0 : self.image_rows, 0 : self.image_cols]
        return input_arr, edges


class Range_Domain_Processing:
    def __init__(
        self,
        image_rows,
        image_cols,
        hex_squares_case1,
        hex_squares_case2,
        hex_squares_case3,
        input_arr,
        domain_size,
        range_size,
        edges,
    ) -> None:
        self.image_rows = image_rows
        self.image_cols = image_cols
        self.hex_squares_case1 = hex_squares_case1
        self.hex_squares_case2 = hex_squares_case2
        self.hex_squares_case3 = hex_squares_case3
        self.input_arr = input_arr
        self.domain_size = domain_size
        self.range_size = range_size
        self.edges = edges

    def is_edge(self, block):
        if np.sum(block) > 0:
            return True
        return False

    def check_features(self, block):
        features = {"edge": False}
        if self.is_edge(block):
            features["edge"] = True
        return features

    def convert_domain_to_4x4(self, domain_block):
        rows = 0
        cols = 0
        fourx4 = []
        for i in range(self.range_size):
            for j in range(self.range_size):
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
        fourx4 = fourx4.reshape(self.range_size, self.range_size)
        return fourx4

    def find_case(self, i):
        case_info = {"case_no": {}, "hex_count": 0}
        if i % 3 == 0:
            case_info["case_no"] = self.hex_squares_case1
            case_info["hex_count"] = 11
        elif i % 3 == 1:
            case_info["case_no"] = self.hex_squares_case2
            case_info["hex_count"] = 12
        else:
            case_info["case_no"] = self.hex_squares_case3
            case_info["hex_count"] = 11
        return case_info

    def find_hex_intensity(self, block, case):
        intensity = {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
            10: 0,
            11: 0,
            12: 0,
        }
        for key in case:
            sq_pixel_sum = 0
            sum_of_weight = 0
            for i in range(0, len(case[key]), 2):
                ele = int(block[int(case[key][i][0])][int(case[key][i][1])])
                sum_of_weight += case[key][i + 1]
                sq_pixel_sum += ele * case[key][i + 1]
            sq_pixel_sum /= sum_of_weight
            intensity[key] = int(sq_pixel_sum)
        return intensity

    def find_contrast(self, range_intensity, domain_intensity, case_info):
        sum_prod = 0
        sum_r = 0
        sum_d = 0
        sum_d2 = 0
        for key in domain_intensity:
            sum_prod += range_intensity[key] * domain_intensity[key]
            sum_r += range_intensity[key]
            sum_d += domain_intensity[key]
            sum_d2 += domain_intensity[key] * domain_intensity[key]
        s = ((case_info["hex_count"] * sum_prod) - (sum_d * sum_r)) / (
            (case_info["hex_count"] * (sum_d2)) - sum_d2
        )
        # final rms_error results in NaN if denominator becomes 0, have to handle this
        return s

    def find_brightness(self, range_intensity, domain_intensity, case_info, contrast):
        sum_r = 0
        sum_d = 0
        for key in domain_intensity:
            sum_r += range_intensity[key]
            sum_d += domain_intensity[key]
        o = (sum_r - (contrast * sum_d)) / case_info["hex_count"]
        return o

    def find_rms_error(self, range_intensity, domain_intensity, contrast, brightness):
        R = 0
        for key in range_intensity:
            x = contrast * domain_intensity[key] + brightness - range_intensity[key]
            R += pow(x, 2)
        return math.sqrt(R)

    def create_range_pool(self):
        no_of_range_in_row = (int)(self.image_cols / 4)
        row = 0
        col = 0
        all_ranges = []
        all_range_indexes = []
        edge_ranges = []
        edge_ranges_indexes = []
        for i in range(no_of_range_in_row):
            for j in range(no_of_range_in_row):
                temp_range = self.input_arr[
                    row : row + self.range_size, col : col + self.range_size
                ]
                all_ranges.append(temp_range)
                all_range_indexes.append((row, col))
                # classifying ranges
                # print(f"Checking features on this array {edges[row:row+4, col:col+4]}")
                features_range = self.check_features(
                    self.edges[row : row + self.range_size, col : col + self.range_size]
                )
                if features_range["edge"] == True:
                    edge_ranges.append(temp_range)
                    edge_ranges_indexes.append((row, col))
                col += self.range_size
            row += self.range_size
            col = 0

        print(f"\n Printing length of all_range: {len(all_ranges)}")
        print(f"\n Length of edge_ranges list = {len(edge_ranges)}")
        return all_ranges, all_range_indexes, edge_ranges, all_range_indexes

    def create_domain_pool(self):
        no_of_domain_in_row = self.image_cols - self.domain_size + 1
        temp_input_arr = self.input_arr
        edge_domains = []
        edge_domain_indexes = []
        not_edge_domains = []
        not_edge_domain_indexes = []

        temp_domain = [[0] * self.domain_size for i in range(self.domain_size)]

        row_overlaps = 0
        col_overlaps = 0
        while col_overlaps < no_of_domain_in_row:
            while row_overlaps < no_of_domain_in_row:
                temp_domain = [[0] * self.domain_size for i in range(self.domain_size)]
                for i in range(self.domain_size):
                    for j in range(self.domain_size):
                        temp_domain[i][j] = self.input_arr[i + col_overlaps][
                            j + row_overlaps
                        ]
                features_range = self.check_features(
                    self.edges[
                        col_overlaps : col_overlaps + self.domain_size,
                        row_overlaps : row_overlaps + self.domain_size,
                    ]
                )
                if features_range["edge"]:
                    edge_domains.append(temp_domain)
                    edge_domain_indexes.append((col_overlaps, row_overlaps))
                else:
                    not_edge_domains.append(temp_domain)
                    not_edge_domain_indexes.append((col_overlaps, row_overlaps))
                row_overlaps += 1
            col_overlaps += 1
            row_overlaps = 0
        print(f"\n Printing length of edge_domains: {len(edge_domains)}")
        print(f"\n Printing length of not_edge_domains: {len(not_edge_domains)}")
        return (
            edge_domains,
            edge_domain_indexes,
            not_edge_domains,
            not_edge_domain_indexes,
        )


class File_Operations(Range_Domain_Processing):
    def __init__(
        self,
        image_rows,
        image_cols,
        hex_squares_case1,
        hex_squares_case2,
        hex_squares_case3,
        input_arr,
        domain_size,
        range_size,
        edges,
    ) -> None:
        super().__init__(
            image_rows,
            image_cols,
            hex_squares_case1,
            hex_squares_case2,
            hex_squares_case3,
            input_arr,
            domain_size,
            range_size,
            edges,
        )

    def save_image(self, image_name):
        image_name = im.fromarray(
            self.input_arr[0 : self.image_rows, 0 : self.image_cols]
        )
        image_name.save("image_name.png")

    def set_encoding_file_data(
        self, range_index, domain_index, contrast, brightness, best_rms, isEdge, index
    ):
        encoding_data = pd.DataFrame(
            {
                "Range_index_row": range_index[0],
                "Range_index_col": range_index[1],
                "Domain_index_row": domain_index[0],
                "Domain_index_col": domain_index[1],
                "contrast": contrast,
                "brightness": brightness,
                "best_rms": best_rms,
                "isEdge": isEdge,
            },
            index=[index],
        )
        return encoding_data

    def create_dataframe(
        self,
        all_range,
        all_ranges_indexes,
        edge_ranges_indexes,
        not_edge_domains,
        not_edge_domains_indexes,
        edge_domains_indexes,
        edge_domains,
    ):
        # print(f"\n Printing length of all_range: {len(all_range)}")
        # print(f"\n Printing length of all_ranges_indexes: {len(all_ranges_indexes)}")
        # print(f"\n Printing length of edge_ranges_indexes: {len(edge_ranges_indexes)}")
        # print(f"\n Printing length of not_edge_domains: {len(not_edge_domains)}")
        # print(
        #     f"\n Printing length of not_edge_domains_indexes: {len(not_edge_domains_indexes)}"
        # )
        # print(
        #     f"\n Printing length of edge_domains_indexes: {len(edge_domains_indexes)}"
        # )
        # print(f"\n Printing length of edge_domains: {len(edge_domains)}")

        encoding_file_data = []
        encoding_file_index = 0
        for i in range(len(all_range)):
            print(f"############ Range {i} ############")
            case_info = self.find_case(i)
            best_rms_error = 1000000000
            range_index = (0, 0)
            domain_index = (0, 0)
            contrast = 0
            brightness = 0
            range_intensity = self.find_hex_intensity(
                all_range[i], case_info["case_no"]
            )
            # print(f"Range Intensity = {range_intensity}")
            if all_ranges_indexes[i] in edge_ranges_indexes:
                # print("in if")
                for j in range(len(not_edge_domains)):
                    # print("inside not edge domain")
                    domain_block = np.array(not_edge_domains[j])
                    domain_block = self.convert_domain_to_4x4(domain_block)
                    domain_intensity = self.find_hex_intensity(
                        domain_block, case_info["case_no"]
                    )
                    # print(f"Domain Intensity = {domain_intensity}")
                    contrast = self.find_contrast(
                        range_intensity, domain_intensity, case_info
                    )
                    brightness = self.find_brightness(
                        range_intensity, domain_intensity, case_info, contrast
                    )
                    rms_error = self.find_rms_error(
                        range_intensity, domain_intensity, contrast, brightness
                    )
                    # print(f"error:{rms_error}")
                    if rms_error < best_rms_error:
                        best_rms_error = rms_error
                        range_index = all_ranges_indexes[i]
                        domain_index = not_edge_domains_indexes[j]
                encoding_file_index += 1
                temp_data = self.set_encoding_file_data(
                    range_index,
                    domain_index,
                    contrast,
                    brightness,
                    best_rms_error,
                    False,
                    encoding_file_index,
                )
                # print(temp_data)
                encoding_file_data.append(temp_data)
            else:
                # print("in else")
                for j in range(len(edge_domains)):
                    # print("Inside edge domain")
                    domain_block = np.array(edge_domains[j])
                    domain_block = self.convert_domain_to_4x4(domain_block)
                    domain_intensity = self.find_hex_intensity(
                        domain_block, case_info["case_no"]
                    )
                    # print(f"Domain Intensity = {domain_intensity}")
                    contrast = self.find_contrast(
                        range_intensity, domain_intensity, case_info
                    )
                    brightness = self.find_brightness(
                        range_intensity, domain_intensity, case_info, contrast
                    )
                    rms_error = self.find_rms_error(
                        range_intensity, domain_intensity, contrast, brightness
                    )
                    # print(f"error:{rms_error}")
                    if rms_error < best_rms_error:
                        best_rms_error = rms_error
                        range_index = all_ranges_indexes[i]
                        domain_index = edge_domains_indexes[j]
                encoding_file_index += 1
                temp_data = self.set_encoding_file_data(
                    range_index,
                    domain_index,
                    contrast,
                    brightness,
                    best_rms_error,
                    True,
                    encoding_file_index,
                )
                # print(temp_data)
                encoding_file_data.append(temp_data)
        return encoding_file_data

    def save_encoding_file(self, encoding_file_data, file_name):
        final_data = pd.concat(encoding_file_data)
        print(final_data.head())
        final_data.to_csv(file_name, sep="\t", encoding="utf-8", index=False)
