import numpy as np

import data_management as dm


def extract_coord():
    coordinates_arr = []
    for row in BNA_atlas:
        coordinates_arr.append([row[3], row[4], row[5]])
    return np.array(coordinates_arr)


if __name__ == "__main__":
    BNA_atlas = dm.load_mixed_excel(dm.PATH_excel_BNA_atlas)

    coordinates = extract_coord()
