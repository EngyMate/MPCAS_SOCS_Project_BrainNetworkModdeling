import pandas as pd
import numpy as np
import os

PATH_excel_fMRI_data = "C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\NetworkModelling\\data\\fMRI{0}"
PATH_excel_DTI_data = "C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\NetworkModelling\\data\\DTI{0}"
PATH_excel_BNA_atlas = "C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\NetworkModelling\\data\\bna_atlas.xlsx"
PATH_DTI_data = "C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\data\\DTI\\{0}"
PATH_fMRI_data = "C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\data\\fMRI\\{0}"

fMRI_data_shapes = [(4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246), (4800, 246)]
DTI_data_shape = [(246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246), (246, 246)]


def load_excel_to_numpy(file_path, sheet_name=0):
    """
    Loads a 2D matrix of numbers from an Excel file into a NumPy array.

    Args:
        file_path (str): Path to the Excel file.
        sheet_name (str or int, optional): Name or index of the sheet to load. Defaults to 0.

    Returns:
        np.ndarray: 2D NumPy array of numbers.
    """
    # Read Excel file into pandas DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl', header=None)

    # Convert DataFrame to NumPy array
    matrix = df.to_numpy(dtype=float)  # forces numeric conversion

    return matrix

def list_files_in_folder(folder_path):
    """
    Returns a list of all files in a folder (excluding subfolders).

    Args:
        folder_path (str): Path to the folder.

    Returns:
        list[str]: List of filenames in the folder.
    """
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return files

def count_files_in_folder():
    """
    Returns a list of all files in a folder (excluding subfolders).

    Args:
        folder_path (str): Path to the folder.

    Returns:
        list[str]: List of filenames in the folder.
    """
    folder_path = PATH_DTI_data.format("")
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    num_dti_files = len(files)

    folder_path = PATH_fMRI_data.format("")
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    num_fmri_files = len(files)

    num_files = dict( DTI = num_dti_files, fMRI = num_fmri_files )

    return num_files

def load_mixed_excel(file_path, sheet_name=0):
    """
    Loads rows of strings and floats from an Excel file.

    Args:
        file_path (str): Path to the Excel file.
        sheet_name (str or int, optional): Sheet name or index. Defaults to 0.

    Returns:
        list[list]: 2D list where each row contains strings and floats as in Excel.
    """
    # Read Excel without forcing headers or numeric conversion
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl', header=None)

    # Convert DataFrame to a list of lists
    data = df.values.tolist()

    return data

def load_fMRI_data(file_nr):
    files = list_files_in_folder(PATH_fMRI_data.format(""))
    fMRI_data = np.fromfile(PATH_fMRI_data.format(files[file_nr]), dtype=float).reshape(fMRI_data_shapes[file_nr])
    return fMRI_data

def load_DTI_data(file_nr):
    files = list_files_in_folder(PATH_DTI_data.format(""))
    DTI_data = np.fromfile(PATH_DTI_data.format(files[file_nr]), dtype=float).reshape(DTI_data_shape[file_nr])
    return DTI_data

def excel_to_bin():
    excel_fMRI_data_files = list_files_in_folder(PATH_excel_fMRI_data.format(""))
    shapes_fMRI = []

    for file in excel_fMRI_data_files:
        fMRI_data = load_excel_to_numpy(PATH_excel_fMRI_data.format(f"\\{excel_fMRI_data_files[0]}"))
        fMRI_data.tofile(PATH_fMRI_data.format(file.replace("xlsx", "bin")))
        shapes_fMRI.append(fMRI_data.shape)

    print(shapes_fMRI)

    excel_DTI_data_files = list_files_in_folder(PATH_excel_DTI_data.format(""))

    shapes_DTI = []

    for file in excel_DTI_data_files:
        DTI_data = load_excel_to_numpy(PATH_excel_DTI_data.format(f"\\{excel_DTI_data_files[0]}"))
        DTI_data.tofile(PATH_DTI_data.format(file.replace("xlsx", "bin")))
        shapes_DTI.append(DTI_data.shape)

    print(shapes_DTI)


# Example usage
if __name__ == "__main__":
    pass
    #excel_to_bin()
