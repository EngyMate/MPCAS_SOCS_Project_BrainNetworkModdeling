import pandas as pd
import numpy as np
import os

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


import pandas as pd


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

# Example usage
if __name__ == "__main__":
    file_path = "data.xlsx"
    matrix = load_excel_to_numpy(file_path)
    print("Loaded NumPy matrix:")
    print(matrix)
