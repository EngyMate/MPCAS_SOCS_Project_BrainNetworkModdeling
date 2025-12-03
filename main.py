import data_management as dm

PATH_fMRI_data= "C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\NetworkModelling\\data\\fMRI{0}"
PATH_DTI_data = "C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\NetworkModelling\\data\\DTI{0}"
PATH_BNA_atlas = "C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\NetworkModelling\\data\\bna_atlas.xlsx"

fMRI_data_files = dm.list_files_in_folder(PATH_fMRI_data.format(""))
fMRI_data = dm.load_excel_to_numpy(PATH_fMRI_data.format(f"\\{fMRI_data_files[0]}"))
print(fMRI_data[0,0])

DTI_data_files = dm.list_files_in_folder(PATH_DTI_data.format(""))
DTI_data = dm.load_excel_to_numpy(PATH_DTI_data.format(f"\\{DTI_data_files[0]}"))
print(DTI_data[0,0])

"""
Brainnetome Atlas
BNA Label
BNA Notes
BrainMesh_ICBM152.nv
"""
BNA_atlas = dm.load_mixed_excel(PATH_BNA_atlas)
print(BNA_atlas[0][:])
