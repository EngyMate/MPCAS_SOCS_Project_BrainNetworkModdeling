import data_management as dm
import numpy as np

print(dm.count_files_in_folder())

fMRI_1 = dm.load_fMRI_data(0)
print(fMRI_1[1,0])

"""
Brainnetome Atlas
BNA Label
BNA Notes
BrainMesh_ICBM152.nv
"""
#BNA_atlas = dm.load_mixed_excel(dm.PATH_excel_BNA_atlas)
#print(BNA_atlas[0][:])


