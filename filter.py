"""
import os
import numpy as np
import torch
import h5py

from datasets.dataset_synapse import Synapse_dataset

split = "test"
list_dir = "./lists/lists_brats"
data_dir = "../datasets/brats20/ppbrats-test"

if split=="train":
    sample_list_path = os.path.join(list_dir, split+'.txt')
    sample_list = open(sample_list_path).readlines()
elif split=="test":
    sample_list_path = os.path.join(list_dir, split+'_vol.txt')
    sample_list = open(sample_list_path).readlines()


def get_numpy_unique_values(input_array):
    unique_values = np.unique(input_array)
    return unique_values


data_unique_values = [0, 1, 2, 4]

def splitting_knowing():
    if split == "train":
        new_sample_list = []
        for idx, slice_name_line in enumerate(sample_list):
            slice_name = slice_name_line.strip('\n')
            data_path = os.path.join(data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            unique = get_numpy_unique_values(label)
            
            if not np.array_equal(unique, data_unique_values):
                print(f"Slice name {slice_name} has sparse labels and has unique vals: {unique}")
            
                #os.remove(data_path) 
            else:
                new_sample_list.append(slice_name_line)  

        with open(sample_list_path, 'w') as f:
            f.writelines(new_sample_list)

    elif split=="test":
        new_sample_list=[]
        for idx. slice_name_line in enumerate(sample_list):
            vol_name = sample_list[idx].strip('\n')
            filepath = data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
            image = image.astype(np.float32)  # You can adjust the type based on your needs
            label = label.astype(np.float32)  # You can adjust the type based on your needs
            unique = get_numpy_unique_values(label)

            if not np.array_equal(unique, data_unique_values):
                print(f"Slice name {slice_name} has sparse labels and has unique vals: {unique}")
            else:
                new_sample_list.append(slice_name_line)
        
        with open(sample_list_path, 'w') as f:
            f.writelines(new_sample_list)
"""
"""

for file_name in os.listdir(data_dir):
    if file_name.endswith(".npz"):
        file_path = os.path.join(data_dir, file_name)
        data = np.load(file_path)
        label = data['label']
        
        unique = get_numpy_unique_values(label)
        if not np.array_equal(unique, data_unique_values):
            os.remove(file_path)
            print(f"removed {file_path}")

    elif file_name.endswith(".npy.h5"):
        file_path = os.path.join(data_dir, file_name)
        data = h5py.File(file_path)
        image, label = data['image'][:], data['label'][:]
        image = image.astype(np.float32)  # You can adjust the type based on your needs
        label = label.astype(np.float32)  # You can adjust the type based on your needs
        unique = get_numpy_unique_values(label)

        if not np.array_equal(unique, data_unique_values):
            os.remove(file_path)
            print(f"removed {file_path}")




"""


import os
import numpy as np

# Path to the folder containing .npz files
input_folder = '../datasets/brats20/ppbrats-train'

# List of unique data labels
unique_data_labels = [0, 1, 2, 4]  # Example list, replace with your actual labels

# Iterate through each .npz file in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.npz'):
        file_path = os.path.join(input_folder, file_name)
        
        # Load the .npz file
        data = np.load(file_path)
        
        # Check if the 'label' array contains all unique labels
        label_array = data['label']
        if not all(label in label_array for label in unique_data_labels):
            # If any unique label is missing, remove the file
            os.remove(file_path)
            print(f"Removed {file_name} from directory.")
        else:
            print(f"{file_name} contains all unique labels.")

print("Label check complete.")

# import os
# import h5py

# # Path to the folder containing .npy.h5 files
# input_folder = '../datasets/brats20/ppbrats-test'

# # List of unique data labels
# unique_data_labels = [0, 1, 2, 4]  # Example list, replace with your actual labels

# # Iterate through each .npy.h5 file in the input folder
# for file_name in os.listdir(input_folder):
#     if file_name.endswith('.npy.h5'):
#         file_path = os.path.join(input_folder, file_name)
        
#         # Load the .npy.h5 file
#         with h5py.File(file_path, 'r') as hf:
#             if 'image' in hf and 'label' in hf:
#                 label_array = hf['label'][:]
#                 if not all(label in label_array for label in unique_data_labels):
#                     # If any unique label is missing, remove the file
#                     os.remove(file_path)
#                     print(f"Removed {file_name} from directory.")
#                 else:
#                     print(f"{file_name} contains all unique labels.")
#             else:
#                 print(f"{file_name} does not contain 'image' or 'label' keys.")

# print("Label check complete.")
