# import os
# import numpy as np

# # Path to the folder containing .npz files
# folder_path = './trainset'

# # Iterate through each .npz file in the folder
# for file_name in os.listdir(folder_path):
#     if file_name.endswith('.npz'):
#         file_path = os.path.join(folder_path, file_name)
        
#         # Load the .npz file
#         data = np.load(file_path)
#         image = data['image']
#         label = data['label']
        
#         # Replace all occurrences of 4 with 3 in image and label arrays
#         image[image == 4] = 3
#         label[label == 4] = 3
        
#         # Save the modified arrays back to the .npz file
#         np.savez(file_path, image=image, label=label)
        
#         print(f"Modified {file_name}")

# print("Modification complete.")

import os
import numpy as np
import h5py

# Path to the folder containing .npy.h5 files
folder_path = './testset_brats'

# Iterate through each .npy.h5 file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.npy.h5'):
        file_path = os.path.join(folder_path, file_name)
        
        # Load the .npy.h5 file
        with h5py.File(file_path, 'r+') as f:
            image = np.array(f['image'])
            label = np.array(f['label'])
            
            # Replace all occurrences of 4 with 3 in image and label arrays
            image[image == 4] = 3
            label[label == 4] = 3
            
            # Save the modified arrays back to the .npy.h5 file
            del f['image']
            del f['label']
            f.create_dataset('image', data=image)
            f.create_dataset('label', data=label)
            
        print(f"Modified {file_name}")

print("Modification complete.")
