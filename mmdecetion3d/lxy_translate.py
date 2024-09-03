import open3d as o3d
import numpy as np

def translate_point_cloud(pcd, new_origin):
    # Calculate the translation vector
    translation_vector = -np.array(new_origin)
    # Apply the translation to the point cloud
    translated_pcd = pcd.translate(translation_vector, relative=False)
    return translated_pcd

# Load the point cloud
input_pcd_path = "0820_3_chairs_pointcloud.pcd"  # Replace with your point cloud file path
pcd = o3d.io.read_point_cloud(input_pcd_path)

# New origin coordinates
new_origin = (-3.101107,9.239416,-0.904918)

# Translate the point cloud to the new origin
translated_pcd = translate_point_cloud(pcd, new_origin)

# Save the translated point cloud
output_pcd_path = "0820_3_chairs_pointcloud_translat.pcd"  # Replace with your desired output file path
o3d.io.write_point_cloud(output_pcd_path, translated_pcd)

print(f"Translated point cloud saved to '{output_pcd_path}'.")
