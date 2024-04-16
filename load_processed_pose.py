import numpy as np
from scipy.spatial.transform import Rotation as R

def read_pose_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Split the data into groups of two lines, ignoring possible comments and empty lines
    image_data = [lines[i:i+2] for i in range(0, len(lines), 2) if not lines[i].startswith('#') and not lines[i].strip() == '']

    # Dictionary to hold image poses
    image_poses = {}

    for image_info, _ in image_data:
        # Parse the first line of each image data which contains the pose and other info
        parts = image_info.split()
        image_id = parts[0]
        qw, qx, qy, qz = map(float, parts[1:5])  # Quaternion components
        tx, ty, tz = map(float, parts[5:8])      # Translation components
        camera_id = parts[8]
        image_name = parts[9]

        # Convert quaternion to rotation matrix
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()

        # Combine rotation and translation into a transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation
        transformation_matrix[:3, 3] = [tx, ty, tz]

        # Store the transformation matrix with the image name as the key
        image_poses[image_name] = transformation_matrix

    return image_poses

# Usage:
poses_dict = read_pose_data("colmap_models/dense/_data_dycheck_iphone_handwavy_rgb_2x_txt/images.txt")
sorted_pose_dict = {key: poses_dict[key] for key in sorted(poses_dict)}
sorted_poses = [sorted_pose_dict[key] for key in sorted_pose_dict]
print('length:', len(sorted_poses))

# pred rel pose for samples from sorted poses
indices = [1, 2]
sample_pred_poses = [sorted_poses[i] for i in range(len(indices))]
pred_rel_poses = [sorted_poses[i] @ np.linalg.inv(sorted_poses[i+1]) for i in range(len(indices)-1)]
print('Pred rel poses:', pred_rel_poses)




