import os
import warnings
import os.path as osp
import gin
import tensorflow as tf
from absl import app, flags, logging
import numpy as np
from matplotlib import pyplot as pl
import torchvision.transforms as tvf
from scipy.spatial.transform import Rotation as R

from dycheck import core, geometry

# dycheck configs
flags.DEFINE_multi_string(
    "gin_configs",
    "iphone_dataset.gin",
    "Gin config files.",
)
flags.DEFINE_multi_string("gin_bindings", None, "Gin parameter bindings.")
FLAGS = flags.FLAGS

DATA_ROOT = "/data/dycheck"
# SEQUENCE = 'backpack'
# SEQUENCE = 'creeper'
SEQUENCE = 'handwavy'
# SEQUENCE = 'haru-sit'
# SEQUENCE = 'mochi-high-five'
# SEQUENCE = 'pillow'
# SEQUENCE = 'sriracha-tree'


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
    

def prepare_dataset():
    tf.config.experimental.set_visible_devices([], "GPU")

    core.parse_config_files_and_bindings(
        config_files=FLAGS.gin_configs,
        bindings=(FLAGS.gin_bindings or [])
        + [
            "Config.engine_cls=None",
            "Config.model_cls=None",
            f"SEQUENCE='{SEQUENCE}'",
            f"iPhoneParser.data_root='{DATA_ROOT}'",
            f"iPhoneDatasetFromAllFrames.split='train'",
            f"iPhoneDatasetFromAllFrames.training=False",
            f"iPhoneDatasetFromAllFrames.bkgd_points_batch_size=0",
        ],
        skip_unknown=True,
    )

    config_str = gin.config_str()
    logging.info(f"*** Configuration:\n{config_str}")

    config = core.Config()

    dataset = config.dataset_cls()
    return dataset

def load_dycheck(dataset, idx):
    sample = dataset[idx]
    rgb = sample['rgb']
    camera = dataset.cameras[idx]
    intrin = camera.intrin
    extrin = camera.extrin
    return rgb, intrin, extrin

def calculate_pose_diff(gt_rel_pose, pred_rel_pose):

    # calculate translation diff
    gt_t = gt_rel_pose[:3, 3]
    pred_t = pred_rel_pose[:3, 3]
    t_diff = np.linalg.norm(gt_t - pred_t)

    # calculate rotation diff
    gt_r = gt_rel_pose[:3, :3]
    pred_r = pred_rel_pose[:3, :3]
    rotation_diff_matrix = gt_r.T @ pred_r

    # Convert rotation matrix to rotation vector (in radians)
    rotation_diff_rad = R.from_matrix(rotation_diff_matrix).as_rotvec()
    # norm of the rotation vector is the angle of rotation
    rotation_diff_rad = np.linalg.norm(rotation_diff_rad)

    # Convert angle to degrees
    rotation_diff_deg = np.rad2deg(rotation_diff_rad)

    return t_diff, rotation_diff_deg

def main(_):

    # import pdb; pdb.set_trace()

    # load dycheck dataset
    dataset = prepare_dataset()
    print('Sequence name: ', SEQUENCE)
    print('Dataset length:', len(dataset))

    # get samples
    # indices = [1, 2]
    indices = list(range(len(dataset)))

    samples = [(rgb, intrin, extrin) for rgb, intrin, extrin in [load_dycheck(dataset, idx) for idx in indices]]

    # gt rel pose for samples
    extrins = [sample[2] for sample in samples]
    gt_rel_poses = [extrins[i] @ np.linalg.inv(extrins[i+1]) for i in range(len(extrins)-1)]
    # print('GT rel poses:', gt_rel_poses)

    # load processed pose data
    poses_dict = read_pose_data("colmap_models/dense/_data_dycheck_iphone_{}_rgb_2x_/images.txt".format(SEQUENCE))
    sorted_pose_dict = {key: poses_dict[key] for key in sorted(poses_dict)}
    sorted_poses = [sorted_pose_dict[key] for key in sorted_pose_dict]
    print('num of processed frames:', len(sorted_poses))

    # pred rel pose for samples from sorted poses
    sample_pred_poses = [sorted_poses[i] for i in range(len(indices))]
    pred_rel_poses = [sorted_poses[i] @ np.linalg.inv(sorted_poses[i+1]) for i in range(len(indices)-1)]
    # print('Pred rel poses:', pred_rel_poses)

    # measure pose diff between gt and pred
    pose_diffs = [calculate_pose_diff(gt_rel_pose, pred_rel_pose) for gt_rel_pose, pred_rel_pose in zip(gt_rel_poses, pred_rel_poses)]
    # print('Pose diffs:', pose_diffs)

    # analysis pose diff
    t_diffs = [pose_diff[0] for pose_diff in pose_diffs]
    rotation_diffs = [pose_diff[1] for pose_diff in pose_diffs]
    # mean and std of translation diff
    t_diff_mean = np.mean(t_diffs)
    t_diff_std = np.std(t_diffs)
    # mean and std of rotation diff
    rotation_diff_mean = np.mean(rotation_diffs)
    rotation_diff_std = np.std(rotation_diffs)
    print('Translation diff mean:', t_diff_mean, 'std:', t_diff_std)
    print('Rotation diff mean:', rotation_diff_mean, 'std:', rotation_diff_std)

if __name__ == "__main__":
    app.run(main)


