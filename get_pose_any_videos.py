import subprocess
import concurrent.futures
import glob
import os
import argparse
from utils.lib import *

import shutil
import time
import pycolmap

# Function to parse command-line arguments

def list_of_strings(arg):
    return arg.split(',')

def parse_args():
    parser = argparse.ArgumentParser(description='Get camera poses for any video')
    parser.add_argument('--input_video_frame_paths', type=list_of_strings, required=True,
                        help='list of video paths to be processed')
    parser.add_argument('--sampled_images_path', type=str, default='sampled_frames',
                        help='Path to the output directory containing sampled image files.')
    parser.add_argument('--homography_overlap', type=float, default=0.9,
                        help='Threshold of the homography to sample new frames, higher value samples more images')
    parser.add_argument('--max_concurrent', type=int, default=8,
                        help='Max number of concurrent processes')
    parser.add_argument('--sparse_reconstuctions_root', type=str, default='colmap_models/sparse',
                        help='Path to the sparsely reconstructed models.')
    parser.add_argument('--logs_path', type=str, default='logs/sparse/out_logs_terminal',
                        help='Path to store the log files.')
    parser.add_argument('--summary_path', type=str, default='logs/sparse/out_summary',
                        help='Path to store the summary files.')
    parser.add_argument('--gpu_index', type=int, default=0,
                        help='Index of the GPU to use.')

    parser.add_argument('--dense_reconstuctions_root', type=str, default='colmap_models/dense',
                        help='Path to the densely registered models.')

    return parser.parse_args()

args = parse_args()
input_video_frame_paths = args.input_video_frame_paths
gpu_index = args.gpu_index
print('Input videos: ', input_video_frame_paths)

########################################################################################
# Select spaarse frames by homography
########################################################################################

print('Selecting sparse frames by homography...')

params_list = []
for video_frame_path in input_video_frame_paths:
    video_name = video_frame_path.replace('/','_')
    added_run = ['--src', video_frame_path, '--dst_file', '%s/%s_selected_frames.txt'%(args.sampled_images_path,video_name), '--overlap', str(args.homography_overlap)]
    if not added_run in params_list:
        params_list.append(added_run)
                
if params_list:
    max_concurrent = args.max_concurrent
    # Create a process pool executor with a maximum of K processes
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent)

    # Submit the tasks to the executor
    results = []
    for i in range(len(params_list)):
        future = executor.submit(run_script, 'homography_filter/filter.py', params_list[i % len(params_list)])
        results.append(future)

    # Wait for all tasks to complete
    for r in concurrent.futures.as_completed(results):
        try:
            r.result()
        except Exception as e:
            print(f"Error occurred: {e}")

    # Shut down the executor
    executor.shutdown()

########################################################################################
# Reconstruct sparse frames
########################################################################################

print('Reconstructing sparse frames...')

# videos_list = read_lines_from_file(args.input_videos)
# videos_list = sorted(videos_list)
print('GPU: %d' % (gpu_index))
os.makedirs(args.logs_path, exist_ok=True)
os.makedirs(args.summary_path, exist_ok=True)
os.makedirs(args.sparse_reconstuctions_root, exist_ok=True)

i = 0
for video_frame_path in input_video_frame_paths:
    video_name = video_frame_path.replace('/','_')
    if (not os.path.exists(os.path.join(args.sparse_reconstuctions_root, '%s' % video_name))):
        # check the number of images in this video
        with open(os.path.join(args.sampled_images_path, '%s_selected_frames.txt' % (video_name)), 'r') as f:
            lines = f.readlines()
            num_lines = len(lines)
        # print(f'The file {video_name} contains {num_lines} lines.')
        if num_lines < 100000: #it's too large, so it would take days! 
            print('Processing: ', video_name, '(',num_lines, 'images )')
            start_time = time.time()

            # Define the path to the shell script
            script_path = 'scripts/reconstruct_sparse.sh'

            # Create a unique copy of the script
            script_copy_path = video_name + '_' + str(os.getpid()) + '_' + os.path.basename(script_path)
            shutil.copy(script_path, script_copy_path)

            # Output file
            output_file_path = os.path.join(args.logs_path, script_copy_path.replace('.sh', '.out'))


            # Define the command to execute the script
            command = ["bash", script_copy_path, video_name,args.sparse_reconstuctions_root, video_frame_path, args.sampled_images_path, args.summary_path, str(gpu_index)]
            print('command:', command)
            # Open the output file in write mode
            with open(output_file_path, 'w') as output_file:
                # Run the command and capture its output in real time
                process = subprocess.Popen(command, stdout=output_file, stderr=subprocess.PIPE, text=True)
                while True:
                    output = process.stderr.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        output_file.write(output)
                        output_file.flush()

            # Once the script has finished running, you can delete the copy of the script
            os.remove(script_copy_path)

            #In case of having multiple models, will keep the one with largest number of images and rename it as 0
            reg_images = keep_model_with_largest_images(os.path.join(args.sparse_reconstuctions_root,video_name,'sparse'))
            if reg_images > 0:
                print(f"Registered_images/total_images: {reg_images}/{num_lines} = {round(reg_images/num_lines*100)}%")
            else:
                print('The video reconstruction fails!! no reconstruction file is found!')

            print("Execution time:  %s minutes" % round((time.time() - start_time)/60, 2))
            print('-----------------------------------------------------------')

    i += 1


########################################################################################
# Reconstruct dense frames
########################################################################################

print('Reconstructing dense frames...')

i = 0
# for video in videos_list:
#     pre = video.split('_')[0]
for video_frame_path in input_video_frame_paths:
    video_name = video_frame_path.replace('/','_')
    if (not os.path.exists(os.path.join(args.dense_reconstuctions_root, '%s' % video_name))):
        # check the number of images in this video
        num_lines = len(glob.glob(os.path.join(video_frame_path,'*.jpg')) + glob.glob(os.path.join(video_frame_path,'*.png')))

        print('Processing: ', video_name, '(',num_lines, 'images )')
        start_time = time.time()

        # Define the path to the shell script
        script_path = 'scripts/register_dense.sh'

        # Create a unique copy of the script
        script_copy_path = video_name + '_' + str(os.getpid()) + '_' + os.path.basename(script_path)
        shutil.copy(script_path, script_copy_path)

        # Output file
        output_file_path = os.path.join(args.logs_path, script_copy_path.replace('.sh', '.out'))


        # Define the command to execute the script
        command = ["bash", script_copy_path, video_name, args.sparse_reconstuctions_root, args.dense_reconstuctions_root, video_frame_path, args.summary_path, str(gpu_index)]
        print('command:', command)

        # Open the output file in write mode
        with open(output_file_path, 'w') as output_file:
            # Run the command and capture its output in real time
            process = subprocess.Popen(command, stdout=output_file, stderr=subprocess.PIPE, text=True)
            while True:
                output = process.stderr.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output_file.write(output)
                    output_file.flush()

        # Once the script has finished running, you can delete the copy of the script
        os.remove(script_copy_path)

        reg_images = get_num_images(os.path.join(args.dense_reconstuctions_root, video_name))
        if reg_images > 0:
            print(f"Registered_images/total_images: {reg_images}/{num_lines} = {round(reg_images/num_lines*100)}%")
        else:
            print('The video reconstruction fails!! no colmap files are found!')

        print("Execution time:  %s minutes" % round((time.time() - start_time)/60, 2))
        print('-----------------------------------------------------------')

    i += 1
