__author__ = 'Dhyey Patel'

import argparse
import imageio
import numpy as np
import os
import yaml
from collections import deque
from tqdm import tqdm

from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from TDDFA import TDDFA
from utils.functions import get_suffix
from utils.serialization import write_obj_with_texture
from utils.uv import uv_tex

import cv2

TDDFA_CONFIG = "configs/mb1_120x120.yml"

def main(args):
    print("\n\n***** Reconstructing ***** \n")

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = yaml.load(open(TDDFA_CONFIG), Loader=yaml.SafeLoader)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'

    face_boxes = FaceBoxes_ONNX()
    tddfa = TDDFA(gpu_mode=True, **cfg)

    # Process Input
    file_name = args.input_file.split('/')[-1]
    reader = imageio.get_reader(args.input_file)

    # Moving average smoothing
    n_prev = int(args.smoothing_window / 2)
    n_next = int(args.smoothing_window / 2)
    n = n_prev + n_next + 1
    queue_ver = deque()
    queue_frame = deque()

    queue_smoothed_ver = deque()
    queue_param_lst = deque()

    # Variable to track previous vertex
    prev_ver = None

    # Run
    for index, frame in tqdm(enumerate(reader)):

        frame_bgr = frame[..., ::-1] # RGB -> BGR

        if index == 0:
            # detect
            boxes = face_boxes(frame_bgr)
            boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
            ver, _ = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
            ver = ver[0]

            # refine
            param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy="landmark")
            ver, _ = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
            ver = ver[0]

            # padding queue
            for _ in range(n_prev):
                queue_ver.append(ver.copy())
            queue_ver.append(ver.copy())

            for _ in range(n_prev):
                queue_frame.append(frame_bgr.copy())

            queue_param_lst.append(param_lst)
        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, [prev_ver], crop_policy="landmark")

            # Fail safe to avoid frame being lost
            roi_box = roi_box_lst[0]
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2000:
                boxes = face_boxes(frame_bgr)
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

            ver, _ = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
            ver = ver[0]

            queue_ver.append(ver.copy())
            queue_frame.append(frame_bgr.copy())

            queue_param_lst.append(param_lst)

        prev_ver = ver # track

        if len(queue_ver) >= n:
            ver_ave = np.mean(queue_ver, axis=0)
            queue_smoothed_ver.append(ver_ave)

            queue_ver.popleft()

    # we will lost the last n_next frames, still padding
    for _ in range(n_next):
        queue_ver.append(ver.copy())

        ver_ave = np.mean(queue_ver, axis=0)
        queue_smoothed_ver.append(ver_ave)

        queue_ver.popleft()

    print("\n\n***** Saving Output ***** \n")

    # Save the output
    w, h = ver.shape
    vertices = np.empty(shape=(0, w, h))


    for index in tqdm(range(len(queue_smoothed_ver))):
        out_wfp = os.path.join(args.output_dir, 'frame-{0:06d}'.format(index))
        os.makedirs(out_wfp, exist_ok=True)

        # frame
        frame = queue_frame.popleft()
        cv2.imwrite(os.path.join(out_wfp, "frame.jpg"), frame)

        ver = queue_smoothed_ver.popleft()
        vertices = np.append(vertices, ver[np.newaxis, ...], axis=0)

        # 3D Facial Keypoints
        landmarks_68 = np.reshape(np.reshape(ver.T, (-1, 1))[tddfa.bfm.keypoints], (-1, 3))
        np.savetxt(os.path.join(out_wfp, 'landmarks_68.xyz'), landmarks_68)

        # UV texture
        ver_lst = [ver]
        texture, uv_coords = uv_tex(frame[..., ::-1], ver_lst, tddfa.tri, show_flag=False)
        cv2.imwrite(os.path.join(out_wfp, "texture.png"), texture[..., ::-1])

        # .obj
        wfp = os.path.join(out_wfp, "face.obj")
        write_obj_with_texture(ver_lst[0], tddfa.tri, uv_coords / 256, frame.shape[0], wfp)


    np.save(os.path.join(args.output_dir, "vertices.npy"), vertices, allow_pickle=False)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Reconstruct video')
    parser.add_argument('-i', '--input_file', required=True, type=str, help="Input video file")
    parser.add_argument('-o', '--output_dir', required=True, type=str, help="Output directory")
    parser.add_argument('-w', '--smoothing_window', default=2, type=int, help='Smoothing window')
    args = parser.parse_args()
    main(args)