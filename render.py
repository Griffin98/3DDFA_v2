import argparse
import cv2
import numpy as np
import os
import yaml
from tqdm import tqdm

from TDDFA import TDDFA
from utils.render import render_with_colors

from MeshPyIO.Wavefront import WavefrontOBJ

TDDFA_CONFIG = "configs/mb1_120x120.yml"

def main(args):
    print("\n\n***** Rendering *****\n")

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = yaml.load(open(TDDFA_CONFIG), Loader=yaml.SafeLoader)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'

    tddfa = TDDFA(gpu_mode=True, **cfg)

    input_vertices = np.load(os.path.join(args.input_dir, "vertices.npy"))

    for index in tqdm(range(input_vertices.shape[0])):

        ver = input_vertices[index, ...].astype(np.float32)

        frame_path = os.path.join(args.input_dir, 'frame-{0:06d}'.format(index))

        frame = cv2.imread(os.path.join(frame_path, "frame.jpg"))
        face_obj_path = os.path.join(frame_path, "face.obj")
        face_obj = WavefrontOBJ.load_obj(face_obj_path)

        colors = face_obj.get_verts_colors()
        ver = [ver]

        render_img = render_with_colors(frame, ver, tddfa.tri, colors, show_flag=False)

        cv2.imwrite(os.path.join(args.output_dir, 'frame-{0:06d}.jpg'.format(index)), render_img[..., ::-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render reconstructed faces")
    parser.add_argument("-i", "--input_dir", required=True, type=str, help="Input directory of reconstructed faces")
    parser.add_argument("-o", "--output_dir", required=True, type=str, help="Output directory path")
    args = parser.parse_args()
    main(args)