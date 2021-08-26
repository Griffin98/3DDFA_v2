import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
import yaml

from TDDFA import TDDFA
from utils.uv import uv_tex
from utils.serialization import write_obj_with_texture, ser_to_obj
from utils.utils import realign

LANDMARKS_FILE = 'landmarks_68.xyz'
TDDFA_CONFIG = "configs/mb1_120x120.yml"

def main(args):
    print("\n\n***** 3D Align *****\n")

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = yaml.load(open(TDDFA_CONFIG), Loader=yaml.SafeLoader)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'

    tddfa = TDDFA(gpu_mode=True, **cfg)

    actor_vertices = np.load(os.path.join(args.actor_dir, "vertices.npy"))
    dubber_vertices = np.load(os.path.join(args.dubber_dir, "vertices.npy"))

    len_actor_vertices = actor_vertices.shape[0]
    len_dubber_vertices = dubber_vertices.shape[0]

    n = min(len_actor_vertices, len_dubber_vertices)
    print("Will use {} frames from actor and dubber".format(n))

    vertices = np.empty(shape=(0, dubber_vertices.shape[1], dubber_vertices.shape[2]))

    for index in tqdm(range(n)):
        out_wfp = os.path.join(args.output_dir, 'frame-{0:06d}'.format(index))
        os.makedirs(out_wfp, exist_ok=True)

        dubber_ver = dubber_vertices[index, ...]

        actor_frame_path = os.path.join(args.actor_dir, 'frame-{0:06d}'.format(index))
        dubber_frame_path = os.path.join(args.dubber_dir, 'frame-{0:06d}'.format(index))

        actor_landmarks_68 = np.loadtxt(os.path.join(actor_frame_path, LANDMARKS_FILE))
        dubber_landmarks_68 = np.loadtxt(os.path.join(dubber_frame_path, LANDMARKS_FILE))

        # Calculate Affine transformation
        c, R, t = realign(dubber_landmarks_68.T, actor_landmarks_68.T)

        # 3D Align
        algined_ver = np.empty_like(dubber_ver.T)
        for i, v in enumerate(dubber_ver.T):
            algined_ver[i] = ((c * R).dot(v.reshape(-1, 1)) + t.reshape(-1, 1)).reshape(1, -1)

        vertices = np.append(vertices, algined_ver.T[np.newaxis, ...], axis=0)

        # New 68 landmarks
        landmarks_68 = np.reshape(np.reshape(algined_ver, (-1, 1))[tddfa.bfm.keypoints], (-1, 3))
        np.savetxt(os.path.join(out_wfp, 'landmarks_68.xyz'), landmarks_68)

        # Save obj with texture
        ver_lst = [algined_ver.T]
        actor_frame = cv2.imread(os.path.join(actor_frame_path, "frame.jpg"))
        dubber_frame = cv2.imread(os.path.join(dubber_frame_path, "frame.jpg"))

        texture, uv_coords = uv_tex(dubber_frame[..., ::-1], [dubber_ver], tddfa.tri, show_flag=False)
        cv2.imwrite(os.path.join(out_wfp, "texture.png"), texture[..., ::-1])

        # .obj
        wfp = os.path.join(out_wfp, "face.obj")
        write_obj_with_texture(ver_lst[0], tddfa.tri, uv_coords / 256, actor_frame.shape[0], wfp)

        # Save actor frame for render purpose
        cv2.imwrite(os.path.join(out_wfp, "frame.jpg"), actor_frame)

    np.save(os.path.join(args.output_dir, "vertices.npy"), vertices, allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Align Dubber w.r.t Actor')
    parser.add_argument('-a', '--actor_dir', required=True, type=str, help="Input actor directory")
    parser.add_argument('-d', '--dubber_dir', required=True, type=str, help="Input dubber directory")
    parser.add_argument('-o', '--output_dir', required=True, type=str, help='Output directory')
    args = parser.parse_args()
    main(args)