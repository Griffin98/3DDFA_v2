# coding: utf-8

__author__ = 'cleardusk'

import numpy as np

from .tddfa_util import _to_ctype
from .functions import get_suffix

header_temp = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
element face {}
property list uchar int vertex_indices
end_header
"""


def ser_to_ply_single(ver_lst, tri, height, wfp, reverse=True):
    suffix = get_suffix(wfp)

    for i, ver in enumerate(ver_lst):
        wfp_new = wfp.replace(suffix, f'_{i + 1}{suffix}')

        n_vertex = ver.shape[1]
        n_face = tri.shape[0]
        header = header_temp.format(n_vertex, n_face)

        with open(wfp_new, 'w') as f:
            f.write(header + '\n')
            for i in range(n_vertex):
                x, y, z = ver[:, i]
                if reverse:
                    f.write(f'{x:.2f} {height-y:.2f} {z:.2f}\n')
                else:
                    f.write(f'{x:.2f} {y:.2f} {z:.2f}\n')
            for i in range(n_face):
                idx1, idx2, idx3 = tri[i]  # m x 3
                if reverse:
                    f.write(f'3 {idx3} {idx2} {idx1}\n')
                else:
                    f.write(f'3 {idx1} {idx2} {idx3}\n')

        print(f'Dump tp {wfp_new}')


def ser_to_ply_multiple(ver_lst, tri, height, wfp, reverse=True):
    n_ply = len(ver_lst)  # count ply

    if n_ply <= 0:
        return

    n_vertex = ver_lst[0].shape[1]
    n_face = tri.shape[0]
    header = header_temp.format(n_vertex * n_ply, n_face * n_ply)

    with open(wfp, 'w') as f:
        f.write(header + '\n')

        for i in range(n_ply):
            ver = ver_lst[i]
            for j in range(n_vertex):
                x, y, z = ver[:, j]
                if reverse:
                    f.write(f'{x:.2f} {height - y:.2f} {z:.2f}\n')
                else:
                    f.write(f'{x:.2f} {y:.2f} {z:.2f}\n')

        for i in range(n_ply):
            offset = i * n_vertex
            for j in range(n_face):
                idx1, idx2, idx3 = tri[j]  # m x 3
                if reverse:
                    f.write(f'3 {idx3 + offset} {idx2 + offset} {idx1 + offset}\n')
                else:
                    f.write(f'3 {idx1 + offset} {idx2 + offset} {idx3 + offset}\n')

    print(f'Dump tp {wfp}')


def get_colors(img, ver):
    h, w, _ = img.shape
    ver[0, :] = np.minimum(np.maximum(ver[0, :], 0), w - 1)  # x
    ver[1, :] = np.minimum(np.maximum(ver[1, :], 0), h - 1)  # y
    ind = np.round(ver).astype(np.int32)
    colors = img[ind[1, :], ind[0, :], :] / 255.  # n x 3

    return colors.copy()


def ser_to_obj_single(img, ver_lst, tri, height, wfp):
    suffix = get_suffix(wfp)

    n_face = tri.shape[0]
    for i, ver in enumerate(ver_lst):
        colors = get_colors(img, ver)

        n_vertex = ver.shape[1]

        wfp_new = wfp.replace(suffix, f'_{i + 1}{suffix}')

        with open(wfp_new, 'w') as f:
            for i in range(n_vertex):
                x, y, z = ver[:, i]
                f.write(
                    f'v {x:.2f} {height - y:.2f} {z:.2f} {colors[i, 2]:.2f} {colors[i, 1]:.2f} {colors[i, 0]:.2f}\n')
            for i in range(n_face):
                idx1, idx2, idx3 = tri[i]  # m x 3
                f.write(f'f {idx3 + 1} {idx2 + 1} {idx1 + 1}\n')

        print(f'Dump tp {wfp_new}')


def ser_to_obj_multiple(img, ver_lst, tri, height, wfp):
    n_obj = len(ver_lst)  # count obj

    if n_obj <= 0:
        return

    n_vertex = ver_lst[0].shape[1]
    n_face = tri.shape[0]

    with open(wfp, 'w') as f:
        for i in range(n_obj):
            ver = ver_lst[i]
            colors = get_colors(img, ver)

            for j in range(n_vertex):
                x, y, z = ver[:, j]
                f.write(
                    f'v {x:.2f} {height - y:.2f} {z:.2f} {colors[j, 2]:.2f} {colors[j, 1]:.2f} {colors[j, 0]:.2f}\n')

        for i in range(n_obj):
            offset = i * n_vertex
            for j in range(n_face):
                idx1, idx2, idx3 = tri[j]  # m x 3
                f.write(f'f {idx3 + 1 + offset} {idx2 + 1 + offset} {idx1 + 1 + offset}\n')

    print(f'Dump tp {wfp}')


def write_obj_with_texture(vertices, triangles, uv_coords, height, wfp):
    """ Save 3D face model with texture represented by texture map.
    Ref: https://github.com/patrikhuber/eos/blob/bd00155ebae4b1a13b08bf5a991694d682abbada/include/eos/core/Mesh.hpp
    Args:
        vertices: shape = (nver, 3)
        triangles: shape = (ntri, 3)
        uv_coords: shape = (nver, 3) max value<=1
        height: height of the image
        wfp: file pointer
    """
    if wfp.split('.')[-1] != 'obj':
        wfp = wfp + '.obj'
    mtl_name = wfp.replace('.obj', '.mtl')
    texture_name = wfp.replace('.obj', '_texture.jpg')

    triangles = triangles.copy()
    triangles += 1  # mesh lab start with 1

    # write obj
    with open(wfp, 'w') as f:
        # first line: write mtlib(material library)
        s = "mtllib {}\n".format(mtl_name)
        f.write(s)

        # write vertices
        for i in range(vertices.shape[1]):
            x, y, z = vertices[:, i]
            s = 'v {} {} {}\n'.format(x, height - y, z)
            f.write(s)

        # write uv coords
        for i in range(uv_coords.shape[0]):
            s = 'vt {} {}\n'.format(uv_coords[i, 0], 1 - uv_coords[i, 1])
            f.write(s)

        f.write("usemtl FaceTexture\n")

        # write f: ver ind/ uv ind
        for i in range(triangles.shape[0]):
            s = 'f {}/{} {}/{} {}/{}\n'.format(triangles[i, 2], triangles[i, 2], triangles[i, 1], triangles[i, 1],
                                               triangles[i, 0], triangles[i, 0])
            f.write(s)

    # write mtl
    with open(mtl_name, 'w') as f:
        f.write("newmtl FaceTexture\n")
        s = 'map_Kd {}\n'.format(texture_name)  # map to image
        f.write(s)


ser_to_ply = ser_to_ply_multiple  # ser_to_ply_single
ser_to_obj = ser_to_obj_multiple  # ser_to_obj_multiple
