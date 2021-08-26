import cv2
import os
import numpy as np
import math
import os
import glob
import scipy
from numpy import linalg as LA
from scipy.signal import savgol_filter
from sklearn import preprocessing
from scipy.spatial import distance
from tqdm import tqdm
from scipy.spatial import cKDTree as KDTree
from scipy.linalg import svd

def extract_frames(video_path, output_path):
    image_path_list = []

    video_file = cv2.VideoCapture(video_path)
    success, image = video_file.read()
    if not success:
        print('Error: unable to read %s with OpenCV' % video_file)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    index = 0
    print("\n\t ******* extracting frames *******")
    while success:
        image_path_list.append(os.path.join(output_path, "frame-{}.jpg".format(index)))
        cv2.imwrite(os.path.join(output_path, "frame-{}.jpg".format(index)), image)  # save frame as JPEG file
        success, image = video_file.read()
        print('extracting frame {}'.format(index))
        index += 1

    return image_path_list




class opencv:
    @staticmethod
    def conv_np_cv2(pos, image):
        """
        convert numpy 2D position to opencv position in the image
        """
        height, width, channels = image.shape
        x = np.round(pos[0] * width).astype("int")
        y = np.round(height - pos[1] * height).astype("int")
        return (x, y)

    @staticmethod
    def conv_cv2_np(pos, image):
        """
        convert opencv position numpy 3D position
        """
        height, width, channels = image.shape
        x = pos[0] / width
        y = height - pos[1] / height
        z = pos[2]
        return np.array([x, y, z]).reshape(1, 3)


class system:
    @staticmethod
    def log(text, debug=False):
        if debug:
            print ("{}".format(text))
    @staticmethod
    def prepare_folder(_path):
        if not os.path.exists(_path):
            os.makedirs(_path)


class MathOperation:

    @staticmethod
    def distance(point_1, point_2):
        return  np.linalg.norm(point_1 - point_2)

    @staticmethod
    def get_vectors_Angle(vect1, vect2):
        """
            | A·B | = | A | | B | COS(θ)
            | A×B | = | A | | B | SIN(θ)
        """

        #vect1 =  MathOperation.swap(vect1,1,2)
        #vect2 =  MathOperation.swap(vect2,1,2)
        A = vect1[1][:2] - vect1[0][:2]
        B = vect2[1][:2] - vect2[0][:2]

        A = A / np.linalg.norm(A)
        B = B / np.linalg.norm(B)
        cross = np.cross(A, B)
        dot = np.dot(A, B)
        tang = math.atan2(cross, dot)
        return tang

    def init_rotations(_vertices, angle=90):
        object_vertices = _vertices.copy()
        angle = math.radians(angle)
        rotation = np.array(
            [
                [1, 0, 0, 0],
                [0, math.cos(angle), -math.sin(angle), 0],
                [0, math.sin(angle), math.cos(angle), 0],
                [0, 0, 0, 1]
            ])
        for index, vertex in enumerate(object_vertices):
            _vertex = rotation @ np.array([vertex[0], vertex[1], vertex[2], 1])
            object_vertices[index] = [_vertex[0], _vertex[1], _vertex[2]]
        return object_vertices

    @staticmethod
    def index_min_dist(point, points_list):
        min_distance = 5000000
        index_min_distance = -1
        for index in range(points_list.shape[0]):
            # finding the closest vertex
            distance = MathOperation.distance(points_list[index],point)
            if distance < min_distance:
                index_min_distance = index
                min_distance = distance
        return index_min_distance

    @staticmethod
    def closest_vertex(node, nodes):
        closest_index = distance.cdist([node], nodes).argmin()
        if np.linalg.norm(nodes[closest_index]-node)>0.000001:
            node = nodes[closest_index].copy()
        return closest_index, node

    @staticmethod
    def swap(matrix, index_1, index_2):
        _matrix = matrix.copy()
        _matrix[:, index_1] = -1 * matrix[:, index_2]
        _matrix[:, index_2] = -1 * matrix[:, index_1]
        return _matrix

    @staticmethod
    def svd_denoise(data, r_param):
        """
        Summary:
           Apply SVD like decomposition and reduce the data noise by keeping only r columns.
        :param data: Face data matrix
        :param r_param: Number of eigenvectors to keep.
        :return:  the noise reduced face data matrix.
        """

        # Decompose the matrix
        U1, s1, v1 = scipy.linalg.svd(data, full_matrices=False)
        print("Data decomposition: \n\t U.shape={},\n\t Sigma.shape={}, \n\t V.shape={}".format(
            U1.shape, s1.shape, v1.shape))

        new_matrix = np.dot(np.dot(U1[:, :r_param], np.diag(s1)[:r_param, :r_param]), v1[:r_param, :])
        print("SVD denoised data shape {}".format(U1.shape, new_matrix.shape))

        return new_matrix

    @staticmethod
    def smooth_savgol_filter(data_to_smooth, w=3, p=1):
        """
        summary:
            Calculate the Savitzky-Golay Filter on the clumns values.
        params:
            * data_to_smooth: matrix (N*M): The 'N' input vertices through 'M' frames to smooth. We can see the lines (N) as
            vertices and columns (M) as the position of a these vertices in each frame.
            * w: int : The length of the filter window which represent the number of coefficients vertices . window_length
            must be a positive odd integer.
            * p: int : The order of the polynomial used to fit the samples. polyorder must be less than window_length.
        return:
            smoothed_curve: the smoothed curves.
        Reference:
            Savitzky-Golay Filter : http://www.statistics4u.com/fundstat_eng/cc_filter_savgolay.html
            numpy: https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.signal.savgol_filter.html
        """

        smoothed_curve = np.ndarray(shape=data_to_smooth.shape, dtype=np.float_)
        for row in tqdm(range(0, data_to_smooth.shape[0])):
            smoothed_curve[row, :] = savgol_filter(data_to_smooth[row, :], window_length=w, polyorder=p)
        return smoothed_curve

    @staticmethod
    def SVD_pseudo_inverse(matrix, compute_full_matrix = False, debug=False):
        """
         compute the pseudo inverse of a matrix.
         ref: # reference https://www.johndcook.com/blog/2018/05/05/svd/
        :param matrix:
        :param compute_full_matrix:
        :return:
        """
        n, p = matrix.shape
        U, sigma, Vt = LA.svd(matrix, full_matrices=compute_full_matrix)
        Sigma = np.zeros((min(n, p), min(n, p)))
        Sigma[:min(n, p), :min(n, p)] = np.diag(1 / sigma)

        system.log("u:{}".format(U.shape),debug=debug)
        system.log("sigma:{}".format(sigma.shape),debug=debug)
        system.log("Vt:{}".format(Vt.shape,debug=debug))
        # Pen-roose psedoinverse
        pseudo_inverse = (Vt.T.dot(Sigma).dot(U.T))
        return pseudo_inverse

    @staticmethod
    def minmax_norm(data):
        data=np.array(data)
        min_max_scaler = preprocessing.MinMaxScaler()
        return min_max_scaler.fit_transform(data.reshape(data.shape[0], 1)).reshape(data.shape[0])

    @staticmethod
    def normal(v1, v2, v3):
        N = np.cross(v2-v1, v3-v1)
        N = N / np.linalg.norm(N)
        return N
    @staticmethod
    def interpolate(a, b, nb, lst=np.empty(shape=(0, 3), dtype=np.float_)):
        """
        Create nb points between a and b.
        :param a: array of 3 positions (x, y, z) represents the first point.
        :param b: array of 3 positions (x, y, z) represents the first point.
        :param nb: nb of interpolated points.
        :param lst: if provided the interpolated points will appended to it other wise it will contains only the result.
        """
        dist = np.array(b - a).reshape(1, 3)
        _nb = nb + 1
        for pt in range(1, _nb):
            new_pt = a + dist * pt / _nb
            lst = np.append(lst, new_pt, axis=0)
        return lst


class Barycentric:
    @staticmethod
    def compute_coordinates(p, a, b, c):
        v0 = b - a
        v1 = c - a
        v2 = p - a
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        denom = (d00 * d11 - d01 * d01)
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return [u, v, w]

    @staticmethod
    def compute_position(b_c, a, b, c):
        if len(a) == 3:
            x = b_c[0] * a[0] + b_c[1] * b[0] + b_c[2] * c[0]
            y = b_c[0] * a[1] + b_c[1] * b[1] + b_c[2] * c[1]
            z = b_c[0]*a[2] + b_c[1]*b[2] + b_c[2]*c[2]
            return [x, y, z]
        else:
            x = b_c[0] * a[0] + b_c[1] * b[0] + b_c[2] * c[0]
            y = b_c[0] * a[1] + b_c[1] * b[1] + b_c[2] * c[1]
            return [x, y]


class Search:
    """
    search using trees
    """
    def __init__(self, array):
        self.data = KDTree(array.T)

    def find(self, array):
        """
        find the closest index of an array in the data
        return: index of the closest
        """
        # create one new vector and find distance and index of closest
        if isinstance(array, str):
            query_w = np.loadtxt(array).reshape(1, -1)
        else:
            query_w = array.reshape(1, -1)

        d, i = self.data.query(query_w)
        return i[0]


class data:
    def __init__(self, files_paths, weight_length):
        self.files_paths = files_paths.copy()
        self.weights = np.zeros(shape=(weight_length, 0), dtype=np.float64)

        for path in files_paths:
            weights = np.loadtxt(path, dtype=np.float64).reshape(-1, 1)
            self.weights = np.append(self.weights , weights, axis=1)

        self.search_engine = Search(self.weights)

    def find(self, query_data):
        index = self.search_engine.find(query_data)
        file_name = os.path.basename(self.files_paths[index]).split(sep="_")[0]
        return index, file_name

def realign(X, Y):
    """
    compute the affine transformation of the set of points X to Y.
    :param X:
    :param Y:
    :return:
    """
    m, n =  X.shape
    mx = np.mean(X, axis=1).reshape(-1,1)
    my = np.mean(Y, axis=1).reshape(-1,1)
    Xc = X - np.tile(mx, [1,n])
    Yc = Y - np.tile(my, [1,n])
    sx = np.mean(np.sum(np.square(Xc), axis=0))
    sy = np.mean(np.sum(np.square(Yc), axis=0))
    Sxy = np.divide(Yc.dot(Xc.T), n)

    U, D, V = svd(Sxy)

    r = np.linalg.matrix_rank(Sxy)
    d = np.linalg.det(Sxy)

    S = np.eye(m)
    # print(S)

    if r > m-1:
        if d < 0:
            S[m - 1, m - 1] = -1
    elif r == m:
        if (np.linalg.det(U) * np.linalg.det(V)) < 0:
            S[m - 1, m - 1] = -1
    else:
        print('Insufficient rank in covariance to determine rigid transform')
        R = np.array([[1, 0], [0, 1]])
        c = 1
        t = np.array([[0], [0]])
        return (c, R, t)

    R = U.dot(S).dot(V)
    c = np.sum(np.diagonal(np.diag(D).dot(S))) / sx
    #c = np.trace(np.diag(D).dot(S))
    t = my - c * R.dot(mx)
    return c, R, t