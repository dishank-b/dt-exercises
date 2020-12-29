import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


#Gave coordinates in cam2 frame I guess. Not sure though. 
def TwoDtoThreeD_cam2(samples, matrix):
    Threedboxes = []
    # CAM_HEIGHT = 1.72
    CAM_HEIGHT = 1.65
    for bottom_xy in samples:
        bottom_xy = np.append(bottom_xy, 1.0).T
        n = np.asarray([0, 1.0, 0])
        numerator = np.linalg.solve(matrix, bottom_xy)
        denominator = n.dot(numerator)
        bboxBottomIn3D = np.reshape(CAM_HEIGHT * (numerator/denominator), (3,1))
        Threedboxes.append(bboxBottomIn3D[[0,2]])
    return np.array(Threedboxes)

# Coordinates in cam0 frame
def TwoDtoThreeD_cam0(samples, p_matrix, rect_matrix=0):
    Threedboxes = []
    # CAM_HEIGHT = 1.72
    CAM_HEIGHT = 1.65
    A = np.array([[1,0,0],[0,0,CAM_HEIGHT], [0,1,0], [0,0,1]])
    p_matrix = np.matmul(p_matrix, A)
    for bottom_xy in samples:
        bottom_xy = np.append(bottom_xy, 1.0).T
        numerator = np.linalg.solve(p_matrix, bottom_xy)
        #coords in rectified frame
        bboxBottomIn3D = numerator/numerator[2]
        Threedboxes.append(bboxBottomIn3D[:2])
    return np.array(Threedboxes)


def read_matrix(path):
    file = open(path)
    lines = file.read().splitlines()
    p_matrix = lines[2].split(':', 1)[1].strip()
    p_matrix = np.array(p_matrix.split(' '), dtype='float').reshape((3,4))
    
    r_matrix = lines[4].split(' ', 1)[1].strip()
    r_matrix = np.array(r_matrix.split(' '), dtype='float').reshape((3,3))
    r_matrix = np.vstack((r_matrix, np.array([0,0,0])))
    r_matrix = np.column_stack((r_matrix, np.array([0,0,0,1])))
    # matrix = matrix[:,:-1]

    return np.matmul(p_matrix, r_matrix)
            
def ground_project(instances, path="./datasets/kitti_tracking/training/calib/0001.txt"):
    means = instances.pred_boxes
    means = [[(x[0]+x[2])/2,x[3]] for x in means]
    sigmas = instances.pred_variance
    sigmas = [[(y[0]+y[2])/4, y[3]] for y in sigmas]
    # K_matrix, rect_matrix = read_matrix(path)
    matrix = read_matrix(path)
    gd_means=[]
    gd_sigmas = []

    for mean, sigma in zip(means, sigmas): #iterating gthrough each of the box
        samples = np.random.normal(np.full((10,2), mean), sigma) #Draw 10 samples
        # ground_points = TwoDtoThreeD(samples, K_matrix)
        ground_points = TwoDtoThreeD_cam0(samples, matrix)
        gd_mean = ground_points.mean(axis=0, keepdims=False)
        gd_std = ground_points.std(axis=0, keepdims=False)
        gd_means.append(np.squeeze(gd_mean))
        gd_sigmas.append(np.squeeze(gd_std))

    return np.asarray(gd_means), np.asarray(gd_sigmas)
