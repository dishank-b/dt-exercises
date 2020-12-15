
from collections import OrderedDict
from scipy.stats import multivariate_normal
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from math import floor, sqrt



class LaneFilterHistogramKF():
    """ Generates an estimate of the lane pose.

    TODO: Fill in the details

    Args:
        configuration (:obj:`List`): A list of the parameters for the filter

    """

    def __init__(self, **kwargs):
        param_names = [
            # TODO all the parameters in the default.yaml should be listed here.
            'mean_d_0',
            'mean_phi_0',
            'sigma_d_0',
            'sigma_phi_0',
            'sigma_d_pro',
            'sigma_phi_pro',
            'sigma_d_measure',
            'sigma_phi_measure',
            'delta_d',
            'delta_phi',
            'd_max',
            'd_min',
            'phi_max',
            'phi_min',
            'cov_v',
            'linewidth_white',
            'linewidth_yellow',
            'lanewidth',
            'min_max',
            'sigma_d_mask',
            'sigma_phi_mask',
            'range_min',
            'range_est',
            'range_max',
        ]

        for p_name in param_names:
            assert p_name in kwargs
            setattr(self, p_name, kwargs[p_name])

        self.mean_0 = np.array([self.mean_d_0, self.mean_phi_0])
        self.cov_0 = np.array([[self.sigma_d_0, 0.0], [0.0, self.sigma_phi_0]])

        self.Q = np.array([[self.sigma_d_pro**2, 0.0],[0.0, self.sigma_phi_pro**2]])
        self.R = np.array([[self.sigma_d_measure**2, 0.0],[0.0, self.sigma_phi_measure**2]])


        self.encoder_resolution = 0.0
        self.wheel_radius = 0.0
        self.baseline = 0.0
        self.initialized = False
        self.reset()

    def reset(self):
        self.mean_0 = [self.mean_d_0, self.mean_phi_0]
        self.cov_0 = [[self.sigma_d_0, 0], [0, self.sigma_phi_0]]
        self.belief = {'mean': self.mean_0, 'covariance': self.cov_0}

    def predict(self, dt, left_encoder_delta, right_encoder_delta):
        #TODO update self.belief based on right and left encoder data + kinematics
        if not self.initialized:
            return
        v_l = (left_encoder_delta*2*np.pi/(self.encoder_resolution*dt))*self.wheel_radius
        v_r = (right_encoder_delta*2*np.pi/(self.encoder_resolution*dt))*self.wheel_radius
        
        v_t = (v_l+v_r)/2.0
        omega_t = (v_r-v_l)/self.baseline

        d, phi = self.belief['mean']
        F = np.array([[1.0, v_t*dt*np.cos(phi)],[0.0, 1.0]])

        self.belief['mean'] = np.array([ d + v_t*dt*np.sin(phi), phi+omega_t*dt])
        self.belief['covariance'] = np.matmul(np.matmul(F ,self.belief['covariance']), F.T) + self.Q

        print("Predicted State - ")
        # print("v: {}, w: {}".format(v_t, omega_t))
        self.printState()

    def update(self, segments):
        # prepare the segments for each belief array    
        segmentsArray = self.prepareSegments(segments)
        # generate all belief arrays

        measurement_likelihood = self.generate_measurement_likelihood(
            segmentsArray)

        if isinstance(measurement_likelihood, type(None)):
            return
        
        # # Calcualting these just to compare. But these are not used in update
        # maxids = np.unravel_index(
        # measurement_likelihood.argmax(), measurement_likelihood.shape)
        # d_max = self.d_min + (maxids[0] + 0.5) * self.delta_d
        # phi_max = self.phi_min + (maxids[1] + 0.5) * self.delta_phi
        
        # Measure

        z_d, z_phi, d_std, phi_std = self.get_mean_var(measurement_likelihood)
        # z_d, z_phi, d_std, phi_std = self.get_mean_var_diff(measurement_likelihood)
        
        z = np.array([z_d, z_phi], dtype=np.float)
        R =  np.diag([d_std**2, phi_std**2]) + np.array([[1e-7, 2e-7],[5e-7, 6e-7]])

        print("Measurement-")
        print("d: {}, phi: {}".format(z_d, z_phi))
        print("Cov: ", R)
        # print("d_max {}, phi_max: {}".format(d_max, phi_max))

        # Update
        x_hat = self.belief['mean']
        p_hat = self.belief['covariance']

        K = np.matmul(p_hat, np.linalg.inv(p_hat+R))

        self.belief['mean'] = x_hat + np.matmul(K, z-x_hat)
        self.belief['covariance'] = np.matmul(np.eye(2)-K, p_hat)
        
        print("Updated State - ")
        self.printState()

    def getEstimate(self):
        return self.belief

    def printState(self):
        print("d: {}, phi: {}".format(self.belief["mean"][0], self.belief["mean"][1]))
        print("Cov: ", self.belief["covariance"])

    def generate_measurement_likelihood(self, segments):

        if len(segments) == 0:
            return None

        grid = np.mgrid[self.d_min:self.d_max:self.delta_d,
                                    self.phi_min:self.phi_max:self.delta_phi]

        # initialize measurement likelihood to all zeros
        measurement_likelihood = np.zeros(grid[0].shape)

        for segment in segments:
            d_i, phi_i, l_i, weight = self.generateVote(segment)

            # if the vote lands outside of the histogram discard it
            if d_i > self.d_max or d_i < self.d_min or phi_i < self.phi_min or phi_i > self.phi_max:
                continue

            i = int(floor((d_i - self.d_min) / self.delta_d))
            j = int(floor((phi_i - self.phi_min) / self.delta_phi))
            measurement_likelihood[i, j] = measurement_likelihood[i, j] + 1

        if np.linalg.norm(measurement_likelihood) == 0:
            return None

        # lastly normalize so that we have a valid probability density function

        measurement_likelihood = measurement_likelihood / \
            np.sum(measurement_likelihood)  
        return measurement_likelihood

    # def get_mean_var_diff(self, measurement_likelihood):
    #     ids = np.unravel_index(
    #         measurement_likelihood.argsort(axis=None)[-25:][::-1], measurement_likelihood.shape)
    #     d_max, phi_max = ids[0][0], ids[1][0]
    #     # print(d_max, phi_max)

    #     ds = []
    #     phis= []
    #     weights = []

    #     for i, j in zip(ids[0], ids[1]):
    #         if measurement_likelihood[i,j]>0 and abs(i-d_max)<=2 and abs(j-phi_max)<=2:
    #             ds.append(i)
    #             phis.append(j)
    #             weights.append(measurement_likelihood[i,j])

    #     # print(ds, phis)
        
    #     ds = np.array([self.d_min + (i + 0.5) * self.delta_d for i in ds])
    #     phis = np.array([self.phi_min + (j + 0.5) * self.delta_phi for j in phis])

    #     d_mean = np.average(ds, weights=weights)
    #     d_std = np.sqrt(np.average((ds-d_mean)**2, weights=weights))
    #     phi_mean = np.average(phis, weights=weights)
    #     phi_std = np.sqrt(np.average((phis-phi_mean)**2, weights=weights))

    #     return d_mean, phi_mean, d_std, phi_std

    def get_mean_var(self, measurement_likelihood):
        shape = measurement_likelihood.shape
        maxids = np.unravel_index(
                measurement_likelihood.argmax(), shape)
        d_max, phi_max = maxids[0], maxids[1]
        # print(d_max, phi_max)

        ds = []
        phis= []
        weights = []

        for i in range(max(d_max-2,0), min(d_max+3,shape[0])):
            for j in range(max(phi_max-2,0), min(phi_max+3,shape[1])):
                if measurement_likelihood[i,j]>0:
                    ds.append(i)
                    phis.append(j)
                    weights.append(measurement_likelihood[i,j])
        
        # print(ds, phis)
        
        ds = np.array([self.d_min + (i + 0.5) * self.delta_d for i in ds])
        phis = np.array([self.phi_min + (j + 0.5) * self.delta_phi for j in phis])

        d_mean = np.average(ds, weights=weights)
        d_std = np.sqrt(np.average((ds-d_mean)**2, weights=weights))
        phi_mean = np.average(phis, weights=weights)
        phi_std = np.sqrt(np.average((phis-phi_mean)**2, weights=weights))

        return d_mean, phi_mean, d_std, phi_std

    # generate a vote for one segment
    def generateVote(self, segment):
        p1 = np.array([segment.points[0].x, segment.points[0].y])
        p2 = np.array([segment.points[1].x, segment.points[1].y])
        t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)

        n_hat = np.array([-t_hat[1], t_hat[0]])
        d1 = np.inner(n_hat, p1)
        d2 = np.inner(n_hat, p2)
        l1 = np.inner(t_hat, p1)
        l2 = np.inner(t_hat, p2)
        if (l1 < 0):
            l1 = -l1
        if (l2 < 0):
            l2 = -l2

        l_i = (l1 + l2) / 2
        d_i = (d1 + d2) / 2
        phi_i = np.arcsin(t_hat[1])
        if segment.color == segment.WHITE:  # right lane is white
            if(p1[0] > p2[0]):  # right edge of white lane
                d_i = d_i - self.linewidth_white
            else:  # left edge of white lane

                d_i = - d_i

                phi_i = -phi_i
            d_i = d_i - self.lanewidth / 2

        elif segment.color == segment.YELLOW:  # left lane is yellow
            if (p2[0] > p1[0]):  # left edge of yellow lane
                d_i = d_i - self.linewidth_yellow
                phi_i = -phi_i
            else:  # right edge of white lane
                d_i = -d_i
            d_i = self.lanewidth / 2 - d_i

        # weight = distance
        weight = 1
        return d_i, phi_i, l_i, weight

    def get_inlier_segments(self, segments, d_max, phi_max):
        inlier_segments = []
        for segment in segments:
            d_s, phi_s, l, w = self.generateVote(segment)
            if abs(d_s - d_max) < 3*self.delta_d and abs(phi_s - phi_max) < 3*self.delta_phi:
                inlier_segments.append(segment)
        return inlier_segments

    # get the distance from the center of the Duckiebot to the center point of a segment
    def getSegmentDistance(self, segment):
        x_c = (segment.points[0].x + segment.points[1].x) / 2
        y_c = (segment.points[0].y + segment.points[1].y) / 2
        return sqrt(x_c**2 + y_c**2)

    # prepare the segments for the creation of the belief arrays
    def prepareSegments(self, segments):
        segmentsArray = []
        self.filtered_segments = []
        for segment in segments:

            # we don't care about RED ones for now
            if segment.color != segment.WHITE and segment.color != segment.YELLOW:
                continue
            # filter out any segments that are behind us
            if segment.points[0].x < 0 or segment.points[1].x < 0:
                continue

            self.filtered_segments.append(segment)
            # only consider points in a certain range from the Duckiebot for the position estimation
            point_range = self.getSegmentDistance(segment)
            if point_range < self.range_est:
                segmentsArray.append(segment)

        return segmentsArray