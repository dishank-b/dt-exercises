import numpy as np
import scipy.linalg
import torch

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    The 8-dimensional state space
        x1, y1, x2, y2, vx1, vy1, vx2, vy2
    contains the bounding box coridnates and their respective velocities.
    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    """

    def __init__(self, std_weight_position, std_weight_velocity):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = torch.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = torch.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

    def initiate(self, measurement, measurement_var):
        """Create track from unassociated measurement.
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        measurement_nosie: ndarray
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        device = measurement.device
        self._motion_mat = self._motion_mat.to(device)
        self._update_mat = self._update_mat.to(device)

        mean_pos = measurement
        mean_vel = torch.zeros_like(mean_pos)
        mean = torch.cat([mean_pos, mean_vel])


        vel_std = torch.tensor([10 * self._std_weight_velocity * (measurement[2]-measurement[0]),
                    10 * self._std_weight_velocity * (measurement[3]-measurement[1]),
                    10 * self._std_weight_velocity * (measurement[2]-measurement[0]),
                    10 * self._std_weight_velocity * (measurement[3]-measurement[1])], device=device)
        
        var = torch.cat([measurement_var, vel_std**2])

        # var = [
        #     2 * self._std_weight_position * measurement[3],
        #     2 * self._std_weight_position * measurement[3],
        #     1e-2,
        #     2 * self._std_weight_position * measurement[3],
        #     10 * self._std_weight_velocity * measurement[3],
        #     10 * self._std_weight_velocity * measurement[3],
        #     1e-5,
        #     10 * self._std_weight_velocity * measurement[3]]
        covariance = torch.diag(var)
        
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.
        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step. 
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        device = mean.device

        std_pos = torch.tensor([
            self._std_weight_position * (mean[2]- mean[0]),  
            self._std_weight_position * (mean[3]- mean[1]),
            self._std_weight_position * (mean[2]- mean[0]),
            self._std_weight_position * (mean[3]- mean[1])], device =device)

        std_vel = torch.tensor([
            self._std_weight_velocity * (mean[2]- mean[0]),
            self._std_weight_velocity * (mean[3]- mean[1]),
            self._std_weight_velocity * (mean[2]- mean[0]),
            self._std_weight_velocity * (mean[3]- mean[1])], device=device)

        motion_cov = torch.diag(torch.cat([std_pos, std_vel])**2)
        # motion_cov = 0


        mean = torch.matmul(self._motion_mat, mean)
 
        covariance = torch.chain_matmul(
            self._motion_mat, covariance, self._motion_mat.T) + motion_cov

        return mean, covariance

    def project(self, mean, covariance, measurement_noise=0):
        """Project state distribution to measurement space.
        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        device = mean.device

        mean = torch.matmul(self._update_mat, mean)
        covariance = torch.chain_matmul(
            self._update_mat, covariance, self._update_mat.T)

        return mean, covariance + measurement_noise

    def update(self, mean, covariance, measurement, measurement_noise):
        """Run Kalman filter correction step.
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance, measurement_noise)

        # chol_factor, lower = scipy.linalg.cho_factor(
        #     projected_cov, lower=True, check_finite=False)
        # kalman_gain = scipy.linalg.cho_solve(
        #     (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
        #     check_finite=False).T
        # innovation = measurement - projected_mean

        # new_mean = mean + np.dot(innovation, kalman_gain.T)
        # new_covariance = covariance - np.linalg.multi_dot((
        #     kalman_gain, projected_cov, kalman_gain.T))

        innovation = measurement - projected_mean
        kalman_gain = torch.chain_matmul(covariance, self._update_mat.T, torch.inverse(projected_cov))
        new_mean = mean + torch.matmul(kalman_gain, innovation)
        new_covariance = covariance - torch.chain_matmul(kalman_gain, self._update_mat, covariance)
        
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
