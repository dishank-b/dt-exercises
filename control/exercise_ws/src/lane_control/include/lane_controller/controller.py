import numpy as np


class PurePursuitLaneController:
    """
    Pure Purusit Controller: It makes bot move towards a point which L(look-ahead) disdance away from the bot on a reference stracjectory.

    Explaining the derivation: Here we have a reference trajectory which is straight line in this case, defined by parameters
                               d and \phi, that is perpendicular distance of from bot and relative orientation respectively. 
                               Now you take a point on the reference trajectory which is L distance away from the bot. And 
                               then using this, you can calculate the control commands. 

    d_err: Perpendicular distance of bot from the refrence trajectory (middle line). 
    phi:   Relative angle between y axis of bot and reference trajectory. 
                                 
    The control commands are in terms of linear and angular velocity (v, omega). The input are errors in the relative
    pose of the Duckiebot in the current lane.

    """

    def __init__(self, parameters):
        self.parameters = parameters

    def compute_control_action(self, d_err, phi):
        L_d = self.parameters["~L_d"].value
        v = self.parameters["~v"].value
        k = self.parameters["~k"].value
        v_min = self.parameters["~v_min"].value

        if d_err > L_d:
            L_d = d_err + 0.05
        
        if abs(phi)>0.5:
            v = v_min

        alpha = np.arcsin(d_err/L_d)+phi
        omega = -np.sin(alpha)/k

        return v, omega, alpha

    def update_parameters(self, parameters):
        """Updates parameters of LaneController object.

            Args:
                parameters (:obj:`dict`): dictionary containing the new parameters for LaneController object.
        """
        self.parameters = parameters


