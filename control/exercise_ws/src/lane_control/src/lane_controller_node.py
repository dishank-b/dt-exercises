#!/usr/bin/env python3
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading

from lane_controller.controller import PurePursuitLaneController


class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocitie.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        # Add the node parameters to the parameters dictionary
        self.params = dict()
        self.params['~L_d'] = DTParam(
            '~L_d',
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=0.5
        )

        self.params['~L_max'] = DTParam(
            '~L_max',
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=0.5
        )

        self.params['~v'] = DTParam(
            '~v',
            param_type=ParamType.FLOAT,
            min_value=0.05,
            max_value=1.0
        )
        self.params['~k'] = DTParam(
            '~k',
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=10.0
        )
        self.params['~v_min'] = DTParam(
            '~v_min',
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=0.5
        )
        self.params['~d_offset'] = rospy.get_param('~d_offset', None)
        self.params['~omega_ff'] = rospy.get_param('~omega_ff', None)

        self.pp_controller = PurePursuitLaneController(self.params)

        # Construct publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd",
                                           Twist2DStamped,
                                           queue_size=1,
                                           dt_topic_type=TopicType.CONTROL)

        # Construct subscribers
        self.sub_lane_reading = rospy.Subscriber("~lane_pose",
                                                 LanePose,
                                                 self.cbLanePoses,
                                                 queue_size=1)

        self.last_s = None

        self.log("Initialized!")

    def cbLanePoses(self, input_pose_msg):
        """Callback receiving pose messages

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        self.pose_msg = input_pose_msg

        car_control_msg = Twist2DStamped()
        car_control_msg.header = self.pose_msg.header

        # TODO This needs to get changed
        
        vel, omega = self.getControlAction(self.pose_msg)

        car_control_msg.v = vel
        car_control_msg.omega = omega

        self.publishCmd(car_control_msg)


    def getControlAction(self, pose_msg):
        """Callback that receives a pose message and updates the related control command.

        Using a controller object, computes the control action using the current pose estimate.

        Args:
            pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        current_s = rospy.Time.now().to_sec()
        dt = None
        if self.last_s is not None:
            dt = (current_s - self.last_s)
      
        # Compute errors
        d_err = pose_msg.d - self.params['~d_offset']
        phi = pose_msg.phi

        rospy.loginfo("d: {}, phi:{}".format(d_err, phi))

        v, omega, alpha = self.pp_controller.compute_control_action(d_err, phi)

        rospy.loginfo("Velocity: {}, Omega:{}, alpha: {}".format(v, omega, alpha))

        # For feedforward action (i.e. during intersection navigation)
        omega += self.params['~omega_ff']

        self.last_s = rospy.Time.now().to_sec()

        return v, omega
        
    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        self.pub_car_cmd.publish(car_cmd_msg)


    def cbParametersChanged(self):
        """Updates parameters in the controller object."""

        self.pp_controller.update_parameters(self.params)


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()
