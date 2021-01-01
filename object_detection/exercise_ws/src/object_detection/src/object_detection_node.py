#!/usr/bin/env python3
import numpy as np
import rospy
import rospkg
import yaml
from PIL import Image as PIL_Image, ImageDraw

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading, AntiInstagramThresholds, Vector2D

from image_processing.anti_instagram import AntiInstagram
from image_processing.ground_projection_geometry import Point, GroundProjectionGeometry
from image_processing.rectification import Rectify

from image_geometry import PinholeCameraModel

import os
import cv2
from object_detection.model import Wrapper
from cv_bridge import CvBridge

def load_extrinsics():
    """
    Loads the homography matrix from the extrinsic calibration file.

    Returns:
        :obj:`numpy array`: the loaded homography matrix

    """
    # load intrinsic calibration
    cali_file_folder = '/data/config/calibrations/camera_extrinsic/'
    cali_file = cali_file_folder + rospy.get_namespace().strip("/") + ".yaml"

    # Locate calibration yaml file or use the default otherwise
    if not os.path.isfile(cali_file):
        rospy.logwarn("Can't find calibration file: %s.\n Using default calibration instead."
                    % cali_file)
        cali_file = (cali_file_folder + "default.yaml")

    # Shutdown if no calibration file not found
    if not os.path.isfile(cali_file):
        msg = 'Found no calibration file ... aborting'
        rospy.logerr(msg)
        rospy.signal_shutdown(msg)

    try:
        with open(cali_file,'r') as stream:
            calib_data = yaml.load(stream)
    except yaml.YAMLError:
        msg = 'Error in parsing calibration file %s ... aborting' % cali_file
        rospy.logerr(msg)
        rospy.signal_shutdown(msg)

    return calib_data['homography']

class ObjectDetectionNode(DTROS):

    def __init__(self, node_name):
        
        self.initialized = False
        veh = os.getenv("VEHICLE_NAME")

        # Initialize the DTROS parent class
        super(ObjectDetectionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )

        # Construct publishers
        self.pub_obj_dets = rospy.Publisher(
            "~duckie_detected",
            BoolStamped,
            queue_size=1,
            dt_topic_type=TopicType.PERCEPTION
        )

        self.image_publish = rospy.Publisher(
            "~duckie_image",
            Image,
            queue_size=1,
            dt_topic_type=TopicType.PERCEPTION
        )

        # Construct subscribers
        self.sub_image = rospy.Subscriber(
            "~image/compressed",
            CompressedImage,
            self.image_cb,
            buff_size=10000000,
            queue_size=1
        )
        
        self.sub_thresholds = rospy.Subscriber(
            "~thresholds",
            AntiInstagramThresholds,
            self.thresholds_cb,
            queue_size=1
        )

        self.sub_camera_info = rospy.Subscriber(
            f"/{veh}/camera_node/camera_info", 
            CameraInfo, 
            self.cb_camera_info, 
            queue_size=5
        )

        self.ai_thresholds_received = False
        self.anti_instagram_thresholds=dict()
        self.ai = AntiInstagram()
        self.bridge = CvBridge()

        self.camera_info_received = False
        self.homography = load_extrinsics()
        self.img_size = None

        model_file = rospy.get_param('~model_file','.')
        config_file = rospy.get_param('~config_file', '.')
        self.safe_distance = rospy.get_param('~safe_distance')

        rospack = rospkg.RosPack()
        model_file_absolute = rospack.get_path('object_detection') + model_file
        config_file_absolute = rospack.get_path('object_detection') + config_file

        self.model_wrapper = Wrapper(model_file_absolute, config_file_absolute)
        self.initialized = True
        self.log("Initialized!")
    
    def thresholds_cb(self, thresh_msg):
        self.anti_instagram_thresholds["lower"] = thresh_msg.low
        self.anti_instagram_thresholds["higher"] = thresh_msg.high
        self.ai_thresholds_received = True

    def cb_camera_info(self, msg):
        """
        Initializes a :py:class:`image_processing.GroundProjectionGeometry` object and a
        :py:class:`image_processing.Rectify` object for image rectification

        Args:
            msg (:obj:`sensor_msgs.msg.CameraInfo`): Intrinsic properties of the camera.

        """

        if not self.camera_info_received:
            self.rectifier = Rectify(msg)
            self.ground_projector = GroundProjectionGeometry(im_width=msg.width,
                                                            im_height=msg.height,
                                                            homography=np.array(self.homography).reshape((3, 3)))
        self.camera_info_received=True

    def image_cb(self, image_msg):
        if not self.initialized or not self.camera_info_received:
            return

        # self.image_publish.publish(image_msg)
        # Decode from compressed image with OpenCV
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
        except ValueError as e:
            self.logerr('Could not decode image: %s' % e)
            return
        
        # Perform color correction
        # if self.ai_thresholds_received:
        #     image = self.ai.apply_color_balance(
        #         self.anti_instagram_thresholds["lower"],
        #         self.anti_instagram_thresholds["higher"],
        #         image
        #     )
        
        # image = cv2.resize(image, (224,224))
        self.img_size = image.shape[:2]

        bboxes, classes, scores = self.model_wrapper.predict([image])

        img_boxes = self._draw_boxes(image, bboxes[0], classes[0])

        self.image_publish.publish(self.bridge.cv2_to_imgmsg(img_boxes, encoding="rgb8"))
        
        msg = BoolStamped()
        msg.header = image_msg.header
        msg.data = self.det2bool(bboxes[0], classes[0]) # [0] because our batch size given to the wrapper is 1

        bboxes = self._resize_boxes(bboxes) # return boxes are for image 224x224, but in the processing 480x640 size boxes are used
        
        self.pub_obj_dets.publish(msg)

        return bboxes, classes, scores
    
    def det2bool(self, bboxes, classes):
        obj_det_list = []
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i]
            label = classes[i]
            if label==0:
                low_center = Vector2D((x1 + x2)/2.0/self.img_size[1], y2/self.img_size[0])
                norm_pt = Point.from_message(low_center)
                pixel = self.ground_projector.vector2pixel(norm_pt)
                rect = self.rectifier.rectify_point(pixel)
                rect_pt = Point.from_message(rect)
                ground_pt = self.ground_projector.pixel2ground(rect_pt)
                
                dist = np.sqrt(ground_pt.x**2 + ground_pt.y**2)
                print(ground_pt.x, ground_pt.y)

                if dist < self.safe_distance:
                    print(ground_pt.x, ground_pt.y)
                    rospy.logwarn("Pedestrian ahead, in unsafe distance, stop!")
                    return True

        return False

    def _resize_boxes(self, bboxes):
        resize_boxes = []
        for boxes in bboxes:
            boxes = boxes*np.array([224.0/self.img_size[1], 224/self.img_size[0], 224/self.img_size[1], 224/self.img_size[0]])
            resize_boxes.append(boxes)
        
        return resize_boxes

    def _draw_boxes(self, image, bboxes, classes):
        img =  PIL_Image.fromarray(image.copy())
        draw_obj = ImageDraw.Draw(img)
        colors = {0: 'red', 1:'blue', 2:'green', 3:'black'}
        for box, clss in zip(bboxes, classes):
            draw_obj.rectangle(box, outline=colors[clss])
        
        return np.array(img)


if __name__ == "__main__":
    # Initialize the node
    object_detection_node = ObjectDetectionNode(node_name='object_detection_node')
    # Keep it spinning
    rospy.spin()
