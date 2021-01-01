'''
Contains default params for object detector
'''
from .yacs.config import CfgNode as CN


conf_params = CN()

conf_params.SEED = 5
conf_params.USE_CUDA = True

conf_params.DATASET = CN()
conf_params.DATASET.NAME = "detection" #option: "detections", "tracking"
conf_params.DATASET.PATH = ""
conf_params.DATASET.LENGTH = 7000
 
conf_params.LOGS = CN()
conf_params.LOGS.BASE_PATH = "./logs"

conf_params.TRAIN = CN()
conf_params.TRAIN.BATCH_SIZE = 10
conf_params.TRAIN.EPOCHS = 50
conf_params.TRAIN.LR = 1e-3
conf_params.TRAIN.LR_DECAY = 0.2
conf_params.TRAIN.MOMENTUM = 0.9
conf_params.TRAIN.MILESTONES = [10,20]
conf_params.TRAIN.DSET_SHUFFLE = False
conf_params.TRAIN.SAVE_MODEL_EPOCHS = 5
conf_params.TRAIN.DATASET_DIVIDE = 0.9

conf_params.INPUT = CN()
# conf_params.INPUT.IMAGE_SIZE = (375,1242) #h,w
conf_params.INPUT.LABELS_TO_TRAIN = []
conf_params.INPUT.NUM_CLASSES = 0


"""
Model
"""
conf_params.ARCHITECTURE = CN()
conf_params.ARCHITECTURE.MODEL = "FasterRCNN"

"""
For Backbone
"""
conf_params.BACKBONE = CN()

# Params to define input image transformation.
# The input to backbone network has to be RGB 
# image with intensity scaled between 0 to 1.
# The following normalization is applied on the 
# top of that image. 

# The mean/std are standard used for training the 
# backbone networks. 

# Find more details here.
# https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
# https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
# https://pytorch.org/docs/stable/torchvision/models.html

# The mean and std are of ImageNet dataset, as the models are trained on that. 
# We will stick to same to get better output.
conf_params.BACKBONE.MEAN = [0.485, 0.456, 0.406]
conf_params.BACKBONE.STD = [0.229, 0.224, 0.225]

# choices = ['VGG16', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
conf_params.BACKBONE.MODEL_NAME = 'resnet50'
# choices = [1,2,3,4]
conf_params.BACKBONE.RESNET_STOP_LAYER = 3
conf_params.BACKBONE.FREEZE_AT = 2



"""
For Anchor Generator
"""
conf_params.ANCHORS = CN()
conf_params.ANCHORS.ASPECT_RATIOS = [0.5,1,2]
conf_params.ANCHORS.ANCHOR_SCALES = [64, 128, 256]
conf_params.ANCHORS.POS_PROPOSAL_THRES = 0.7
conf_params.ANCHORS.NEG_PROPOSAL_THRES = 0.3


"""
For Region Proposal Network
"""
conf_params.RPN = CN()
conf_params.RPN.CONV_MEAN = 0.01
conf_params.RPN.CONV_VAR = 0.02
conf_params.RPN.BIAS = 0.01
conf_params.RPN.UNCERTAIN_MEAN = 0.01
conf_params.RPN.UNCERTAIN_VAR = 0.02
conf_params.RPN.UNCERTAIN_BIAS = 0.01
conf_params.RPN.ACTIVATION_ALPHA = 1
conf_params.RPN.LOSS_WEIGHT = 1.0
conf_params.RPN.BATCH_SIZE_PER_IMAGE = 256
conf_params.RPN.NMS_THRESH = 0.7
conf_params.RPN.POSITIVE_FRACTION = 0.5
conf_params.RPN.MIN_SIZE_PROPOSAL = 5
conf_params.RPN.PRE_NMS_TOPK_TRAIN = 4000
conf_params.RPN.PRE_NMS_TOPK_TEST = 4000
conf_params.RPN.POST_NMS_TOPK_TRAIN = 2000
conf_params.RPN.POST_NMS_TOPK_TEST = 50 
conf_params.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
conf_params.RPN.IOU_THRESHOLDS = [0.3, 0.7]
conf_params.RPN.IOU_LABELS = [0, -1, 1]
conf_params.RPN.BOUNDARY_THRESH = -1
conf_params.RPN.SMOOTH_L1_BETA = 0.0


"""
For ROI and Detection
"""
conf_params.ROI_HEADS = CN()
conf_params.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
conf_params.ROI_HEADS.POSITIVE_FRACTION = 0.25
conf_params.ROI_HEADS.SCORE_THRESH_TEST = 0.5
conf_params.ROI_HEADS.NMS_THRESH_TEST = 0.5	
conf_params.ROI_HEADS.PROPOSAL_APPEND_GT = True
conf_params.ROI_HEADS.IOU_THRESHOLDS = [0.5]
conf_params.ROI_HEADS.IOU_LABELS = [0, 1]
# conf_params.ROI_HEADS.POOLER_TYPE = "ROIPool"
conf_params.ROI_HEADS.POOLER_TYPE = "ROIAlign"
# conf_params.ROI_HEADS.POOLER_RESOLUTION = 14 # After this there is MaxPool2D, so final resolution is 7x7
conf_params.ROI_HEADS.POOLER_RESOLUTION = 7 # No maxpool after this
conf_params.ROI_HEADS.POOLER_SAMPLING_RATIO = 0
conf_params.ROI_HEADS.FC_DIM = 1024
conf_params.ROI_HEADS.CLS_AGNOSTIC_BBOX_REG = True
conf_params.ROI_HEADS.LOSS_TYPE = "deterministic" # Options: "deterministic, loss_attenuation, loss_attenuation_with_calibration"
conf_params.ROI_HEADS.SMOOTH_L1_BETA = 0.0
conf_params.ROI_HEADS.BBOX_REG_WEIGHTS = (10.0, 10.0, 10.0, 10.0)
# conf_params.ROI_HEADS.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
conf_params.ROI_HEADS.DETECTIONS_PER_IMAGE = 100
conf_params.ROI_HEADS.MERGING = "nms" #option: "nms", "bayesian_clustering"


"""
Tracker
"""
conf_params.TRACKING = CN()
conf_params.TRACKING.IOU_DISTANCE = 0.85
conf_params.TRACKING.MAX_AGE = 1
conf_params.TRACKING.N_INIT = 2
conf_params.TRACKING.POS_NOISE_WEIGHT = 1. / 40
conf_params.TRACKING.VEL_NOISE_WEIGHT = 1. / 160

"""
Solver
"""

conf_params.SOLVER = CN() #options: "adam", "sgd"
conf_params.SOLVER.OPTIM = "sgd" #options: "adam", "sgd"



