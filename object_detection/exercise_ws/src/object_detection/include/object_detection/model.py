import torch 
import numpy as np

from faster_rcnn.src.architecture import FasterRCNN

config_file = "../config/default.yaml"

class NoGPUAvailable(Exception):
    pass

class Wrapper():
    def __init__(self, model_file):
        checkpoint = torch.load(checkpoint_path)
        cfg = checkpoint['cfg']
        cfg.merge_from_file(config_file)
        cfg.freeze()

        device = torch.device("cuda") if (torch.cuda.is_available() and cfg.USE_CUDA) else torch.device("cpu")
        print("Using the device for training: {} \n".format(device))

        model = FasterRCNN(cfg)
        model = model.to(device)

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        self.model = model.eval()

    def predict(self, batch_or_image):
        boxes = []
        labels = []
        scores = []
        for img in batch_or_image:  # or simply pipe the whole batch to the model instead of using a loop!
            box, label, score = self.model(img) # TODO you probably need to send the image to a tensor, etc.
            boxes.append(box)
            labels.append(label)
            scores.append(score)

        return boxes, labels, scores

