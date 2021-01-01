import torch 
import os
import numpy as np
import torchvision.transforms as T

from object_detection.faster_rcnn.src.architecture import FasterRCNN
from object_detection.faster_rcnn.src.config import Cfg as cfg

class NoGPUAvailable(Exception):
    def __init__(self):
        print("GPU not available!")


class Wrapper():
    def __init__(self, checkpoint_path, config_file):
        print(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        cfg.merge_from_file(config_file)
        cfg.freeze()
        self.cfg = cfg

        self.device = torch.device("cuda") if (torch.cuda.is_available() and cfg.USE_CUDA) else torch.device("cpu")
        print("Using the device for training: {} \n".format(self.device))

        if not torch.cuda.is_available():
            raise NoGPUAvailable()

        model = FasterRCNN(cfg)
        model = model.to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'], strict=True)

        self.model = model.eval()

    def predict(self, batch_or_image: list):
        boxes = [] 
        labels = []
        scores = []

        with torch.no_grad():
            for img in batch_or_image:  # or simply pipe the whole batch to the model instead of using a loop!
                img = self._image_transform(img)
                _, instances, _ , _ = self.model(img)
                
                boxes.append(instances[0].pred_boxes)
                labels.append(instances[0].pred_classes)
                scores.append(instances[0].scores)
        
        boxes = [x.numpy() for x in boxes]
        labels = [x.cpu().numpy() for x in labels]
        scores = [x.cpu().numpy() for x in scores] 

        return boxes, labels, scores #boxes: MxNx4- list of M images where each image can have N_i objects and 4 coordinates for each object.

    def _image_transform(self, img):
        if list(img.shape[:2]) != [480, 640]:
            transform = T.Compose([T.ToTensor(),
                            T.Normalize(mean=self.cfg.BACKBONE.MEAN, std=self.cfg.BACKBONE.STD), T.resize((480, 640))])

        else:
            transform = T.Compose([T.ToTensor(),
                            T.Normalize(mean=self.cfg.BACKBONE.MEAN, std=self.cfg.BACKBONE.STD)])

        img = transform(img)
        img = img.unsqueeze(0).to(self.device)

        return img

def draw_boxes(image, bboxes, classes=None):
    img =  Image.fromarray(image.copy())
    draw_obj = ImageDraw.Draw(img)
    for box in bboxes:
        draw_obj.rectangle(box)
    
    img.show()
    return np.array(img)

if __name__ == "__main__" :
    from faster_rcnn.src.architecture import FasterRCNN
    from faster_rcnn.src.config import Cfg as cfg
    from PIL import Image, ImageDraw
    config = "/home/dishank/dt-exercises/object_detection/exercise_ws/src/object_detection/config/object_detection_node/sim_config.yaml"
    model_fiel = "/home/dishank/dt-exercises/object_detection/exercise_ws/src/object_detection/include/object_detection/weights/sim.pt"
    model = Wrapper(model_fiel, config)

    img_path = "/home/dishank/dt-exercises/object_detection/dataset/sim_data"
    img = np.array(Image.open(os.path.join(img_path,'1.png')), dtype=np.uint8)

    boxes, labels, scores = model.predict([img])

    draw_boxes(img, boxes[0])


    

