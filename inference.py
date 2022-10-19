
import torch
from torchvision.datasets import VOCDetection
from yolo import Yolo
import numpy as np

from utils import (
    non_max_suppression,
    cellboxes_to_boxes,
    draw_boxes
)

dataset = VOCDetection(root="./", image_set="val", download=False)

model = Yolo(num_boxes=2, num_classes=20)
model.load_state_dict(torch.load("yolov1_weights.pth"))
model.to(device=torch.device("cuda"))
model.eval()

for idx in range(50,60):
    
    image = np.array(dataset[idx][0].resize((224 * 2,224 * 2)))
    x = torch.tensor(image.transpose(2,0,1)).float().cuda()
    x = x[None,...]
    with torch.no_grad():
        output = model(x)
    bboxes = cellboxes_to_boxes(out=output, S=7)[0]
    bboxes = non_max_suppression(bboxes=bboxes, iou_threshold=0.5, 
        threshold=0.4, box_format="midpoint")
   
    print(bboxes)
    draw_boxes(image = np.array(dataset[idx][0].resize((224 * 2,224*2))), bboxes=bboxes)