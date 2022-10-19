import torch
from torch.utils.data import DataLoader
from yolo import Yolo
from dataset import VOCDataset
from utils import (
    mean_average_precision,
    get_bboxes
 )
from rich.progress import track
from loss_function import YoloLoss
import torch.optim as optim
import copy


device = torch.device("cuda")
model = Yolo(num_boxes=2, num_classes=20)
model.to(device=device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)
loss_fn = YoloLoss()

train_dataset = VOCDataset(imageset="trainval")
train_dataloader =  DataLoader(
        dataset=train_dataset,
        batch_size=16,
        num_workers=4,
        shuffle=True,
    )

# Passing
# sanity_check_dataset = torch.utils.data.Subset(train_dataset, list(range(0,7)))
# sanity_check_dataloader =  DataLoader(
#         dataset=sanity_check_dataset,
#         batch_size=8,
#         num_workers=4,
#         shuffle=False,
#     )

model.train()
best_map = 0.0
epoches = 100
for epoch in range(1,epoches + 1):
    print(f"Epoch: {epoch}/{epoches - 1}")
    if epoch%5==0:
        pred_boxes, target_boxes = get_bboxes(
            train_dataloader, model, iou_threshold=0.5, threshold=0.4, box_format="midpoint"
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")
        if mean_avg_prec > best_map:
            best_map = mean_avg_prec
            best_model_wts = copy.deepcopy(model.state_dict())

    mean_loss = []
    
    for x,y in track(train_dataloader, description="Training..."):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Loss: {sum(mean_loss)/len(mean_loss)}")

model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), "yolov1_weights.pth")
