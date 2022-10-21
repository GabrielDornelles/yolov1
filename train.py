import torch
from torch.utils.data import DataLoader
from yolo import Yolo
from dataset import VOCDataset
from utils import (
    log_some_examples,
    mean_average_precision,
    get_bboxes
 )
from rich.progress import track
from loss_function import YoloLoss
import torch.optim as optim
import copy
import hydra
from omegaconf import OmegaConf
import os
import wandb


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg):
    print("Configurations:")
    print(OmegaConf.to_yaml(cfg))

    lr = (cfg.training.lr)
    device = torch.device(cfg.processing.device)
    num_epochs = cfg.training.num_epochs
    save_model_every_n_epochs = cfg.processing.save_model_every_n_epochs

    model = Yolo(num_boxes=2, num_classes=20)
    model.to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = YoloLoss()

    train_dataset = VOCDataset(imageset="trainval")
    train_dataloader =  DataLoader(
            dataset=train_dataset,
            batch_size=cfg.processing.batch_size,
            num_workers=cfg.processing.num_workers,
            shuffle=True,
        )
    
    # sanity_check_dataset = torch.utils.data.Subset(train_dataset, list(range(0,7)))
    # sanity_check_dataloader =  DataLoader(
    #         dataset=sanity_check_dataset,
    #         batch_size=8,
    #         num_workers=4,
    #         shuffle=False,
    #     )

    model.train()
    best_map = 0.0

    if not os.path.exists("checkpoints"): os.makedirs("checkpoints")

    experiment = wandb.init(project='yolov1', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=num_epochs, batch_size=cfg.processing.batch_size, learning_rate=lr, amp=False))
    global_step = 0

    for epoch in range(1,num_epochs + 1):
        print(f"Epoch: {epoch}/{num_epochs - 1}")
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

            experiment.log({
                   f"mAP@0.5": mean_avg_prec
            })
        
        mean_loss = []
        
        for x,y in track(train_dataloader, description="Training..."):
            x, y = x.to(device), y.to(device)
            #with torch.cuda.amp.autocast():
            out = model(x)
            loss = loss_fn(out, y)
            mean_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            experiment.log({
                'Loss': loss.item(),
                'global_step': global_step,
                'epoch': epoch
                })
            
        experiment.log({ 
            "Inference + NMS": wandb.Image(log_some_examples(train_dataset, model)),
            "Epoch Loss": sum(mean_loss)/len(mean_loss),
        })
       
        if epoch % save_model_every_n_epochs == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov1_weights_epoch_{epoch}.pth")

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "checkpoints/yolov1_weights_best.pth")

if __name__ == "__main__":
    main()