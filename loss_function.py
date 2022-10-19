import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):

    def __init__(self, S=7, B=2, C=20) -> None:
        # S B and C are parameters, but I'm not sure it will work with different than default values
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.mse = nn.MSELoss(reduction="sum")
        self.lambda_coord = 5
        self.lambda_no_obj = .5

    def forward(self, predictions, target):
        """
        y shape: (batch_size, 7, 7, 30)
        y_hat shape: (batch_size, 7, 7, 30)
        At the last dimension: first the 20 classes (idx 0-19), all are 0 except for the object class (if present in the cell) 
        then 2 boxes of 5 coordinates (object_bool,x,y,w,h), where object bool denotes that there is an object in the following
        coordinates (x,y,w,h). The object class of course is set with an 1 in the first 20 numbers (idx 0-19 mentioned above).
        All others are zero, we only allow one object per grid cell.
        """
        torch.autograd.set_detect_anomaly(True)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Calculate IoU for the two predicted bounding boxes with target bbox
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Take the box with highest IoU out of the two prediction
        # Note that bestbox_idx will be indices of 0, 1 for which bbox was best
        _, bestbox_idx = torch.max(ious, dim=0)

        # torch.Size([batch_size, 7, 7, 1]), that tells us which cells have objects
        exists_box = target[..., 20].unsqueeze(3)
        
        # bestbox is 1obj_ij, exists_box is 1obj_i

        # rewriting 0 on the bbox that wasnt the highest IoU and keeping only a single [7,7,4] prediction 
        box_predictions = (
            (
                bestbox_idx * predictions[..., 26:30]
                + (1 - bestbox_idx) * predictions[..., 21:25]
            )
        )
        # that is: [7,7,1] * ([7,7,4]), we set to zero the predictions where there is no box in the target
        box_predictions = exists_box * box_predictions # now thats a [1, 7, 7, 4] tensor only with highest IoU prediction, last dim is : x,y,w,h
        box_targets = exists_box * target[..., 21:25] # thats the [7,7,4] target for the input image

        # First two lines of the loss function described in the paper (page 4, equation (3))

        # torch.abs avoid negative coord predictions, also (+1e-6) avoid numerical instability for gradients if coords are 0
        # we take abs to avoid negative numbers (so we can take its square root), but the gradient that will be calculated 
        # still needs to know which direciton it should update, thats why torch.sign
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # we now have x and y kept, w and h with the square root taken (both in y and y_hat)
        # now we just need to flatten it all and take its mean squared error
        first_two_Lines_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # third line is related to the score given to the bounding box
        pred_box = (
            bestbox_idx * predictions[..., 25:26] + (1 - bestbox_idx) * predictions[..., 20:21]
        )
        third_line_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )

        # fourth line is the loss for where there is no object, taken from both predicted bboxes
        
        # for the first predicted bounding box
        fourth_line_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        ) + self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # fifth line is related to classification scores among the 20 classes
        fifth_line_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )
        loss = (
            self.lambda_coord * first_two_Lines_loss 
            + third_line_loss 
            + self.lambda_no_obj * fourth_line_loss
            + fifth_line_loss
        )
        return loss
    