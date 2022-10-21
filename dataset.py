import torch
import numpy as np
from torchvision.datasets import VOCDetection


class VOCDataset(torch.utils.data.Dataset):

    def __init__(self, S=7, B=2, C=20, imageset="train", transform=None, download=False):
        self.dataset = VOCDetection(root="./", image_set=imageset, download=download)
        self.classes = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor"
        ]
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset.__getitem__(idx) # explicit getitem call, could be done with [idx]
        image = np.array(data[0].resize((224 * 2,224 * 2)))
        image = torch.tensor(image.transpose(2,0,1)).float()
        annotation = data[1]["annotation"]["object"]
        image_width = int(data[1]["annotation"]["size"]["width"]) 
        image_height = int(data[1]["annotation"]["size"]["height"])
        # Convert the data to cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B)) # same shape as model output, 7,7,30 -> 20 classes probs, then 2 boxes of (x,y,w,h,class)

        for box in annotation:
            # coords must be in 0-1 range
            x = int(box["bndbox"]["xmin"]) / image_width
            y = int(box["bndbox"]["ymin"]) / image_height
            w = (int(box["bndbox"]["xmax"]) - int(box["bndbox"]["xmin"])) / image_width
            h = (int(box["bndbox"]["ymax"]) - int(box["bndbox"]["ymin"])) / image_height
            label = self.classes.index(box["name"])
            i, j = int(self.S * y), int(self.S * x) # coords in the 7x7 grid
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = (
                w * self.S,
                h * self.S,
            )
            # if theres no object in i,j cell
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object, first we tell if there is an object, then its 4 coordinates
                # i,j (grid cell) will be like [1,x,y,w,h], where w and h can be greater than 1, which indicates the object is bigger than the cell itself
                label_matrix[i, j, 20] = 1
                # coordinates relative to the cell
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                # set the grid coordinates
                label_matrix[i, j, 21:25] = box_coordinates

                # set the ground truth object class to 1, others are 0
                label_matrix[i, j, label] = 1
        return image, label_matrix

#if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # import numpy as np
    # # debug it to understand if you want
    # train_dataset = VOCDataset()
    # print(train_dataset[0][0])
    # label_matrix = train_dataset[0][1]
    # image = train_dataset[0][0]
    # for n in range(7):
    #     for m in range(7):
    #         print(f"cell [{n}][{m}]")
    #         print(label_matrix[n][m])
    #         print()
    
    # plt.imshow(np.asarray(image))
    # plt.show()