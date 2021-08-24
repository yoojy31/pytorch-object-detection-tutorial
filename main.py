# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import sys
import math

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor

from engine import evaluate
import utils
import transforms as T


class PennFudanDataset(torch.utils.data.Dataset):
    '''
    PennFudanPed/
      PedMasks/
        FudanPed00001_mask.png
        FudanPed00002_mask.png
        FudanPed00003_mask.png
        FudanPed00004_mask.png
        ...
      PNGImages/
        FudanPed00001.png
        FudanPed00002.png
        FudanPed00003.png
        FudanPed00004.png
    '''
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def create_faster_rcnn():
    resnet = torchvision.models.resnet50(pretrained=True)
    backbone = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3,
    )
    # print(backbone(torch.rand(4, 3, 800, 800)).shape)
    backbone.out_channels = 1024

    rpn_anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

    num_anchors = 5 * 3
    in_channels = backbone.out_channels
    rpn_head = RPNHead(in_channels, num_anchors)
    # prob, box_delta = rpn_head([torch.rand(4, 1024, 50, 50)])
    # print(box_delta[0].shape, prob[0].shape)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], output_size=7, sampling_ratio=2)
    # print(roi_pooler(
    #     x={'0': torch.rand(4, 1024, 50, 50)},
    #     boxes=[torch.randint(0, 800, size=(100, 4)).float()] * 4,
    #     image_shapes=[(800, 800)] * 4).shape)

    in_channles = 7 * 7 * backbone.out_channels
    representation_size = 2048
    box_base = TwoMLPHead(in_channels=in_channles,
                          representation_size=representation_size)
    # print(box_base(torch.rand(400, 1024, 7, 7)).shape)

    num_classes = 2
    box_predictor = FastRCNNPredictor(in_channels=representation_size, num_classes=num_classes)
    # prob, box_delta = box_predictor(torch.rand(400, 2048))
    # print(prob.shape, box_delta.shape)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       # num_classes=2,
                       rpn_head=rpn_head,
                       rpn_anchor_generator=rpn_anchor_generator,
                       box_roi_pool=roi_pooler,
                       box_head=box_base,
                       box_predictor=box_predictor)
    return model


def create_pretrained_faster_rcnn():
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def create_pretrained_ssd():
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
    # feature_maps = model.backbone(torch.rand(4, 3, 320, 320))
    # print(len(feature_maps), feature_maps.keys())
    # print(feature_maps['0'].shape, feature_maps['1'].shape, feature_maps['2'].shape,
    #       feature_maps['3'].shape, feature_maps['4'].shape, feature_maps['5'].shape)
    # out = model.head(list(feature_maps.values()))
    # print(out['bbox_regression'].shape, out['cls_logits'].shape)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    # []
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # for images, targets in data_loader:
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # []
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # []
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

    # split the dataset in train and test set
    indices = list(range(len(dataset)))
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # []
    # get the model using our helper function
    # model = create_faster_rcnn()
    # model = create_pretrained_faster_rcnn()
    model = create_pretrained_ssd()
    # move model to the right device

    # []
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")



if __name__ == "__main__":
    main()
