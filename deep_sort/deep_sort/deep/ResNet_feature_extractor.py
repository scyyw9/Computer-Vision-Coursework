import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2


class ResNet(nn.Module):
    def __init__(self, num_classes=751, reid=False):
        super().__init__()
        self.reid = reid

        self.resnet = models.resnet34(pretrained=False)

        # Modify the fully connected network in ResNet
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

        # Keep only the convolutional part, remove the fully connected layers
        self.resnet_conv = nn.Sequential(*list(self.resnet.children())[:-1])
        # Fully connected network
        self.resnet_fc = self.resnet.fc

    def forward(self, x):
        features = self.resnet_conv(x)  # 8, 512, 1, 1
        features = features.view(features.size(0), -1)

        if self.reid:
            # Perform L2 normalization on the feature vectors
            features = features.div(features.norm(p=2, dim=1, keepdim=True))
            return features
        output = self.resnet_fc(features)
        return output


class FeatureExtractor(object):
    """
        Feature Extractor:
        Extract the features corresponding to the bounding box, and obtain a fixed-dimensional embedding as
        the representative of the bounding box, for use in similarity calculation

        The model training is carried out according to the traditional ReID method
        When using the Extractor class, the input is a list of images,
        and the outputs is the corresponding features of the images
    """

    def __init__(self, model_path, device):
        self.model = ResNet(reid=True)
        assert os.path.isfile(model_path), "Error: no checkpoint file found!"
        print(model_path)
        checkpoint = torch.load(model_path)
        net_dict = checkpoint['net_dict']
        self.model.load_state_dict(net_dict)
        self.device = device
        self.model.to(self.device)

        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.model(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    # img = torch.rand(8, 3, 128, 64)
    # extr = FeatureExtractor()
    # feature = extr(img)
    # print(feature.shape)
    pass
