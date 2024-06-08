"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x



class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp
        self.device = hp.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        # downsample 240x240 -> 30x30
        # Use a pre-trained ResNet model as the encoder
        alexnet = models.alexnet(pretrained=True)
        self.encoder = alexnet.features

        """self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=hp["input_size"], out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout2d(p=0.2)
        )"""

        # upsample 30x30 -> 240x240
        """self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )"""
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.Upsample(scale_factor=5, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, num_classes, kernel_size=7, stride=1, padding=3),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2),
        )

        self.set_optimizer()

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        x = self.encoder(x)
        x = self.decoder(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    # @property
    # def is_cuda(self):
    #     """
    #     Check if model parameters are allocated on the GPU.
    #     """
    #     return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

    # TODO: Mine
    def set_optimizer(self):
        self.optimizer = None
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.hp["learning_rate"],
                                          weight_decay=self.hp["weight_decay"])

    def training_step(self, batch, loss_func):
        loss = None
        self.train()
        self.optimizer.zero_grad()
        images = batch.to(self.device)
        labels = batch.to(self.device)
        images = images.view(images.shape[0], -1)
        labels = labels.view(labels.shape[0], -1)
        pred = self(images)
        loss = loss_func(pred, labels)  # Compute the loss over the predictions and the ground truth.
        loss.backward()  # Stage 2: Backward().
        self.optimizer.step()  # Stage 3: Update the parameters.
        return loss

    def validation_step(self, batch, loss_func):
        loss = None
        self.eval()
        with torch.no_grad():
            images = batch.to(self.device)
            labels = batch.to(self.device)
            images = images.view(images.shape[0], -1)
            labels = labels.view(labels.shape[0], -1)
            pred = self(images)     # self.forward
            loss = loss_func(pred, labels)
        return loss

    def getAcc(self, loader=None):
        assert loader is not None, "Please provide a dataloader for accuracy evaluation"

        self.eval()
        self = self.to(self.device)

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            #flattened_X = X.view(X.shape[0], -1)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc


class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

if __name__ == "__main__":
    from torchinfo import summary
    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")

