import torch.nn as nn


# Modify the model: Replace the final layer
class DepthToClassificationModel(nn.Module):
    def __init__(self, base_model):
        super(DepthToClassificationModel, self).__init__()
        self.base_model = base_model

        # Remove the existing head
        delattr(self.base_model, 'head')

        self.head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))  # Change output dimension to 1
        )
        self.sigmoid = nn.Sigmoid()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_model.config.fusion_hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2),  # 2 classes: foreground and background
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.base_model.backbone(x)
        features = self.base_model.neck(features)
        output = self.head(features)
        output = self.sigmoid(output)  # Apply sigmoid activation for binary output
        return output


