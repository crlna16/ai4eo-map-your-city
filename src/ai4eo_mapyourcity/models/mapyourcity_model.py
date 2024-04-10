from typing import Any, List

import torch
from torch.nn import functional as F
from torch import optim
from torch.nn import Linear

from transformers import ViTForImageClassification, SwinForImageClassification
import torchmetrics

from pytorch_lightning import LightningModule

class MapYourCityModel(LightningModule):
    '''
    Wrapper for the MapYourCity models
    - torchvision pretrained Vision Transformers / SWIN

    Args:
      model (str)       : specifies which flavor of ViT to use
      num_classes (int) : number of output classes
      learning_rate (float) : optimizer learning rate
      weighted_loss (bool) : if True, apply class_weights in CE loss
      class_weights (dict) : class weights for CE loss

    '''

    def __init__(self, model, num_classes, learning_rate, weighted_loss, class_weights):

        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        if model.startswith('google/vit'):
            backbone = ViTForImageClassification.from_pretrained(model, num_labels=num_classes, ignore_mismatched_sizes=True)
        elif model.startswith('microsoft/swin'):
            backbone = SwinForImageClassification.from_pretrained(model, num_labels=num_classes, ignore_mismatched_sizes=True)

        self.backbone = backbone

        # metrics
        self.acc = torchmetrics.Accuracy('multiclass', num_classes=num_classes)
        self.weighted_loss = weighted_loss
        self.class_weights = torch.Tensor(list(class_weights.values())).to('cuda')

        # other
        self.learning_rate = learning_rate

        # store predictions
        self.valid_predictions = {'pid': [], 'predicted_label': []}
        self.test_predictions = {'pid': [], 'predicted_label': []}


    def forward(self, x):
        return self.backbone(x)

    def step(self, batch):
       '''
       Any step processes batch to return loss and predictions
       '''

       x, y, pid = batch
       prediction = self.backbone(x)
       y_hat = torch.argmax(prediction.logits, dim=-1)

       if self.weighted_loss:
           loss = F.cross_entropy(prediction.logits, y, weight=self.class_weights)
       else:
           loss = F.cross_entropy(prediction.logits, y)
       acc = self.acc(y_hat, y)
       
       return loss, acc, y_hat, y, pid

    def training_step(self, batch, batch_idx):
        loss, acc, _, _, _ = self.step(batch)

        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, _, _, _ = self.step(batch)

        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_acc', acc, on_epoch=True, on_step=False, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        loss, acc, y_hat, y, pid = self.step(batch)

        self.valid_predictions['pid'].extend(list(pid))
        self.valid_predictions['predicted_label'].extend(list(y_hat.squeeze().cpu().numpy()))

    def test_step(self, batch, batch_idx):
        loss, acc, y_hat, y, pid = self.step(batch)

        self.test_predictions['pid'].extend(list(pid))
        self.test_predictions['predicted_label'].extend(list(y_hat.squeeze().cpu().numpy()))

        self.log('test_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_acc', acc, on_epoch=True, on_step=False, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer