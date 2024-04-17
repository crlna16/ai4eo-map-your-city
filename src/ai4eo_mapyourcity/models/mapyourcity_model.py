from typing import Any, List

import torch
from torch.nn import functional as F
from torch import optim
from torch.nn import Linear

import timm
import torchmetrics

from pytorch_lightning import LightningModule

class MapYourCityModel(LightningModule):
    '''
    Wrapper for the MapYourCity models
    Architectures from TIMM
    https://huggingface.co/timm/swin_tiny_patch4_window7_224.ms_in1k

    Args:
      model (str)       : specifies which model to use
      pretrained (bool) : if True, use pretrained model weights
      num_classes (int) : number of output classes
      learning_rate (float) : optimizer learning rate
      weight_decay (float) : optimizer weight decay
      weighted_loss (bool) : if True, apply class_weights in CE loss
      class_weights (dict) : class weights for CE loss

    '''

    def __init__(self,
                 backbone,
                 num_classes,
                 learning_rate,
                 weight_decay,
                 weighted_loss,
                 class_weights,
                 validation_metric):

        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.backbone = backbone

        # metrics
        self.validation_metric = validation_metric

        match validation_metric: 
            case 'accuracy':
                self.acc = torchmetrics.Accuracy('multiclass', num_classes=num_classes)
            case 'confusion_matrix':
                self.confmat = torchmetrics.ConfusionMatrix(task='multiclass', threshold=None, num_classes=num_classes, normalize='true')
            case 'mean_average_precision':
                self.maprec = torchmetrics.AveragePrecision(task='multiclass',
                                                            num_classes=num_classes,
                                                            average='macro',
                                                            thresholds=None)
            case _:
                raise ValueError('Validation metric not implemented:', validation_metric)


        self.weighted_loss = weighted_loss
        self.class_weights = torch.Tensor(list(class_weights.values())).to('cuda')

        # optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

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
        y_hat = torch.argmax(prediction, dim=-1)

        if self.weighted_loss:
            loss = F.cross_entropy(prediction, y, weight=self.class_weights)
        else:
            loss = F.cross_entropy(prediction, y)

        if self.validation_metric == 'accuracy':
            metric = self.acc(y_hat, y)
        elif self.validation_metric == 'confusion_matrix':
            metric = self.confmat(y_hat, y).diag().mean()
        elif self.validation_metric == 'mean_average_precision':
            metric = self.maprec(prediction, y)

        return loss, metric, y_hat, y, pid

    def training_step(self, batch, batch_idx):
        loss, metric, _, _, _ = self.step(batch)

        self.log('train_loss', loss)
        self.log('train_metric', metric)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, metric, _, _, _ = self.step(batch)

        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_metric', metric, on_epoch=True, on_step=False, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        loss, metric, y_hat, y, pid = self.step(batch)

        self.valid_predictions['pid'].extend(list(pid))
        self.valid_predictions['predicted_label'].extend(list(y_hat.squeeze().cpu().numpy()))

        self.log('predict_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('predict_metric', metric, on_epoch=True, on_step=False, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, metric, y_hat, y, pid = self.step(batch)

        self.test_predictions['pid'].extend(list(pid))
        self.test_predictions['predicted_label'].extend(list(y_hat.squeeze().cpu().numpy()))

        self.log('test_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_metric', metric, on_epoch=True, on_step=False, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

