'''Model for MapYourCity challenge'''
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

import torchmetrics

from pytorch_lightning import LightningModule

class MapYourCityModel(LightningModule):
    '''
    Wrapper LightningModule for the backbones.

    Attributes:
        backbone (torch.nn.Module): Model, defines architecture and forward step.
        validation_metric (str): Choice of validation metric.
        weighted_loss (bool): If True, use the class weights in the loss function.
        class_weights (Dict): The class weights.
        learning_rate (float): Optimizer learning rate.
        weight_decay (float): Optimizer weight decay.
    '''

    def __init__(self,
                 backbone: torch.nn.Module,
                 num_classes: int,
                 learning_rate: float,
                 weight_decay: float,
                 weighted_loss: bool,
                 class_weights: Dict,
                 loss_id: str,
                 validation_metric: str
                 ):
        '''
        Initialize MapYourCityModel.

        Arguments:
            backbone (torch.nn.Module): Model, defines architecture and forward step.
            validation_metric (str): Choice of validation metric.
            weighted_loss (bool): If True, use the class weights in the loss function.
            class_weights (Dict): The class weights.
            learning_rate (float): Optimizer learning rate.
            weight_decay (float): Optimizer weight decay.
            loss_id (str): Choice of loss function.
            num_classes (int): Number of classes.

        Raises:
            ValueError if validation_metric not in ['accuracy', 'confusion_matrix']

        '''

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
                self.confmat = torchmetrics.ConfusionMatrix(task='multiclass',
                                                            threshold=None,
                                                            num_classes=num_classes,
                                                            normalize='true')
            case _:
                raise ValueError('Validation metric not implemented:', validation_metric)


        self.weighted_loss = weighted_loss
        self.class_weights = torch.Tensor(list(class_weights.values())).to('cuda')

        match loss_id:
            case 'cross_entropy':
                if self.weighted_loss:
                    self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
                else:
                    self.criterion = nn.CrossEntropyLoss()
            case 'ordinal_cross_entropy':
                if self.weighted_loss:
                    self.criterion = OrdinalCrossEntropyLoss(num_classes, weight=self.class_weights)
                else:
                    self.criterion = OrdinalCrossEntropyLoss(num_classes)
            case _:
                raise ValueError('Loss ID not implemented:', loss_id)

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

        loss = self.criterion(prediction, y)

        if self.validation_metric == 'accuracy':
            metric = self.acc(y_hat, y)
        elif self.validation_metric == 'confusion_matrix':
            metric = self.confmat(y_hat, y).diag().mean()

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
        _, _, y_hat, _, pid = self.step(batch)

        self.valid_predictions['pid'].extend(list(pid))
        self.valid_predictions['predicted_label'].extend(list(y_hat.squeeze().cpu().numpy()))

    def test_step(self, batch, batch_idx):
        loss, metric, y_hat, _, pid = self.step(batch)

        self.test_predictions['pid'].extend(list(pid))
        if len(y_hat.shape) == 1:
            self.test_predictions['predicted_label'].extend(list(y_hat.cpu().numpy()))
        else:
            self.test_predictions['predicted_label'].extend(list(y_hat.squeeze().cpu().numpy()))

        self.log('test_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_metric', metric, on_epoch=True, on_step=False, sync_dist=True)

    def configure_optimizers(self):
        '''Configure the optimizer to train all model parameters'''
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

class OrdinalCrossEntropyLoss(nn.Module):
    '''
    Ordinal Cross-Entropy Loss function.

    Respects the natural order of classes.
    '''
    def __init__(self, num_classes, weight=None):
        '''
        Initialize OrdinalCrossEntropyLoss.

        Arguments:
            num_classes (int): Number of classes.
            weight (torch.Tensor): Class weights.
        '''
        super().__init__()

        self.num_classes = num_classes
        self.weight = weight

    def forward(self, logits, targets):
        '''
        Calculate the loss.

        Arguments:
            logits (torch.Tensor): Prediction logits
            targets (torch.Tensor): Targets.

        Returns:
            Loss.
        '''

        #batch_size = targets.size(0)
        #binary_targets = torch.zeros_like(logits)
        #for i in range(batch_size):
        #    k = targets[i]
        #    if k < self.num_classes:
        #        binary_targets[i, k:] = 1

        #loss = F.binary_cross_entropy_with_logits(logits, binary_targets, weight=self.weight, reduction='mean')

        alpha = torch.abs( torch.argmax(logits, dim=1) - targets ) / ( self.num_classes - 1 )
        CE = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')

        return torch.mean((1 + alpha) * CE)
