import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

import math


# -------------------------- CREATE MODEL ----------------------------------------

class Pawpularity2018Model(nn.Module):
    def __init__(
            self,
            model_name,
            config,
            multisample_dropout=False
    ):
        super(Pawpularity2018Model, self).__init__()
        self.model_name = model_name
        self.config = config
        self.backbone = EfficientNet.from_pretrained(config['efficient_net_version'])
        self.backbone._fc = nn.Identity()

        self.hidden_size = self.get_hidden_size()

        if multisample_dropout:
            self.dropouts = nn.ModuleList([
                nn.Dropout(0.5) for _ in range(5)
            ])
        else:
            self.dropouts = nn.ModuleList([nn.Dropout(0.3)])
        self.logits = nn.Linear(self.hidden_size, 5)

        self._init_weights(self.logits)

    def get_hidden_size(self):
        x = torch.randn(1, 3, 300, 300)
        return self.backbone(x).shape[-1]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init_range = 1.0 / math.sqrt(module.weight.shape[1])
            module.weight.data.uniform_(-init_range, init_range)

            # if module.bias is not None:
            #     module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def forward(
            self,
            image=None,
            target=None
    ):

        backbone_output = self.backbone(image)

        # layer_norm_out = self.layer_norm(backbone_output)

        # multi-sample dropout
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.logits(dropout(backbone_output))
            else:
                logits += self.logits(dropout(backbone_output))

        logits /= len(self.dropouts)

        logits = torch.nn.Softmax(dim=-1)(logits)

        #         # calculate loss
        loss = None
        if target is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            #logits = logits.view(-1)
            loss = loss_fn(logits, target.long().view(-1))

        return (loss, logits) if loss is not None else logits


# config = {'efficient_net_version': 'efficientnet-b3'}
# mod = Pawpularity2018Model(model_name='', config=config)
# import numpy as np
# x = mod(torch.randn(3, 3, 500, 500), torch.randint(0, 5, (3, 1)))
# print(np.sum(x[1].detach().numpy(),axis=1))