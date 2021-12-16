import torch
import torch.nn as nn
import timm

import math


# -------------------------- CREATE MODEL ----------------------------------------

class SwinModel(nn.Module):
    def __init__(
            self,
            model_name,
            config,
            multisample_dropout=False
    ):
        super(SwinModel, self).__init__()
        self.model_name = model_name
        self.config = config

        self.backbone = timm.create_model(config['efficient_net_version'], pretrained=True)
        self.hidden_size = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        if multisample_dropout:
            self.dropouts = nn.ModuleList([
                nn.Dropout(0.5) for _ in range(5)
            ])
        else:
            self.dropouts = nn.ModuleList([nn.Dropout(0.3)])
        self.regressor = nn.Linear(self.hidden_size, 1)

        self._init_weights(self.regressor)

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
                logits = self.regressor(dropout(backbone_output))
            else:
                logits += self.regressor(dropout(backbone_output))

        logits /= len(self.dropouts)

        logits = torch.clip(logits, min=0, max=100)

        #
        #         # calculate loss
        loss = None
        if target is not None:
            # regression task
            loss_fn = torch.nn.MSELoss()
            logits = logits.view(-1).to(target.dtype)
            # print(logits)
            loss = torch.sqrt(loss_fn(logits, target.view(-1)) + 1e-6)

        return (loss, logits) if loss is not None else logits
