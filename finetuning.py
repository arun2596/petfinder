import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

import math


# -------------------------- CREATE MODEL ----------------------------------------

class PawpularityModel(nn.Module):
    def __init__(
            self,
            model_name,
            config,
            multisample_dropout=False
    ):
        super(PawpularityModel, self).__init__()
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
        self.regressor = nn.Linear(self.hidden_size, 1)

        self._init_weights(self.regressor)

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

# SEND EFFICIENT NET MODEL NAME
# UPDATE WEIGHT DECAY
# MODEL FOLDER NAME MATCHING
# WHAT NAME DOES IT SAVE MODEL WITH ?
# CREATE FOLDERS :( .. UGHH


# def main():
#     train = pd.read_csv('data/raw/train.csv')
#     test = pd.read_csv('data/raw/test.csv')
#
#     set_seed(0)
#
#     validation_proportion = 0.15
#
#     target_frequency = train[['Id', 'Pawpularity']].groupby('Pawpularity').count().reset_index()
#     target_frequency['Proba'] = sum(target_frequency['Id']) / target_frequency['Id']
#     target_frequency['Proba'] = target_frequency['Proba'] / sum(target_frequency['Proba'])
#
#     row_proba = \
#         train[['Pawpularity']].merge(target_frequency, how='left', left_on='Pawpularity', right_on='Pawpularity')[
#             'Proba']
#     row_proba = row_proba / sum(row_proba)
#
#     indices = np.random.choice(train.index, size=int(train.shape[0] * validation_proportion), replace=False,
#                                p=row_proba)
#
#     train['kfold'] = 1
#     train.loc[indices, 'kfold'] = 0
#
#     model_output_location = 'model_output/finetuning/'
#
#     result_list = []
#     for fold in range(1):
#         print('----')
#         print(f'FOLD: {fold}')
#         result_dict = run(train, fold, model_output_location, freeze_backbone=True)
#         result_list.append(result_dict)
#         print('----')
#
#     [print("FOLD::", i, "Loss:: ", fold['best_val_loss']) for i, fold in enumerate(result_list)]
#
#     oof = np.zeros(len(train))
#     for fold in tqdm.tqdm(range(1), total=1):
#         model = PawpularityModel('efficientnet-b3-pawpularity')
#         model.load_state_dict(
#             torch.load(model_output_location + f'model{fold}.bin')
#         )
#         model.cuda()
#         model.eval()
#         val_index = train[train.kfold == fold].index.tolist()
#         train_loader, val_loader = make_loader(train, batch_size=16, fold=fold)
#
#         preds = []
#         for index, data in enumerate(val_loader):
#             image, target = data['image'], data['target']
#
#             image, target = image.cuda(), target.cuda()
#
#             outputs = model(
#                 image=image,
#                 target=target
#             )
#
#             loss, logits = outputs
#             preds += logits.cpu().detach().numpy().tolist()
#         oof[val_index] = preds
#
#     print("cv", round(np.sqrt(mean_squared_error(train.Pawpularity.values[val_index], oof[val_index])), 4))
#
#
# main()

# backbone = EfficientNet.from_pretrained('efficientnet-b3')
# backbone._fc = nn.Identity()
# add_layer_norm = nn.LayerNorm(1536)(backbone(x))
# add_dropout = nn.Dropout(0.3)(add_layer_norm)
# regressor = nn.Linear(1536, 1)(add_dropout)


# x = torch.randn(10, 3, 300, 300)


# print(backbone._fc)

# a = add_dropout.detach().numpy()
# print(regressor.detach().shape)
# print()

# print(.shape)

#
# print([x for x in backbone.named_modules()])

# print(dir(backbone))


## ADD CODE TO TRAIN FRON LAYERS FIRST AND THEN BACK LAYERS
## SEED
