# coding = utf-8
# author = xy


import torch
import utils
from torch.nn.modules import loss
import torch.nn.functional as f


class LossJoin(loss._Loss):
    """ MLE 最大似然估计 """
    def __init__(self):
        super(LossJoin, self).__init__()

    def forward(self, outputs, y):
        """
        :param outputs: tensor (batch_size, 3)
        :param y: tensor
        :return:loss
        """
        outputs_1 = f.log_softmax(outputs, dim=1)
        loss_joint = f.nll_loss(outputs_1, y)

        # y_mask_2 = y.eq(2)
        # batch_size = outputs.size(0) - torch.sum(y_mask_2).item()
        # y_2 = y.masked_fill(y_mask_2, 0)
        # outputs_2 = f.log_softmax(outputs[:, :2], dim=1)
        # y_mask_2 = y_mask_2.unsqueeze(1).expand(outputs_2.size())
        # outputs_2 = outputs_2.masked_fill(y_mask_2, 0)
        # loss_zhengfu = f.nll_loss(outputs_2, y_2, reduction='sum')
        # loss_zhengfu = loss_zhengfu / batch_size if batch_size != 0 else 0
        #
        # y_mask_3 = y.ne(2)
        # y_3 = y.masked_fill(y_mask_3, 0)
        # y_mask_4 = y.eq(2)
        # y_3 = y_3.masked_fill(y_mask_4, 1)
        # y_3 = y_3.float()
        # outputs_3 = outputs[:, 2]
        # loss_youwu = f.binary_cross_entropy_with_logits(outputs_3, y_3)
        #
        # loss_value = loss_joint + 0.3*loss_zhengfu + loss_youwu

        loss_value = loss_joint

        return loss_value
