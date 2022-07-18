
import torch
import torch.nn.functional as F
from torch import nn

class MyBceloss12_n(nn.Module):
    def __init__(self):
        super(MyBceloss12_n, self).__init__()


    def forward(self, pred, gt,dst1,dst2):
        eposion = 1e-10
     
        sigmoid_pred = torch.sigmoid(pred)
        count_pos = torch.sum(gt)*1.0+eposion
        count_neg = torch.sum(1.-gt)*1.0
        beta = count_neg/count_pos

        beta_back = count_pos/(count_pos+count_neg)



        dst_loss = beta*(1+dst2)*gt*F.binary_cross_entropy_with_logits(pred, gt, reduce=False) + \
                   (1+dst1)*(1-gt)*F.binary_cross_entropy_with_logits(pred, gt, reduce=False)
        bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
        bce2_lss = torch.mean(dst_loss)
        loss = beta_back*bce1(pred, gt) + beta_back*bce2_lss

        return loss


class MyWcploss(nn.Module):
    def __init__(self):
        super(MyWcploss, self).__init__()


    def forward(self, pred, gt):
        eposion = 1e-10
        sigmoid_pred = torch.sigmoid(pred)
        count_pos = torch.sum(gt)*1.0+eposion
        count_neg = torch.sum(1.-gt)*1.0
        beta = count_neg/count_pos
        beta_back = count_pos / (count_pos + count_neg)


        bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
        loss = beta_back*bce1(pred, gt)

        return loss
