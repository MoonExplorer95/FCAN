from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

def _mix_rbf_kernel(X, Y, sigma_list):
    assert (X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma ** 2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
                + Kt_YY_sum / (m * (m - 1))
                - 2.0 * K_XY_sum / (m * m))

    return mmd2


def coral(source, target):
    """
    Compute CORAL loss between two feature vectors (https://arxiv.org/abs/1607.01719)
    :param source: source vector [N_S, D]
    :param target: target vector [N_T, D]
    :return: CORAL loss
    """
    d = source.size(1)

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    # loss = loss / (4*d*d)

    return loss

def mmd(source, target, gamma=10**3):
    """
    Compute MMD loss between two feature vectors (https://arxiv.org/abs/1605.06636)
    :param source: source vector [N_S, D]
    :param target: target vector [N_T, D]
    :return: MMD loss
    """
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(source, target, [gamma])

    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=True) / source.size(1)

class FreqMMDAlignmentLoss(nn.Module):

    def __init__(self):
        super(FreqMMDAlignmentLoss, self).__init__()

    def feature_regularization_loss(self, f_src, f_tar, method='coral', n_samples=None):

        # view features to [N, D] shape
        src = f_src.reshape(f_src.size(0), -1)
        tar = f_tar.reshape(f_tar.size(0), -1)

        if n_samples is None:
            fs = src
            ft = tar
        else:
            inds = torch.randperm(src.size(1))[:n_samples]
            fs = src[:, inds.to(src.device)]
            ft = tar[:, inds.to(tar.device)]

        if method == 'coral':
            return coral(fs, ft)
        else:
            return mmd(fs, ft) 

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        return self.feature_regularization_loss(f_s, f_t)


# 'https://github1s.com/dongbo811/AFFormer/blob/HEAD/tools/afformer.py#L244-L263'
class LowPassModule(nn.Module):
    def __init__(self, dim, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        self.relu = nn.ReLU()
        ch =  dim // 4
        self.channel_splits = [ch, ch, ch, ch]

    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return nn.Sequential(prior)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        feats = torch.split(feats, self.channel_splits, dim=1)
        priors = [F.upsample(input=self.stages[i](feats[i]), size=(h, w), mode='bilinear') for i in range(4)]
        bottle = torch.cat(priors, 1)
        
        return self.relu(bottle)

class HighPassModule(nn.Module):
    def __init__(self, dim, window={3: 2, 5: 3, 7: 3}):
        super().__init__()
        self.conv_list = nn.ModuleList()
        self.head_splits = []
        Ch = dim // 8
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * Ch,
                cur_head_split * Ch,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * Ch,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]

    def forward(self, feats):
        # Split according to channels.
        v_img_list = torch.split(feats, self.channel_splits, dim=1)
        HP_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list)
        ]
        HP = torch.cat(HP_list, dim=1)
        
        return HP + feats