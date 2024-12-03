import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from embed import DataEmbedding
from torch.utils.data import Dataset, DataLoader

# V1 . Timesnet 
# (+) cryptocurrncies have discrete cycles that can be daily, weekly, monthly, etc..
# by converting time series to 2d tensors we can catch interperiod and intraperiod pattersn adn relations 
# utilize 2d convolutional networks to process temporal variations 

# 1.Adaptive Period Transform (APT) Module
# 	•	Learns the optimal periods in the data.
# 	•	Transforms the 1D time series into a 2D tensor based on the learned periods.
# 2.2D Convolutional Layers (Inception Block)
# 	•	Processes the 2D tensor to capture intraperiod and interperiod variations.
# 	•	Utilizes multiple kernel sizes to handle different temporal scales.
# 3.Temporal Aggregation
# 	•	Aggregates the features extracted from the 2D convolutional layers.
# 	•	Produces the final output for forecasting or other time series tasks.
# 4.Stacked TimesBlocks
# 	•	Multiple TimesBlocks are stacked to deepen the network and enhance feature representation.

# different timeframes connected with self attention mechansm 
# 1m - 5m - 15 m 
# 1 h - 4 h - d
# 

#creates the block performing the convolutional operations on the 2d matrix on the 2d matrix 
class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


#data (imported form data.py) ( B, T , C )
#generates periods from data ( module is form timesnet) : 
def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model_v1(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model_v1, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        print(configs.c_out)
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Extract mean and std for the percentage change column (index 9)
        mean_pct = x_enc[:, :, 8].mean(dim=1, keepdim=True).detach()  # Mean of the last column
        std_pct = x_enc[:, :, 8].std(dim=1, keepdim=True).detach()  # Std of the last column

        # Normalize input
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Embedding and model processing
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, T, C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # Align temporal dimension
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Project back
        dec_out = self.projection(enc_out)

        # De-Normalization using the specific percentage change stats
        dec_out = dec_out * std_pct.unsqueeze(1)
        dec_out = dec_out + mean_pct.unsqueeze(1)

        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        
            



