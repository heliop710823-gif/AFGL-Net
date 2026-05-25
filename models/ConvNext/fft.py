import torch.fft as fft
import torch 
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DCTFrequencyEnhance(object):
    def __init__(self, low_freq_ratio=0.3, high_freq_boost=1.5):
        """
        参数:
        ▪ low_freq_ratio: 低频部分比例 (0~1)
        ▪ high_freq_boost: 高频增强系数
        """
        self.low_freq_ratio = low_freq_ratio
        self.beta = high_freq_boost

    def __call__(self, x):
        # 计算动态阈值（基于图像尺寸）
        h, w = x.shape[-2:]
        thresh_h = int(h * self.low_freq_ratio)
        thresh_w = int(w * self.low_freq_ratio)       
        # DCT变换
        x_dct = dct_2d(x)    
        # 生成自适应掩码
        low_freq_mask = torch.zeros_like(x_dct)
        low_freq_mask[..., :thresh_h, :thresh_w] = 1     
        # 分离与增强频段
        low_freq = x_dct * low_freq_mask
        high_freq = (x_dct - low_freq) * self.beta  # 直接增强高频        
        # 融合并逆变换
        fused_dct = low_freq + high_freq
        return idct_2d(fused_dct)
def dct1(x):
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])
    return torch.fft.fft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1).real.view(*x_shape)


def idct1(X):
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.fft.fft(v)
    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    V = Vc.real * W_r - Vc.imag * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
    V = 2 * V.view(*x_shape)
    return V


def idct(X, norm=None):
    x_shape = X.shape
    N = x_shape[-1]
    X_v = X.contiguous().view(-1, x_shape[-1]) / 2
    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2
    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)
    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r
    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
    v = torch.fft.ifft(tmp)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]
    return x.view(*x_shape).real


def dct_2d(x, norm=None):
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)

def idct_3d(X, norm=None):
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)
