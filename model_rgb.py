from collections import OrderedDict

import torch
import torch.nn as nn

from Config import cfg

class LayerNorm2d(nn.Module):
    """
    Simple, readable normalization for conv features.
    Equivalent-ish to "LayerNorm over channels" for each spatial location.
    Using GroupNorm(1, C) is a common LayerNorm-like choice in CNNs.
    """
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.gn = nn.GroupNorm(1, num_channels, eps=eps)

    def forward(self, x):
        return self.gn(x)


class ConvLSTMCell(nn.Module):
    """
    A minimal ConvLSTM cell.
    Input:  x_t (B, C_in, H, W)
    Hidden: h, c (B, C_hidden, H, W)
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, h, c):
        # Concatenate along channel dim
        combined = torch.cat([x, h], dim=1)  # (B, C_in + C_hidden, H, W)
        gates = self.conv(combined)          # (B, 4*C_hidden, H, W)

        i, f, o, g = torch.chunk(gates, chunks=4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    """
    ConvLSTM layer that returns sequences (like return_sequences=True).
    Input:  (B, T, C, H, W)
    Output: (B, T, hidden_C, H, W)
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size, padding)

    def forward(self, x):
        B, T, C, H, W = x.shape
        device = x.device

        h = torch.zeros(B, self.cell.hidden_channels, H, W, device=device, dtype=x.dtype)
        c = torch.zeros(B, self.cell.hidden_channels, H, W, device=device, dtype=x.dtype)

        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
            outputs.append(h)

        return torch.stack(outputs, dim=1)  # (B, T, hidden_C, H, W)
    
    

class ConvLSTMAE(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels

        self.enc_conv1 = nn.Conv2d(in_channels, 128, kernel_size=11, stride=4, padding=5)
        self.enc_ln1 = LayerNorm2d(128)

        self.enc_conv2 = nn.Conv2d(128, 64, kernel_size=5, stride=2, padding=2)
        self.enc_ln2 = LayerNorm2d(64)

        self.lstm1 = ConvLSTM(in_channels=64, hidden_channels=64, kernel_size=3, padding=1)
        self.ln3 = LayerNorm2d(64)

        self.lstm2 = ConvLSTM(in_channels=64, hidden_channels=32, kernel_size=3, padding=1)
        self.ln4 = LayerNorm2d(32)

        self.lstm3 = ConvLSTM(in_channels=32, hidden_channels=64, kernel_size=3, padding=1)
        self.ln5 = LayerNorm2d(64)

        self.dec_deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_ln1 = LayerNorm2d(64)

        self.dec_deconv2 = nn.ConvTranspose2d(64, 128, kernel_size=11, stride=4, padding=5, output_padding=3)
        self.dec_ln2 = LayerNorm2d(128)

        self.out_conv = nn.Conv2d(128, self.out_channels, kernel_size=11, padding=5)
        self.out_act = nn.Sigmoid()

    def _time_distributed_conv(self, x, layer):
        B, T, C, H, W = x.shape
        y = layer(x.reshape(B * T, C, H, W))
        return y.reshape(B, T, y.shape[1], y.shape[2], y.shape[3])

    def _time_distributed_norm(self, x, layer):
        B, T, C, H, W = x.shape
        y = layer(x.reshape(B * T, C, H, W))
        return y.reshape(B, T, C, H, W)

    def _time_distributed_deconv(self, x, layer):
        B, T, C, H, W = x.shape
        y = layer(x.reshape(B * T, C, H, W))
        return y.reshape(B, T, y.shape[1], y.shape[2], y.shape[3])

    def forward(self, x):
        x = self._time_distributed_conv(x, self.enc_conv1)
        x = self._time_distributed_norm(x, self.enc_ln1)

        x = self._time_distributed_conv(x, self.enc_conv2)
        x = self._time_distributed_norm(x, self.enc_ln2)

        x = self.lstm1(x)
        x = self._time_distributed_norm(x, self.ln3)

        x = self.lstm2(x)
        x = self._time_distributed_norm(x, self.ln4)

        x = self.lstm3(x)
        x = self._time_distributed_norm(x, self.ln5)

        x = self._time_distributed_deconv(x, self.dec_deconv1)
        x = self._time_distributed_norm(x, self.dec_ln1)

        x = self._time_distributed_deconv(x, self.dec_deconv2)
        x = self._time_distributed_norm(x, self.dec_ln2)

        B, T, C, H, W = x.shape
        y = self.out_conv(x.reshape(B * T, C, H, W))
        y = self.out_act(y)

        # reshape back to (B,T,C,H,W) with actual channel count
        return y.reshape(B, T, y.shape[1], y.shape[2], y.shape[3])
    
    
    
def get_model(reload_model=True):
    """
    Parameters
    ----------
    reload_model : bool
        Load saved model or retrain it
    """
    model = ConvLSTMAE(in_channels=3)        # for RGB

    if reload_model:
        ckpt = torch.load(cfg.model_path, map_location=torch.device('cpu'))
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt

        clean_state = OrderedDict()
        for k, v in state_dict.items():
            nk = k.replace('module.', '', 1) if k.startswith('module.') else k
            clean_state[nk] = v

        incompat = model.load_state_dict(clean_state, strict=False)
        if incompat.missing_keys:
            print("[Warn] Missing keys:", incompat.missing_keys)
        if incompat.unexpected_keys:
            print("[Warn] Unexpected keys:", incompat.unexpected_keys)

        print('+++++++++++++++++++++++++++++++++++++++++++++')
        print("Trained Model loaded from", cfg.model_path)
        print('+++++++++++++++++++++++++++++++++++++++++++++')
        model.eval()
        return model
    else:
        print('+++++++++++++++++++++++++++++++++++++++++++++')
        print("Training model from scratch...")
        print('+++++++++++++++++++++++++++++++++++++++++++++')
        return model

