import warnings
from typing import Optional, Tuple, List

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class STFT(nn.Module):
    '''Short Time Fourier Transform
    forward(x):
        x: [B, T_wav] or [B, 1, T_wav]
        output: [B, n_fft//2+1, T_spec, 2]   (magnitude = False)
        output: [B, n_fft//2+1, T_spec]      (magnitude = True)
    inverse(x):
        x: [B,  n_fft//2+1, T_spec, 2]
        output: [B, T_wav]
    '''

    __constants__ = ["normalize", "center", "magnitude", "n_fft",
                     "hop_size", "win_size", "padding", "clip", "pad_mode"]
    __annotations__ = {'window': Optional[Tensor]}

    def __init__(
        self, n_fft: int, hop_size: int, win_size: Optional[int] = None,
        center: bool = True, magnitude: bool = False,
        win_type: Optional[str] = "hann",
        window: Optional[Tensor] = None, normalized: bool = False,
        pad_mode: str = "reflect",
        device=None, dtype=None
    ):
        super().__init__()
        self.normalized = normalized
        self.center = center
        self.magnitude = magnitude
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.padding = 0 if center else (n_fft + 1 - hop_size) // 2
        self.clip = (hop_size % 2 == 1)
        self.pad_mode = pad_mode
        if win_size is None:
            win_size = n_fft
        
        if window is not None:
            win_size = window.size(-1)
        elif win_type is None:
            window = torch.ones(win_size, device=device, dtype=dtype)
        elif win_type == "povey":
            window = torch.hann_window(
                win_size,
                periodic=False,
                device=device,
                dtype=dtype
            ).pow(0.85)
        elif win_type == "hann-sqrt":
            window = torch.hann_window(
                win_size,
                periodic=False,
                device=device,
                dtype=dtype
            ).pow(0.5)
        else:
            window: Tensor = getattr(torch, f"{win_type}_window")(win_size,
                device=device, dtype=dtype)
        self.register_buffer("window", window, persistent=False)
        self.window: Tensor
        self.win_size = win_size
        assert n_fft >= win_size, f"n_fft({n_fft}) must be bigger than win_size({win_size})"

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T_wav] or [B, 1, T_wav]
        # output: [B, n_fft//2+1, T_spec(, 2)]
        if x.dim() == 3:  # [B, 1, T] -> [B, T]
            x = x.squeeze(1)
        if self.padding > 0:
            x = F.pad(x.unsqueeze(0), (self.padding, self.padding), mode=self.pad_mode).squeeze(0)

        spec = torch.stft(x, self.n_fft, hop_length=self.hop_size, win_length=self.win_size,
            window=self.window, center=self.center, pad_mode=self.pad_mode,
            normalized=self.normalized, onesided=True, return_complex=True)

        if self.magnitude:
            spec = spec.abs()
        else:
            spec = torch.view_as_real(spec)
        
        if self.clip:
            spec = spec[:, :, :-1]

        return spec

    def inverse(self, spec: Tensor) -> Tensor:
        # x: [B, n_fft//2+1, T_spec, 2]
        # output: [B, T_wav]
        if not self.center:
            raise NotImplementedError("center=False is currently not implemented. "
                "Please set center=True")

        spec = torch.view_as_complex(spec.contiguous())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wav = torch.istft(spec, self.n_fft, hop_length=self.hop_size,
                win_length=self.win_size, center=self.center, normalized=self.normalized,
                window=self.window, onesided=True, return_complex=False)

        return wav

    def inverse_complex(self, spec: Tensor) -> Tensor:
        # x: [B, n_fft//2+1, T_spec] (complex)
        # output: [B, T_wav]
        if not self.center:
            raise NotImplementedError("center=False is currently not implemented. "
                "Please set center=True")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wav = torch.istft(spec, self.n_fft, hop_length=self.hop_size,
                win_length=self.win_size, center=self.center, normalized=self.normalized,
                window=self.window, onesided=True, return_complex=False)

        return wav


class CompressedSTFT(STFT):
    def __init__(
        self,
        n_fft: int,
        hop_size: int,
        win_size: int,
        win_type: str = "hann",
        normalized: bool = False,
        compression: float = 1.0,
        discard_last_freq_bin: bool = False,
        eps: float = 1.0e-5,
    ) -> None:
        assert compression <= 1.0, compression
        super().__init__(
            n_fft=n_fft, hop_size=hop_size, win_size=win_size,
            win_type=win_type, normalized=normalized, magnitude=False
        )
        self.compression = compression
        self.eps = eps
        self.discard_last_freq_bin = discard_last_freq_bin
    
    def forward(self, x: Tensor) -> Tensor:
        # x: [B, 1, T_wav] or [B, T_wav]
        # output: [B, n_fft//2, T, 2] (real) if discard_last_freq_bin=True
        # output: [B, n_fft//2+1, T, 2] (real) if discard_last_freq_bin=False
        x = super().forward(x)
        if self.discard_last_freq_bin:
            x = x[:, :-1, :, :]
        mag = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        x = x * mag.pow(self.compression - 1.0)
        return x

    def inverse(self, x: Tensor) -> Tensor:
        # x: [B, n_fft//2, T] (complex) if discard_last_freq_bin=True
        # x: [B, n_fft//2+1, T] (complex) if discard_last_freq_bin=False
        # output: [B, T_wav]
        mag_compressed = x.abs()
        x = x * mag_compressed.pow(1.0 / self.compression - 1.0)
        if self.discard_last_freq_bin:
            x = F.pad(x, (0, 0, 0, 1))  # [B, n_fft//2F+1, T]
        return super().inverse_complex(x)
