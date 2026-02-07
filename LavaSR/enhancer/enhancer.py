## Proposed work 
## This BWE model is based on Vocos, excellant speed with good quality.


import torch
import types
from vocos import Vocos

## used to improve quality in end
from LavaSR.enhancer.linkwitz_merge import FastLRMerge

## quick monkey patch to improve quality slightly
def custom_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the ISTFTHead module.

    Args:
        x (Tensor): Input tensor of shape (B, L, H)

    Returns:
        Tensor: Reconstructed time-domain audio signal
    """
    x = self.out(x).transpose(1, 2)
    mag, p = x.chunk(2, dim=1)
    mag = torch.exp(mag)
    mag = torch.clip(mag, max=1e3)
    x_real = torch.cos(p)
    x_imag = torch.sin(p)
    S = mag * (x_real + 1j * x_imag)
    audio = self.istft(S)
    return audio
  
class LavaBWE:
    def __init__(self, model_path, device='cpu'):
      
        self.device = device
        self.lr_refiner = FastLRMerge(device=device)
        self.bwe_model = Vocos.from_pretrained(model_path).eval().to(device)
        self.bwe_model.head.forward = types.MethodType(custom_forward, self.bwe_model.head)

        

    def infer(self, wav, autocast=False):
        """Inference function for bwe"""
      
        wav = wav.to(self.device)
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.float16, enabled=autocast):
            features_input = self.bwe_model.feature_extractor(wav)
            features = self.bwe_model.backbone(features_input)
            pred_audio = self.bwe_model.head(features)
            pred_audio = self.lr_refiner(pred_audio[:, :wav.shape[1]], wav[:, :pred_audio.shape[1]])

        return pred_audio



