## Proposed work 
## This BWE model is based on Vocos, excellant speed with good quality.


import torch
from vocos import Vocos

## used to improve quality in end
from LavaSR.enhancer.linkwitz_merge import FastLRMerge

## quick monkey patch to improve quality slightly
def forward(x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        self = vocos.head
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e3)  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)

    
        S = mag * (x + 1j * y)
        audio = self.istft(S)
        return audio
  
class LavaBWE:
    def __init__(self, model_path, device='cpu'):
      
        self.device = device
        self.lr_refiner = FastLRMerge(device=device)
        self.bwe_model = Vocos.from_pretrained(model_path).eval().to(device)
        

    def infer(self, wav, autocast=False):
        """Inference function for bwe"""
      
        wav = wav.to(self.device)
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.float16, enabled=autocast):
            features_input = self.bwe_model.feature_extractor(wav)
            features = self.bwe_model.backbone(features_input)
            pred_audio = self.bwe_model.head(features)
        return pred_audio



