# UNET Vocal Extractor
### Description
This is a simple PyTorch reimplementation of the original UNET paper with the intent of separating the vocal track from a given audio track. 
original paper: [U-Net paper](https://arxiv.org/abs/1505.04597)
It is trained off the MUSDB18 database, which comprises around 150 segmentations of songs, separated into individual audio stems.
This was a pretty simple implementation with minimal optimizations aside from basic DSP operations. The model learns off the magnitude spectral representation
of the training vocal sample, and later reconstructs the audio file using the phase info from the mix (certainly not ideal); Output from the actual UNet is a magnitude spectrograph.
Output quality varies based on the input audio file; it works with reasonable clarity with simpler songs without many instruments that clash with the vocals spectral range.

### Next Steps
Improved audio reconstruction - currently audio is reconstructed using the phase information of the mix - this is technically incorrect
We can do better by learning the phase directly in the model, or by generating phase information using a GAN vocoder at the output.  

Temporal loss - some modern models use a loss criterion characterized by both temporal and spectral info for better accuracy.

### Other Notes
- Prefilting of the mix samples at 12kHz was attempted with the hypothesis that it would limit the total error magnitude when compared to the vocals, which have most
of its character within 10kHz. This seemed to cause heavy error and distortion in the file output, likely because there were also frequencies above 12kHz in the vocal
sample as a result of spectral leakage from the STFT operations.
