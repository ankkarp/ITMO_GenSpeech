import argparse
from typing import Optional

import torch
from torch import nn
import torchaudio
from torchaudio import functional as F
import matplotlib.pyplot as plt


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cuda:0'

class LogMelFilterBanks(nn.Module):
    def __init__(self,
            n_fft: int = 400,
            samplerate: int = 16000,
            hop_length: int = 160,
            n_mels: int = 80,
            pad_mode: str = 'reflect',
            power: float = 2.0,
            normalize_stft: bool = False,
            onesided: bool = True,
            center: bool = True,
            return_complex: bool = True,
            f_min_hz: float = 0.0,
            f_max_hz: Optional[float] = None,
            norm_mel: Optional[str] = None,
            mel_scale: str = 'htk',
            device=DEVICE
        ):
        super(LogMelFilterBanks, self).__init__()
        # general params and params defined by the exercise
        self.n_fft = n_fft
        self.samplerate = samplerate
        self.window_length = n_fft
        self.window = torch.hann_window(self.window_length).to(device)
        self.device = device
        # Do correct initialization of stft params below:
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.center = center
        self.return_complex = return_complex
        self.onesided = onesided
        self.normalize_stft = normalize_stft
        self.pad_mode = pad_mode
        self.power = power
        # Do correct initialization of mel fbanks params below:
        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz or samplerate // 2
        self.norm_mel = norm_mel
        self.mel_scale = mel_scale
        self.n_freqs = self.n_fft // 2 + 1 if self.onesided else self.n_fft
        # finish parameters initialization
        self.mel_fbanks = self._init_melscale_fbanks()

    def _init_melscale_fbanks(self):
        return F.melscale_fbanks(
            n_freqs=self.n_freqs,
            f_min=self.f_min_hz,
            f_max=self.f_max_hz,
            n_mels=self.n_mels,
            sample_rate=self.samplerate,
            norm=self.norm_mel,
            mel_scale=self.mel_scale
        ).to(self.device)

    def spectrogram(self, x):
        return torch.stft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalize_stft,
            onesided=self.onesided,
            return_complex=self.return_complex
        )

    def forward(self, x):
        """
        Args:
            x (Torch.Tensor): Tensor of audio of dimension (batch, time), audiosignal
        Returns:
            Torch.Tensor: Tensor of log mel filterbanks of dimension (batch, n_mels, n_frames),
                where n_frames is a function of the window_length, hop_length and length of audio
        """
        spectogram = self.spectrogram(x)
        if self.return_complex:
            spectogram = torch.abs(spectogram) ** self.power
        else:
            spectogram = (spectogram[..., 0] ** 2 + spectogram[..., 1] ** 2) ** (self.power / 2)
        filterbanks = torch.matmul(self.mel_fbanks.T, spectogram)
        log_filterbanks = torch.log(filterbanks + 1e-6)
        # Return log mel filterbanks matrix
        return log_filterbanks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', type=str, help='Path to input WAV-file')
    parser.add_argument('--plot', type=str, help='Whether to plot spectogram comparison')
    args = parser.parse_args()

    signal, sr = torchaudio.load(args.input_file)

    signal = signal.to(DEVICE)
    melspec = torchaudio.transforms.MelSpectrogram(
        hop_length=160,
        n_mels=80
    ).to(DEVICE)(signal)
    logmelbanks = LogMelFilterBanks().to(DEVICE)(signal)
    if args.plot:
        fig, axes = plt.subplots(2, figsize=(10, 7))
        axes[0].imshow(logmelbanks[0], aspect='auto')
        axes[0].set_title('HW')
        axes[1].imshow(torch.log(melspec + 1e-6).numpy()[0], aspect='auto')
        axes[1].set_title('Torch (reference)')
        plt.savefig('plots.png')

    assert torch.log(melspec + 1e-6).shape == logmelbanks.shape
    assert torch.allclose(torch.log(melspec + 1e-6), logmelbanks)