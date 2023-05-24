# read WAV file metadata from "Sampledateien" dir
from pathlib import Path

import pandas as pd
import numpy as np
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import librosa
from pprint import pprint
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
from TrainSoundExploration import demo_dir, examples

# data dir for this project
prj_root = Path.home() / "prj"
sample_dir = prj_root / "Sampledateien"


def plot_spectrogram(spec, title=None, aspect='auto', xmax=None, ax=None, fig=None, vrange=None, scale="linear",
                     Fs=1, stride=1, n_fft=1,
                     **kwargs):
    if Fs is None:
        ylabel = 'freq_bin'
        xlabel = 'frame'
        dt = df = 1
    else:
        dt = stride/Fs
        df = Fs/n_fft
        ylabel = 'f [Hz]'
        xlabel = 't [s]'
    if not ax:
        fig, ax = plt.subplots(1, 1)
        ax.set_title(title or 'Spectrogram')
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
    scale = scale.lower()
    if scale == "db":
        spec = librosa.power_to_db(spec)
    if vrange is not None:
        assert vrange > 0
        kwargs.update(vmin=np.max(spec) - vrange)
    extent = (0, spec.shape[1]*dt, 0, spec.shape[0]*df)
    im = ax.imshow(spec, origin='lower', aspect=aspect, extent=extent, **kwargs)
    if xmax:
        ax.set_xlim((0, xmax))
    if fig:
        fig.colorbar(im, ax=ax).set_label(f'sound power level [{"dB" if scale == "db" else "1"}]')
    plt.show(block=False)


def plot_mel_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Filter bank')
    axs.imshow(fbank, aspect='auto')
    axs.set_ylabel('frequency bin')
    axs.set_xlabel('mel bin')
    plt.show(block=False)


if __name__ == "__main__":
    data_dir = sample_dir
    # data_dir = acr_dir
    # data_dirs = [v for v in acr_dir.glob("*") if v.is_dir()]
    # files = [list(data_dir.glob("*.wav")) for data_dir in data_dirs]
    files = list(data_dir.glob("**/*.wav"))

    # get metadata, takes a lot of time
    print("Reading audio metadata ...")
    info = {k.stem[:-3]: vars(torchaudio.info(k)) for k in tqdm(files) if k.stem.endswith("01")}
    info = pd.DataFrame(info).T.infer_objects()
    info["dur"] = info["num_frames"] / info["sample_rate"]
    print("First file metadata:")
    print(info.iloc[0])

    n_fft = 1024
    n_fft = 256
    hop_length = n_fft//2
    # define transformation
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
    )

    file = demo_dir / examples["Bogengeräusche"]["auffaellig"]
    # file = sample_dir / f"{info.index[1]}_01.wav"
    cmap = "jet"
    wave, sr = torchaudio.load(file)
    spec = spectrogram(wave)[0]
    dt = hop_length/sr
    df = sr/n_fft

    # pick first 2D-page
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
    fig.suptitle(file.parts[-1])
    plot_spectrogram(spec, ax=axs[1], fig=fig, title="Spectrogram - torchaudio",
                     cmap=cmap, Fs=sr, stride=hop_length, n_fft=n_fft, vrange=60)
    # MPL spectrogram
    pxx, freq, t, im = axs[0].specgram(  # wave[0]*5e4, vmin=20.0, vmax=100.0, scale="dB", mode="magnitude",
        wave[0], mode="magnitude", vmin=-60,  # vmax=0, #scale="linear",
        cmap=cmap,
        Fs=sr,
    )
    axs[0].set_title("plt.specgram")
    fig.colorbar(im, ax=axs[0])

    #n_fft = 256
    n_mels = 64
    #sample_rate = 6000
    mel_filters = F.melscale_fbanks(
        int(n_fft // 2 + 1),
        n_mels=n_mels,
        f_min=0.0,
        f_max=sr / 2.0,
        sample_rate=sr,
        norm="slaney",
    )
    plot_mel_fbank(fbank=mel_filters)

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    )

    melspec = mel_spectrogram(wave)[0]
    plot_spectrogram(melspec, title="MelSpectrogram - torchaudio", ax=axs[2], fig=fig, cmap=cmap,
                     Fs=sr, n_fft=2*n_mels, stride=hop_length)
    axs[2].set_ylabel("mel freq")
    #axs[2].set_ylim([0, 10000])
    plt.show()
    print("done")
