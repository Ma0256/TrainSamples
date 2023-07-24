# plot time and frequency from files in data dir 'Fehlerübersicht', according to notebook 'Masterthesis'
# this file expects the data dir as working dir
import pandas as pd
import io
# import IPython.display as ipd
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
import numpy as np
import scipy
from scipy.io import wavfile
from pathlib import Path

# data dir for this project
demo_dir = Path.home() / "prj" / 'Fehlerübersicht'


def plot_normal_prominent(normal, prominent, samplerate=1, axs=None, showcolorbar=True, cmap='jet', **kwargs):
    if not len(normal) == len(prominent):
        # raise ValueError
        pass
    nrows = 2  # 3 if showcolorbar else 2
    fig, axs = axs or plt.subplots(nrows=nrows, ncols=2, sharex=True, sharey="row", figsize=(16 / 2, 9 / 2))

    for c, (k, x) in enumerate(dict(normal=normal, prominent=prominent).items()):
        t = np.linspace(0., x.shape[0] / samplerate, x.shape[0])
        ax = axs[0, c]
        ax.plot(t, x)
        # ax.set_title("normal")
        # ax.set_xlabel("time [s]")
        # ax.set_ylabel("sound preassure [pa]")
        ax.grid()
        ax.margins(x=0)

        ax = axs[1, c]
        # ax.set_title("spectrogram")
        ax.set_xlabel("time [s]")
        # ax.set_ylabel("frequency [Hz]")
        # default NFFT=256 stride 128
        pxx, freq, t, im = ax.specgram(x / 0.00002, Fs=samplerate, scale='dB', mode='magnitude', vmin=20.0,
                                       vmax=100.0,
                                       cmap=cmap, **kwargs)
        ax.set_ylim(50, 10000)

    axs[0, 0].set_title("normal")
    axs[0, 1].set_title("prominent")
    axs[0, 0].set_ylabel("sound preassure [pa]")
    axs[1, 0].set_ylabel("frequency [Hz]")
    if showcolorbar:
        plt.subplots_adjust(bottom=0.1, right=0.83, top=0.9)
        cax = plt.axes([0.87, 0.1, 0.02, 0.4])
        plt.colorbar(mappable=im,
                     cax=cax,
                     # fraction=0.07,
                     # pad=0.5,
                     # panchor=(2, 20)
                     ).set_label('sound power level [dB]')
    return


# the data
examples = {
    "Bogengeräusche": dict(
        normal=r'.\Bogengeräusche\N81_602HS_002_2016-10-15_20-06-17_01.ZS.wav',
        auffaellig=r'.\Bogengeräusche\N81_602HS_002_2016-10-21_17-14-47_01.ZS.wav',
    ),
    "Flachstellen Variante Güterzug": dict(
        normal=r'.\Flachstellen\2015-05-26_03-05-08_01.ZS.txt.wav',
        auffaellig=r'.\Flachstellen\2015-05-26_00-49-13_01.ZS.txt.wav'
    ),
    "Mikrophon Deffekt (Membran)": dict(
        normal=r'.\Sensorfehler Membran\G1_2019-04-30_05-53-10_01.ZS.txt',
        auffaellig=r'.\Sensorfehler Membran\G1_2019-04-30_05-53-10_02.ZS.txt'
    ),
    "Mikrophon Deffekt (Membran) Variante 1 (weniger Pegeldiff)": dict(
        normal=r'.\Sensorfehler Membran\G1_2019-04-29_12-03-14_01.ZS.txt',
        auffaellig=r'.\Sensorfehler Membran\G1_2019-04-29_12-03-14_02.ZS.txt',
    ),
    "Mikrophon Deffekt (Membran) Variante 2 (weniger Pegeldiff)": dict(
        normal=r'.\Sensorfehler Membran\G1_2019-04-29_12-16-37_01.ZS.txt',
        auffaellig=r'.\Sensorfehler Membran\G1_2019-04-29_12-16-37_02.ZS.txt',
    ),
    "Beschleunigungsaufnehmer Deffekt": dict(
        normal=r'.\Sensorfehler Membran\G1_2019-04-30_05-53-10_04.ZS.txt',
        auffaellig=r'.\Sensorfehler Membran\G1_2019-05-15_15-27-02_04.ZS.txt',
    ),
    "Annenheim (Beschleunigungen)": dict(
        normal=r'.\Fehler Beschleunigungen (2020-915)\Gleis1_2021-03-10_15-56-05_04.ZS.txt',
        auffaellig=r'.\Fehler Beschleunigungen (2020-915)\Gleis1_2021-03-10_15-56-05_03.ZS.txt',
    ),
}

if __name__ == "__main__":

    # to list
    items = [{**dict(title=k), **v} for k, v in examples.items()]

    # select which to plot
    items = items[:2]
    # read and plot them
    for item in items:
        sel = item["title"]
        dataNormal = demo_dir / item["normal"]
        dataAuffaellig = demo_dir / item["auffaellig"]
        if item["normal"].endswith("wav"):
            samplerate, dataNormal = wavfile.read(dataNormal)
            samplerate, dataAuffaellig = wavfile.read(dataAuffaellig)
        else:
            samplerate = 51200
            colnames = ['time', 'pressure']
            dataNormal = pd.read_csv(dataNormal, sep="\t", names=colnames, header=None, decimal=",")["pressure"]
            dataAuffaellig = pd.read_csv(dataAuffaellig, sep="\t", names=colnames, header=None, decimal=",")["pressure"]

        plot_normal_prominent(normal=dataNormal, prominent=dataAuffaellig, samplerate=samplerate,
                              NFFT=1024, noverlap=512)
        plt.suptitle(sel)  # , fontsize=32)

    plt.show()
    print("done")
