import numpy as np
import pandas as pd
from math import log10, floor, ceil
import torch
import torchaudio
from librosa import power_to_db
from pathlib import Path
from tqdm import tqdm
from convert_labels import acr_dir
from TrainSamples import plot_spectrogram
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from itertools import chain
from functools import partial

# for test code
from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.metrics.cluster import contingency_matrix


def unchain(x, ls):
    js = np.cumsum(ls)
    if js[-1] != len(x):
        raise ValueError("sum of 'ls' doesn't match 'len(x)'")
    y = np.split(x, js[:-1])
    return y


def mapr(func, x):
    return [mapr(func, v) for v in x] if hasattr(x, "__len__") else func(x)


# round x to n digits ending to multiples of d
def round_mult(x, n, d):
    return round(x / d, n) * d


def round_preferred(x, ena_round2=False):
    if x <= 0:
        raise ValueError
    e = floor(log10(x))
    m = x / 10 ** e
    if m < 5:
        d = 5
        if ena_round2 and m < 2.5:
            d = 2
    elif m < 7.5:
        d = 10
    else:
        d = 20
    v = round_mult(x, 2 - e, d)
    if d < 5:
        v5 = round_mult(x, 2 - e, 5)
        if abs(v5 - x) < abs(v - x):
            v = v5
    return v


def third_octave_center(low=15, high=20000, nominal=False):
    low = ceil(10 * log10(low / 1000))
    high = floor(10 * log10(high / 1000))
    f = [10 ** (v / 10) * 1e3 for v in range(low, high + 1)]
    if nominal:
        f = [round_preferred(v) for v in f]
    return f


def third_octave_bands(low=15, high=20000, nominal=False):
    fm = third_octave_center(low=low, high=high, nominal=nominal)
    if nominal:
        # get nominal freqs via upper cut off: results in values identical to
        # https://de.wikipedia.org/wiki/Normfrequenz
        fc = [v * 10 ** .05 for v in fm[:-1]]
        fc = [round_preferred(v, ena_round2=True) for v in fc]
    else:
        # no rounding = even logarithmic spacing
        # geometric mean
        # fc = [(a * b) ** .5 for a, b in zip(fm, fm[1:])]
        # same result as geometric mean
        fc = [v * 10 ** .05 for v in fm[:-1]]
    return fc


def crop(X, crop_n, reduce_spectrum):
    # crit = np.array([reduce_spectrum(v) for v in X])
    # faster
    crit = reduce_spectrum(X)
    ix = np.argmax(crit)
    ix = np.clip(ix - crop_n // 2, 0, max(0, len(X) - crop_n)).astype(int)
    ix = [ix, ix + crop_n]
    # fig, axs = plt.subplots(2, 1, sharex=True)
    # axs[1].plot(np.arange(len(crit)) * split, crit)
    # plot_spectrogram(X[i].T, Fs=1 / split, n_fft=n_fft / split / 32e3, stride=1, cmap="jet",
    #                  ax=axs[0],  # fig=fig,
    #                  db_range=40)
    # axs[0].set_title(f"{label[y[i][0]]}")
    # # vertical marker
    # axs[0].plot(np.repeat(ix, 2) * split, band[::-1] + band, 'r', linewidth=2)
    l0 = len(X)
    X = X[ix[0]:ix[1], :]
    if not (len(X) == crop_n or len(X) == l0):
        raise ValueError
    return X


# return acramos dataset as X, y
def load_acramos(dir=acr_dir, n_fft=1024, stride=None, n_mels=512, n_mfcc=512, tf_trans="Spectrogram",
                 trimlr_secs=(0, 0), split=0, crop_n=0, aggregate=np.mean, return_X_y=False, feat_cache: str = '',
                 spectral_pool=0):
    sr = 32000
    stride = stride or n_fft // 2
    # tf_trans = "MFCC"
    if feat_cache:
        feat_cache = Path(feat_cache).with_suffix(".npz")

    labels = pd.read_csv(dir / "adsim_label.csv")
    # sort labels by length: a hack that returns the negative class as 0
    label = sorted(np.unique(labels["Label"]), key=len)
    labels["class"] = [label.index(k) for k in labels["Label"]]
    train = labels[labels["campaign"] != "S7"]
    try:
        X = np.load(feat_cache, allow_pickle=True)['X']
        y = np.load(feat_cache, allow_pickle=True)['y']
        files = np.load(feat_cache, allow_pickle=True)['files']
    except FileNotFoundError as e:
        # find the loudest spot in higher band
        # BEGEL tonale Auffälligkeiten 1,25 kHz bis 12,5 kHz
        band = (2000, sr // 2)
        # band = (1250, 12500)
        nf = n_fft // 2 + 1  # X.shape[1]
        l = band[0] * (nf - 1) // band[1]
        # reduce_spectrum = lambda x: np.mean(np.sort(x[..., l:], axis=-1)[-1:])  # np.max
        reduce_spectrum = lambda x: np.mean(np.sort(x[..., l:], axis=-1)[:])  # np.mean
        # reduce_spectrum = lambda x: np.mean(np.sort(x[..., l:], axis=-1)[..., -10:], axis=-1)

        feats, y, files = [], [], []
        for i in tqdm(range(len(labels))):
            r = labels.iloc[i, :]
            file = f"{r['campaign']}/{r['ID']}_01.wav"
            files.append(file)
            wave, sr = torchaudio.load(acr_dir / file)
            if tf_trans == "Spectrogram":
                spectrogram = torchaudio.transforms.Spectrogram(
                    n_fft=n_fft,
                    hop_length=stride,
                    center=True,
                    pad_mode="reflect",
                    power=2.0,
                )
            elif tf_trans == "MFCC":
                spectrogram = torchaudio.transforms.MFCC(
                    sample_rate=sr,
                    n_mfcc=n_mfcc,
                    melkwargs={
                        "n_fft": n_fft,
                        "n_mels": n_mels,
                        "hop_length": stride,
                        "mel_scale": "htk",
                    },
                )
            else:
                raise ValueError
            tf2d = spectrogram(wave)[0].numpy()
            if trimlr_secs:
                dix = (np.array(trimlr_secs) * sr / stride).astype(int)
                tf2d = tf2d[:, dix[0]:tf2d.shape[1] - dix[1]]
            if crop_n:
                tf2d = crop(tf2d.T, crop_n=crop_n, reduce_spectrum=reduce_spectrum).T
            if split:
                d = int(split * sr // stride)
                # trim odd end
                tf2d = tf2d[:, :(tf2d.shape[1] // d) * d]
                tf3d = np.split(tf2d, np.arange(d, tf2d.shape[1], d), axis=1)
                feats.append(np.array([np.mean(v, axis=1) for v in tf3d]))
                y.append(np.array([labels["class"][i]] * len(tf3d)))
            else:
                feat = aggregate(tf2d, axis=1)
                # feat = tf2d.max(axis=1)
                feats.append(feat)
                y.append(labels["class"][i])
        X = feats
        # y = np.array(y)
        if feat_cache:
            np.savez(file=feat_cache, X=X, y=y, files=files)

    if spectral_pool:
        X = [np.max(np.reshape(v[:, 1:], (len(v), -1, spectral_pool)), axis=-1) for v in X]

    if return_X_y:
        return X, y
    else:
        return dict(X=X, y=y, target_names=label, files=files, fs=sr)


# auto label events in a single spectrogram in decibel. The event peak loudness must be contained in "in_band", larger
# than "q_db" and drop to both sides within frequency span "s" by more than "d_db".
def machine_label_spectrogram(X, q_db, d_db, in_band, s=1 / 16, positives=[None, "[Quietschen]"], use_kernel=False):
    ys = []
    k = "[Quietschen]"
    if k in positives:
        k = positives.index(k) + 1
        for v in X:
            if use_kernel:
                # slide a non-linear kernel
                ks = int(s * len(v))
                cs = []
                for i in np.arange(0, len(v) - ks):
                    m = i + ks // 2
                    c = (np.min(v[i:m]), v[m], np.min(v[m:i + ks]))
                    if m in in_band:
                        cs.append(c[0] < c[1] - d_db and c[1] > q_db and c[2] < c[1] - d_db)
                c = any(cs)
            else:
                # maximum in band
                # i = min(in_band[np.argmax(v[in_band])], len(v) - 2)
                i = in_band[np.argmax(v[in_band])]
                l = max(0, i - int(s * len(v) // 2))
                h = l + int(s * len(v))
                c = np.min(v[l:i]), v[i], np.min(v[i:h])
                c = c[0] < c[1] - d_db and c[1] > q_db and c[2] < c[1] - d_db
            ys.append(c * k)
    ys = np.array(ys)
    return ys


# contains the "secrets" for labelling, i.e., parameters, settings...
# X in decibels
def machine_label_dataset(X_db, fs):
    in_band = 1250, fs / 2
    freq = np.linspace(0, fs / 2, len(X_db[0][0]))
    in_band = np.where((in_band[0] <= freq) & (freq <= in_band[1]))[0]
    label_kwargs = dict(in_band=in_band,
                        q_db=30,
                        d_db=10,
                        s=1 / 10,  # proportion of fs / 2
                        # s = 1/16,   # does not find larger f-span sqeal events
                        use_kernel=True,
                        )

    y_auto = [machine_label_spectrogram(v, **label_kwargs) for v in tqdm(X_db)]
    # debug a = machine_label(X_db[1790], **label_kwargs)
    return y_auto


# convert slice to file label
def conv_slice2file(y):
    yf = []
    for v in y:
        v = tuple(sorted(set(v)))
        l = 0
        if v[0] == 0:
            v = v[1:]
        if len(v) == 1:
            l = v[0]
        elif len(v) > 1:
            m = {(1, 2): 3}
            l = m[v]
        yf.append(l)
    return yf


# wraps classification_report. 'names' is interpreted differently from sklearn 'target_names'. 'names' index must
# correspond to labels in 'y_pred', 'y_true' - different to sklearn classification_report.
def file_classification_report(y_true, y_pred, labels=None, names=None, output_dataframe=False, **kwargs):
    if labels is None:
        labels = unique_labels(y_true, y_pred)
    # select used target_names from labels
    if names is None:
        target_names = kwargs.pop("target_names", None)
    else:
        target_names = list(np.array(names)[labels])
    if len(y_true) != len(y_pred) or hasattr(y_true[0], "__len__") or hasattr(y_pred[0], "__len__"):
        raise ValueError("labels must be 1d sequences of equal length")
    # take care of zero division warning
    prevalence = np.bincount(y_true, minlength=len(labels))
    if 'zero_division' not in kwargs and any(prevalence == 0):
        # zero division will occur, disable the warning
        kwargs['zero_division'] = 0

    if output_dataframe:
        cr = classification_report(y_true, y_pred=y_pred, labels=labels, target_names=target_names,
                                   output_dict=True, **kwargs)
        # harmonize data for columns
        if "accuracy" in cr:
            cr["accuracy"] = {
                "f1-score": cr["accuracy"],
                "support": cr["macro avg"]["support"],
            }
        cr = pd.DataFrame(cr).T.convert_dtypes()
    else:
        cr = classification_report(y_true, y_pred=y_pred, labels=labels, target_names=target_names, **kwargs)
    return cr


# wraps file classification_report for ragged sequence of labels
def slice_classification_report(y_true, y_pred, labels=None, names=None, output_dataframe=False, **kwargs):
    #if hasattr(y_true, "__len__") and hasattr(y_pred, "__len__"):
    # file level comparison
    if [len(v) for v in y_true] != [len(v) for v in y_pred]:
        raise ValueError("'y_true' item lengths different to 'y_pred'")
    d = dict(y_true=y_true, y_pred=y_pred)
    d = dict(file=[np.array(conv_slice2file(v)) for k, v in d.items()],
             slice=[np.array([*chain(*v)]) for k, v in d.items()])

    for k, (y_true, y_pred) in d.items():
        d[k] = file_classification_report(y_true, y_pred, labels=labels, names=names, output_dataframe=output_dataframe,
                                          **kwargs)
    return d


def confusion_matrix_df(y_pred, y_true, names=None):
    #cm = (pd.crosstab(pd.Series(names[y_true]), names[y_pred]))
    cm = confusion_matrix(y_true, y_pred)
    labels = unique_labels(y_true, y_pred)
    if names is None:
        names = labels
    cm = pd.DataFrame(cm, index=names, columns=names)
    return cm


# wraps classification_report for ragged sequence of labels
def slice_error_report(y_true, y_pred):
    if [len(v) for v in y_true] != [len(v) for v in y_pred]:
        raise ValueError("'y_true' has different length than 'y_pred'")
    yf, yhf = [np.array(conv_slice2file(v)) for v in [y_true, y_pred]]

    e = [np.where(y != yh)[0] for y, yh in zip(y_true, y_pred)]
    pos = [np.where(y)[0] for y in y_true]
    fp = [np.where((y == 0) & (y != yh))[0] for y, yh in zip(y_true, y_pred)]
    fn = [np.where((y != 0) & (y != yh))[0] for y, yh in zip(y_true, y_pred)]
    edf = pd.DataFrame(dict(error=np.not_equal(yf, yhf), y_pred=yhf, y_true=yf,
                            slice_fp=fp, slice_fn=fn, slice_pos=pos))
    # select files with slice errors (not identical to file errors)
    edf = edf.loc[[len(v) > 0 for v in e]]
    return edf


def load_acramos_slices(subset=None, n=None, reindex=False, label_report=False, do_plot_spectra=False, **kwargs):
    compare_manual = subset is None

    # event times in seconds (manually labelled by spectrogram inspection)
    # manual label for "Quietschen"
    # event occurrence at second
    # difficult case #s: 13
    t_2_man = [[2], [0], [4, 5, 6], [0, 1, 2, 3], [2, 3, 5, 8, 17, 18, 19], [0, 1, 2, 3, 5, 6], [6, 7, 8, 9],
               [0, 1], [4, 5], [2],
               [8, 9, 10, 12, 13], [20, 21, 22, 23, 24], [2, 5, 6], [4, 5], [12, 14, 15, 16], [3, 4, 5, 6, 7, 8, 9],
               [3, 4, 5, 6], [3, 4, 5, 6, 7, 8], [5, 6, 7], [4],
               [1, 2], [4, 5, 6], [4, 5, 6], [1, 2], [7, 8, 9, 10, 11], [1, 2, 3, 4], [4], [4, 5], [4, 5],
               [4, 5, 6, 7, 8, 9, 11],
               [4, 5], [1, 2, 3], [4, 5], [4], [5, 6], [4, 5], [5, 6], [6, 7], [1, 2], [4, 5, 6, 7],
               [0, 1], [6, 7, 8], [0, 4, 5], [1, 2, 3, 7, 8, 9], [4, 5], [3, 4, 5], [3, 4, 5], [4, 5],
               [0, 1, 4, 5, 6], [3, 4, 5],
               [0, 1, 4], [0, 1, 4, 5], [4, 5], [4, 5], [1, 6, 7, 8, 9, 10], [4], [4, 5, 6], [4, 5], [0, 1, 4, 5],
               [4, 5],
               [1, 2], [4], [3, 4, 5], [3, 4, 5], [4, 5], [4, 5], [5, 6, 7, 8], [3, 4, 5], [4, 5], [4],
               ]

    if compare_manual:
        subset = ["[Quietschen]"]
        subset = ["[Negativ]", "[Quietschen]"]
        t_2 = t_2_man
        if not n:
            n = len(t_2)
        # n = 32
        assert 0 < n <= len(t_2)
        t_2 = t_2[:n]
        if "[Negativ]" in subset:
            # all files must be empty
            t_2 = [[]] * n + t_2
        print(f"Slice level comparing {len(t_2)} files of auto-labelling to manually labelled 'Quietschen' slices ")
    else:
        t_2 = None

    # select 'classes' from dataset
    ds = load_acramos(**kwargs)
    y = ds['y']
    X = ds['X']
    files = ds["files"]
    classes = [ds["target_names"].index(k) for k in subset]
    # classes = range(4)
    if True:
        # use np indexing
        y_files = np.array([v[0] for v in y])
        ixs = [np.where(v == y_files)[0][:n] for v in classes]
        ixs = np.concatenate(ixs)
        # this is a hack: reorder items (1st item as last) for nicer plotting result, should not change ML results
        # ixs = np.roll(ixs, -1)
        y = list(y[ixs])
        X = list(X[ixs])
        files = list(files[ixs])
    else:
        # w/o indexing pure python
        ds = [[(k, v, v2) for k, v, v2 in zip(y, X, files) if k[0] == ix][:n] for ix in ixs]
        y, X, files = [v for v in zip(*[(k, v, v2) for vs in ds for k, v, v2 in vs])]

    print("auto labelling ...")
    y_auto = machine_label_dataset(X_db=[power_to_db(v, top_db=None) for v in X], fs=ds['fs'])

    if compare_manual:
        assert len(t_2) == len(y)
        # transform file label to slice (=event) label
        # put label at positions in 't_2', rest is zero
        y_slice = [np.bincount(ixs, minlength=len(k)) * k[0] for ixs, k in zip(t_2, y)]

        cr = slice_classification_report(y_slice, y_pred=y_auto,
                                         labels=np.arange(len(ds['target_names'])),
                                         target_names=ds['target_names'])
        print(cr['slice'])
        # label differences with auto label
        er = slice_error_report(y_slice, y_pred=y_auto)
        if len(er):
            print(f"{len(er)} files with slice errors")
            print(er)
            print()

    if label_report or compare_manual:
        # file level comparison

        print(f"File label from slice auto-label vs. acramos file label " +
              f"{'' if compare_manual else '(no slice label available)'}")
        cr = slice_classification_report(y, y_pred=y_auto,
                                         labels=np.arange(len(ds['target_names'])),
                                         target_names=ds['target_names'])
        print(cr['file'])
        print("confusion matrix")
        print(confusion_matrix(conv_slice2file(y), conv_slice2file(y_auto)))
        er = slice_error_report(y, y_auto)
        er = er.loc[er['error'], ['y_pred', 'y_true']]
        if len(er):
            print(f"{len(er)} auto-label vs. acramos file errors")
            print(er)

    #
    vrange = 40
    X_db = [power_to_db(v, top_db=None) for v in X]
    # X_db = [power_to_db(v, top_db=vrange) for v in X]
    # db normalized
    # X_db = [v - np.max(v) + vrange for v in X_db]
    # scaled linear magnitude from power spectrum
    # X_db = [vrange*(v**.5)/np.max(v**.5) for v in X]
    # X_db = [(v**.5) for v in X]
    # scaled power spectrum
    # X_db = [v/np.max(v) for v in X]
    # X_db = X
    # scale each item
    # X_db = [v*vrange/np.max(v) for v in X_db]
    # use global scale (vs. individual scale)
    vmin, vmax = [(np.min(v), np.max(v)) for v in [np.concatenate(X_db)]][0]
    # too high vmin for sqeal file #125, misses sqeal at x = 4  vrange = 40, must limit vmin
    # no label secrets allowed vmax = min(label_kwargs["q_db"] - label_kwargs["d_db"] + vrange, vmax)
    vmin = vmax - vrange
    X_db = [np.clip(v, a_min=vmin, a_max=vmax) for v in X_db]
    if do_plot_spectra:
        # vmin, vmax = 0, 40
        y_files = [v[0] for v in y]
        for i, v in enumerate(X_db):
            title = f"{label[y[i][0]]} {files[i]}"
            plot_spectrogram(v.T,
                             Fs=fs, n_fft=1024, stride=32e3, title=title,
                             scale="linear", cmap="jet",  # vrange=vrange,
                             # vmin=vmin, vmax=vmax,
                             )
            # plt.ylim([0, 1e4])

    # change classes to range(n_classes)
    if reindex:
        def reindex(x):
            ls = [len(v) for v in x]
            c, yr = np.unique([*chain(*x)], return_inverse=True)
            return unchain(yr, ls)

        # include negatives
        #target_names = list({**dict.fromkeys(ds["target_names"][:1]), **dict.fromkeys(subset)})
        c = np.unique([*chain(*chain(y, y_auto))])
        y = mapr(list(c).index, y)
        #y_auto = reindex(y_auto)
        y_auto = mapr(list(c).index, y_auto)
        # negative class needed for slice level operation
        ds["target_names"] = [ds["target_names"][k] for k in c]

    # ds.pop('y')
    ds['X'] = X_db
    ds['y'] = y_auto
    ds['y_file'] = conv_slice2file(y_auto)#[sorted(set(v))[-1] for v in y_auto]
    ds["files"] = files
    # the acramos file label from the database
    ds['y_acramos'] = [v[0] for v in y]

    return ds


def train_test_split_acramos(ds, test_size=0.2, random_state=42):
    # n = max([len(v) if hasattr(v, "__len__") else 1 for v in ds.values()])
    n = len(ds['X'])
    data_items = {k: v for k, v in ds.items() if hasattr(v, "__len__") and len(v) == n}
    other_items = {k: ds[k] for k in set(ds) - set(data_items)}

    # X, y, t_2, X_ho1, y_ho1, t_2_ho1 = X[-300:], y[-300:], t_2[-300:], X[:-300], y[:-300], t_2[:-300]
    split = train_test_split(*data_items.values(), test_size=test_size, random_state=random_state)
    split = list(zip(*np.reshape(np.array(split, dtype=object), (-1, 2))))
    train, test = [{k: v for k, v in zip(data_items, vs)} for vs in split]

    return {**train, **other_items}, {**test, **other_items}


def prevalence(ds: dict, margin=False):
    if hasattr(list(ds.values())[0]['y'][0], "__len__"):
        rs = [pd.Series(np.bincount(conv_slice2file(d['y']), minlength=len(d['target_names'])), index=d['target_names'])
              for d in ds.values()]
        prev = dict(file=pd.DataFrame(rs, index=list(ds)))
        rs = [pd.Series(np.bincount([*chain(*d['y'])], minlength=len(d['target_names'])), index=d['target_names'])
              for d in ds.values()]
        prev['slice'] = pd.DataFrame(rs, index=list(ds))
    else:
        rs = [pd.Series(np.bincount(d['y'], minlength=len(d['target_names'])), index=d['target_names'])
              for d in ds.values()]
        prev = dict(file=pd.DataFrame(rs, index=list(ds)))

    if margin:
        prev = {k: v.append(v.agg(["sum"])) for k, v in prev.items()}
        prev = pd.concat(prev, axis=0).T
        prev = prev.append(prev.agg(["sum"]))
    else:
        prev = pd.concat(prev, axis=0).T
    return prev


if __name__ == "__main__":
    # ##################################################################################################################
    # Test third octave bands ##########################################################################################
    f_one_third = [10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80,
                   100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                   1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                   10000, 12500, 16000, 20000]

    # preferred 1/3 octave band center frequencies, according ISO 266 (rounding for preferred numbers)
    fm = third_octave_center(10, 16000, nominal=False)
    fmp = third_octave_center(10, 20000, nominal=True)
    assert all(np.isclose(f_one_third, fmp)), "normative values don't match"

    # preferred cut-offs according ISO
    fcp = third_octave_bands(10, 16000, nominal=True)
    fc = third_octave_bands(10, 16000, nominal=False)
    # fc = [(a * b) ** .5 for a, b in zip(fm, fm[1:])]
    fcp = [round_preferred(v, ena_round2=True) for v in fc]

    # ##################################################################################################################
    # ##################################################################################################################
    # use uniform subset
    #subset = ["[Negativ]", "[Quietschen]"]
    subset = ["[Kreischen]", "[Quietschen]"]
    # classes = [label.index("[Kreischen]"), label.index("[Quietschen]"), label.index("[Kreischen][Quietschen]")]
    # classes = [label.index("[Kreischen]"), label.index("[Kreischen][Quietschen]")]
    # classes = [label.index("[Kreischen][Quietschen]")]
    reindex = False
    ds = load_acramos_slices(subset=subset, #n=100,
                             reindex=reindex, label_report=True,
                             dir=acr_dir, split=1.0, aggregate=lambda x: x, spectral_pool=0,
                             # feat_cache="clustering")
                             feat_cache=Path(__file__).stem)
    labels = np.unique([*chain(*ds['y'])])
    names = np.array(ds['target_names'])

    print("")
    print("Train holdout split")
    ds_trn, ds_ho = train_test_split_acramos(ds)

    # flatten nested files
    X, y = [[*chain(*ds_trn[k])] for k in ['X', 'y']]
    X_ho, y_ho = [[*chain(*ds_ho[k])] for k in ['X', 'y']]

    # build DataFrame for display
    prev = prevalence(dict(train=ds_trn, holdout=ds_ho), margin=True)
    print(prev.to_string(col_space=([*[7] * (len(prev.columns) // 2), 15] * 2)[:len(prev.columns)]))

    print()
    print("Train on slices")
    # fit model on train data
    mdl = LinearDiscriminantAnalysis()
    #mdl = RandomForestClassifier()
    print(str(mdl))
    mdl.fit(X, y)
    yh = mdl.predict(X_ho)
    # reshape 1d list to ragged file structure
    yh = unchain(yh, ls=[len(v) for v in ds_ho['y']])
    print("Holdout test")
    if True:
        cr = slice_classification_report(ds_ho['y'], y_pred=yh, names=names, output_dataframe=True)
        cr = pd.concat(cr, axis=1)
        print(cr.to_string(col_space=[9, 9, 9, 9, 20, 9, 9, 9], float_format='{:0.2f}'.format).replace("<NA>", "    "))
        print()
    else:
        cr = slice_classification_report(ds_ho['y'], y_pred=yh, names=names)
        [(print(f"{k} level"), print(v)) for k, v in cr.items()]
    er = slice_error_report(ds_ho['y'], y_pred=yh,)

    print("Holdout test against file level original acramos label")
    # -not- the ground truth used for training
    cr = file_classification_report(ds_ho['y_acramos'], y_pred=conv_slice2file(yh), names=names, output_dataframe=True)
    # "display.precision", 2
    #with pd.option_context('display.float_format', '{:0.2f}'.format):
    print(cr.to_string(col_space=9, float_format='{:0.2f}'.format).replace("<NA>", "    "))
    print()

    print("Train on files")
    # aggregate = np.mean
    aggregate = partial(np.quantile, q=1)
    # aggregate time axis
    X, X_ho = [np.array([aggregate(v, axis=0) for v in vs]) for vs in [ds_trn['X'], ds_ho['X']]]
    y = ds_trn['y_file']

    # fit model on train data
    mdl = LinearDiscriminantAnalysis()
    #mdl = RandomForestClassifier()
    print(str(mdl))
    mdl.fit(X, y)
    yh = mdl.predict(X_ho)

    print("Holdout test")
    cr = file_classification_report(ds_ho['y_file'], y_pred=yh, names=names)
    cm = pd.crosstab(pd.Series(names[ds_ho['y_file']]), names[yh])
    print(f"{'file'} level")
    print(cr)
    pass
