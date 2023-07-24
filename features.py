import numpy as np
import pandas as pd
from math import log10, floor, ceil
import torch
import torchaudio
import torchaudio.functional as F
from librosa import power_to_db
from pathlib import Path
import re
from tqdm import tqdm
from typing import Dict

from TrainSamples import plot_spectrogram
from sklearn import decomposition, discriminant_analysis
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from itertools import chain, product
from functools import partial

# for test code
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score


# from sklearn.metrics.cluster import contingency_matrix


acr_dirs = dict(
    V1=Path.home() / 'prj' / 'acrDb',
    V2=Path(r"D:\ADSIM\Import-2023-04"),
)


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


# crop region sized 'crop_n' around location with maximum criterion from 'reduce_spectrum'
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


# return subset of 'dataset' dict comprised of first 'n' examples of categories defined by 'subset_labels'
def dataset_class_subset(ds, subset_labels, n_max=None, reindex=False):
    target_names = ds['target_names']
    # guarantee order
    delta = set(subset_labels) - set(target_names)
    assert not delta, f"'{delta}' not in targets"
    subset_labels = [v for v in target_names if v in subset_labels]
    classes = [target_names.index(k) for k in subset_labels]

    data_items = {k: v for k, v in ds.items() if hasattr(v, "__len__") and len(v) == len(ds['X'])}
    assert 'target_names' not in data_items
    other_items = {k: ds[k] for k in set(ds) - set(data_items)}

    # use np indexing
    y_files = np.array([v[0] for v in ds['y']]) if hasattr(ds['y'][0], "__len__") else np.array(ds['y'])
    ixs = np.concatenate([np.where(v == y_files)[0][:n_max] for v in classes])

    data_items = {k: [v[i] for i in ixs] for k, v in data_items.items()}
    if reindex:
        other_items['target_names'] = subset_labels
        #classes = sorted({v for vs in data_items['y'] for v in vs})
        if hasattr(data_items['y'][0], "__len__"):
            data_items['y'] = [np.array(list(map(classes.index, vs))) for vs in data_items['y']]
        else:
            data_items['y'] = np.array(list(map(classes.index, data_items['y'])))
    return {**data_items, **other_items}


# return acramos dataset as X, y. Returns only non-NA 'label_subset' examples (=rows).
def load_acramos(dir, n_fft=1024, stride=None, n_mels=512, n_mfcc=512, tf_trans="Spectrogram",
                 trimlr_secs=(0, 0), split=0, crop_n=0, aggregate=np.mean, return_X_y=False, feat_cache: str = '',
                 spectral_pool=0, multiclass=True, label_subset=None):
    config = locals().copy()
    sr = 32000
    dir = Path(dir)
    stride = stride or n_fft // 2
    # tf_trans = "MFCC"
    if feat_cache:
        feat_cache = Path(feat_cache).with_suffix(".npz")

    labels = pd.read_csv(dir / "adsim_label.csv")
    if "Label" in labels:
        # legacy data format
        # sort labels by length: a hack that returns the negative class as 0
        target_names = sorted(np.unique(labels["Label"]), key=len)
        target_names = {v.capitalize() for s in target_names for v in re.split(r'[^a-zA-Z0-9_]+', s)}
        target_names = sorted(target_names - {'', 'Negativ'}, key=len)
        if label_subset is None:
            label_subset = target_names
        assert set(label_subset) <= set(target_names), f"'label_subset' is no subset of labels"
        y = [tuple(float(v in s) for v in label_subset) for s in labels['Label']]
        target_names = label_subset
        if multiclass:
            # legacy: reverse - results that class integer encoding corresponds to target_names.index
            l = [v[::-1] for v in product([0., 1.], repeat=len(label_subset))]
            y = [l.index(k) for k in y]
            l = {tuple(k): ",".join([s for s, v in zip(label_subset, k) if v]) for k in l}
            target_names = ["Negativ", *[*l.values()][1:]]
        labels["y"] = y
        train = labels[labels["campaign"] != "S7"]
        labels['Wav'] = labels['campaign'] + '/' + labels['ID'] + "_01.wav"
    else:
        i = 1 + list(labels.columns).index('Wav')
        target_names = list(labels.columns)[i:]
        if label_subset is None:
            label_subset = target_names
        # targets as list of tuples
        y = labels[label_subset]
        # filter NA rows
        y = y[y.notna().all(axis=1)]
        y = pd.Series(y.itertuples(index=False, name=None), y.index)
        if multiclass:
            # convert labels to single multiclass label
            # legacy: reverse - results that class integer encoding corresponds to target_names.index
            l = [v[::-1] for v in product([0., 1.], repeat=len(label_subset))]
            y = y.map(l.index)
            l = {tuple(k): ",".join([s for s, v in zip(label_subset, k) if v]) for k in l}
            target_names = ["Negativ", *[*l.values()][1:]]
        # use inner join to avoid float conversion
        labels = labels.join(y.rename('y'), how='inner')
    try:
        config.pop("return_X_y")
        config.pop("feat_cache")
        assert config == np.load(file=feat_cache, allow_pickle=True)['config'][0]
        X = [v.astype(float) for v in np.load(feat_cache, allow_pickle=True)['X']]
        y = list(np.load(feat_cache, allow_pickle=True)['y'])
        files = list(np.load(feat_cache, allow_pickle=True)['files'])
    except (FileNotFoundError, AssertionError) as e:
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
            file = r['Wav']
            files.append(file)
            wave, srw = torchaudio.load(dir / file)
            if srw != sr:
                wave = F.resample(
                    wave,
                    orig_freq=srw,
                    new_freq=sr,
                    resampling_method="sinc_interp_kaiser",
                    # override default Kaiser with "best" settings: greater Nyquist damping
                    # lowpass_filter_width=64,
                    # rolloff=0.9475937167399596,
                    # beta=14.769656459379492,
                )
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
                y.append(np.array([r["y"]] * len(tf3d)))
            else:
                feat = aggregate(tf2d, axis=1)
                # feat = tf2d.max(axis=1)
                feats.append(feat)
                y.append(r["y"])
        X = feats
        # y = np.array(y)
        if feat_cache:
            np.savez(file=feat_cache, config=np.array([config], dtype=object), X=np.array(X, dtype=object),
                     y=np.array(y, dtype=object), files=np.array(files, dtype=object))

    if spectral_pool:
        X = [np.max(np.reshape(v[:, 1:], (len(v), -1, spectral_pool)), axis=-1) for v in X]

    if return_X_y:
        return X, y
    else:
        return dict(X=X, y=y, target_names=target_names, files=files, fs=sr)


# auto label events in a single spectrogram in decibel. The event peak loudness must be contained in "in_band", larger
# than "q_db" and drop to both sides within frequency span "s" by more than "d_db".
def machine_label_spectrogram(X, q_db, d_db, in_band, s=1 / 16, positives=[None, "Quietschen"], use_kernel=False):
    ys = []
    k = "Quietschen"
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
def conv_slice2file(y, split_sizes=None):
    def f(v):
        v = tuple(sorted(set(v)))
        l = 0
        if v[0] == 0:
            v = v[1:]
        if len(v) == 1:
            l = v[0]
        elif len(v) > 1:
            m = {(1, 2): 3}
            l = m[v]
        return l

    if split_sizes:
        y = unchain(y, ls=split_sizes)
    if hasattr(y[0], "__len__"):
        return list(map(f, y))
    else:
        return f(y)


# return average precision for all classes including negative classes. No macro averaging.
def ap_class_score(y_true, y_score, labels=None):
    if labels is None:
        labels = np.unique(y_true)
    return [average_precision_score(y_true == l, y_score[:, i]) for i, l in enumerate(labels)]


# wraps classification_report. 'names' is interpreted differently from sklearn 'target_names'. 'names' index must
# correspond to labels in 'y_pred', 'y_true' - different to sklearn classification_report.
def file_classification_report(y_true, y_pred, labels=None, names=None, as_dataframe=False, **kwargs):
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

    if as_dataframe:
        cr = classification_report(y_true, y_pred=y_pred, labels=labels, target_names=target_names,
                                   output_dict=True, **kwargs)
        # harmonize data for columns
        if "accuracy" in cr:
            cr["accuracy"] = {
                "f1-score": cr["accuracy"],
                "support": cr["macro avg"]["support"],
            }
        # reindex rows, since 'accuracy' would otherwise be the last row
        cr = pd.DataFrame.from_dict(cr, orient="index").loc[list(cr)]
        # convert floating pt support
        cr['support'] = cr['support'].astype("Int64")
        # can infer ints if score in {0, 1} cr = pd.DataFrame(cr).T.convert_dtypes()
    else:
        cr = classification_report(y_true, y_pred=y_pred, labels=labels, target_names=target_names, **kwargs)
    return cr


# wraps file classification_report for 'y_true' and 'y_pred' as hierarchical (ragged) sequence of labels
# return 2 CR's, for flat and aggregated labels.
def slice_classification_report(y_true, y_pred, labels=None, names=None, as_dataframe=False, **kwargs):
    # if hasattr(y_true, "__len__") and hasattr(y_pred, "__len__"):
    # file level comparison
    if [len(v) for v in y_true] != [len(v) for v in y_pred]:
        raise ValueError("'y_true' item lengths different to 'y_pred'")
    d = dict(y_true=y_true, y_pred=y_pred)
    d = dict(file=[np.array(conv_slice2file(v)) for k, v in d.items()],
             slice=[np.array([*chain(*v)]) for k, v in d.items()])

    for k, (y_true, y_pred) in d.items():
        d[k] = file_classification_report(y_true, y_pred, labels=labels, names=names, as_dataframe=as_dataframe,
                                          **kwargs)
    return d


# classification report with optional aggregation and dataframe conversion
# return CR or 2 CR's for aggregation
def agg_classification_report(y_true, y_pred, aggregate: Dict[str, callable] = None, labels=None, names=None,
                              as_dataframe=False,
                              **kwargs):
    if aggregate is None:
        aggregate = lambda x: x
    if hasattr(aggregate, "__len__"):
        cr = {}
        for k, f in (aggregate.items() if hasattr(aggregate, "items") else enumerate(aggregate)):
            cr[k] = file_classification_report(f(y_true), y_pred=f(y_pred),
                                               labels=labels, names=names, as_dataframe=as_dataframe, **kwargs)
        if as_dataframe:
            cr = pd.concat(cr)
    else:
        cr = file_classification_report(aggregate(y_true), y_pred=aggregate(y_pred),
                                        labels=labels, names=names, as_dataframe=as_dataframe, **kwargs)
    return cr


# apply 'score_fun' to y. y can be 1D or ragged 2D. A ragged y is indicated by 2D 'y_true' or 'shape' and will be
# aggregated with the callable 'label_agg'.
def score_agg(y_true, y_pred, score_fun, label_agg=None, shape=None, **kwargs):
    if hasattr(y_true[0], "__len__"):
        # flatten ragged array
        shape = list(map(len, y_true))
        y_true = np.concatenate(y_true)
    assert len(y_true) == len(y_pred), "length of ys must match"
    sf = score_fun(y_true, y_pred=y_pred, **kwargs)
    if label_agg is not None and shape is not None:
        y_tagg = label_agg(unchain(y_true, ls=shape))
        y_pagg = label_agg(unchain(y_pred, ls=shape))
        sa = score_fun(y_tagg, y_pred=y_pagg, **kwargs)
        return sf, sa
    return sf,


# Dataframe CM with optional 'margin'.
def confusion_matrix_df(y_true, y_pred, labels=None, target_names=None, **kwargs):
    # cm = confusion_matrix(y_true, y_pred)
    # labels = unique_labels(y_true, y_pred)
    # if names is None:
    #     names = labels
    # cm = pd.DataFrame(cm, index=names, columns=names)
    labels = labels or unique_labels(y_true, y_pred)
    if target_names is not None:
        y_true = np.take(target_names, list(map(list(labels).index, y_true)))
    cm = pd.crosstab(index=pd.Series(y_true, name='y_true'), columns=pd.Series(y_pred, name='y_pred'), **kwargs)
    # order
    if target_names is not None:
        index = list(target_names) + [v for v in cm.index if v not in target_names]
        cm = cm.loc[index, :]
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


# convert waveform amplitude power in iterable 'X' to representation in 'to'. The signal minimum value can be defined
# via the absolute minimum 'signal_floor' and the minimum from 'abs_top_range'. The 'top_db' param applies only for
# 'to' == 'db' and to each item in 'X' individually.
def power_to(X, to: str = "power", abs_top_range=np.inf, top_db=None, signal_floor=-np.inf, normalize_each=False):
    if to == "db":
        X1 = [power_to_db(v, top_db=top_db) for v in X]
        # db normalized
        # X_db = [v - np.max(v) + vrange for v in X_db]
    elif to == "linear_magnitude":
        X1 = [(v ** .5) for v in X]
    elif to == "power":
        # scaled power spectrum
        # X_db = [v/np.max(v) for v in X]
        X1 = X
    else:
        raise ValueError(f"'{to}'  not recognized")
    # use global scale (vs. individual scale)
    vmin, vmax = [(np.min(v), np.max(v)) for v in [np.concatenate(X1)]][0]
    # too high vmin for sqeal file #125, misses sqeal at x = 4  vrange = 40, must limit vmin
    # no label secrets allowed vmax = min(label_kwargs["q_db"] - label_kwargs["d_db"] + vrange, vmax)
    vmin = max(vmax - abs_top_range, vmin, signal_floor)
    X1 = [np.clip(v, a_min=vmin, a_max=vmax) for v in X1]
    if normalize_each:
        X1 = [(v - np.min(v)) / np.max(v) for v in X1]
        vmin, vmax = 0., 1.
    return X1, vmin, vmax


def load_acramos_slices(subset=None, n=None, reindex=False, label_report=False, do_plot_spectra=False
                        , feat_cache: str = '', **kwargs):
    # parameter configuration
    config = kwargs
    if feat_cache:
        feat_cache = Path(feat_cache).with_suffix(".npz")
    try:
        ds = dict(np.load(feat_cache, allow_pickle=True))
        assert config == ds.pop('config')[0]

        ds = {k: list(v) for k, v in ds.items()}
        ds['X'] = [v.astype(float) for v in ds['X']]
        ds['fs'] = ds['fs'][0]
        # ds['non-existant']
    except (FileNotFoundError, KeyError, AssertionError) as e:
        # select 'classes' from dataset
        ds = load_acramos(**kwargs)

        print("auto labelling ...")
        ds['y_auto'] = machine_label_dataset(X_db=[power_to_db(v, top_db=None) for v in ds['X']], fs=ds['fs'])

        if feat_cache:
            np.savez(file=feat_cache, config=np.array([config], dtype=object), X=np.array(ds['X'], dtype=object),
                     y=np.array(ds['y'], dtype=object), target_names=np.array(ds["target_names"], dtype=object),
                     y_auto=np.array(ds['y_auto'], dtype=object), files=np.array(ds["files"], dtype=object),
                     fs=np.array([ds["fs"]], dtype=object))

    ds = dataset_class_subset(ds, subset_labels=subset, n_max=n, reindex=False)
    X = ds['X']
    y = ds['y']
    y_auto = ds['y_auto']
    files = ds['files']
    target_names = ds['target_names']

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
        # subset = ["Quietschen"]
        subset = ["Negativ", "Quietschen"]
        t_2 = t_2_man
        if not n:
            n = len(t_2)
        # n = 32
        assert 0 < n <= len(t_2)
        t_2 = t_2[:n]
        if "Negativ" in subset:
            # all files must be empty
            t_2 = [[]] * n + t_2
        print(f"Slice level comparing {len(t_2)} files of auto-labelling to manually labelled 'Quietschen' slices ")
    else:
        t_2 = None

    if compare_manual:
        assert len(t_2) == len(y)
        # transform file label to slice (=event) label
        # put label at positions in 't_2', rest is zero
        y_slice = [np.bincount(ixs, minlength=len(k)) * k[0] for ixs, k in zip(t_2, y)]

        cr = slice_classification_report(y_slice, y_pred=y_auto,
                                         labels=np.arange(len(target_names)),
                                         target_names=target_names)
        print(cr['slice'])
        # label differences with auto label
        er = slice_error_report(y_slice, y_pred=y_auto)
        if len(er):
            print(f"{len(er)} files with slice errors")
            print(er)
            print()

    if label_report or compare_manual:
        # file level comparison

        print("confusion matrix")
        print(confusion_matrix(conv_slice2file(y), conv_slice2file(y_auto)))
        print(f"File label from aggregated slice auto-label vs. acramos file label " +
              f"{'' if compare_manual else '(no slice label available)'}")
        cr = slice_classification_report(y, y_pred=y_auto, zero_division=0,
                                         labels=np.arange(len(target_names)),
                                         target_names=target_names)
        print(cr['file'])
        er = slice_error_report(y, y_auto)
        er = er.loc[er['error'], ['y_pred', 'y_true']]
        if len(er):
            print(f"{len(er)} auto-label vs. acramos file errors")
            print(er)

    # change classes to range(n_classes)
    if reindex:
        c = np.unique([*chain(*chain(y, y_auto))])
        # [list(map(list(c).index, v)) for v in y]
        y = mapr(list(c).index, y)
        # y_auto = reindex(y_auto)
        y_auto = mapr(list(c).index, y_auto)
        # negative class needed for slice level operation
        target_names = [target_names[k] for k in c]

    # ds.pop('y')
    ds = {'X': X, 'y': y_auto, 'y_file': conv_slice2file(y_auto), "target_names": target_names,
          # the acramos file label from the database
          "files": files,
          'y_acramos': [v[0] for v in y]}
    return ds


def train_test_split_dataset(ds, test_size=0.2, random_state=42, **kwargs):
    # n = max([len(v) if hasattr(v, "__len__") else 1 for v in ds.values()])
    n = len(ds['X'])
    data_items = {k: v for k, v in ds.items() if hasattr(v, "__len__") and len(v) == n}
    other_items = {k: ds[k] for k in set(ds) - set(data_items)}

    # X, y, t_2, X_ho1, y_ho1, t_2_ho1 = X[-300:], y[-300:], t_2[-300:], X[:-300], y[:-300], t_2[:-300]
    split = train_test_split(*data_items.values(), test_size=test_size, random_state=random_state, **kwargs)
    split = list(zip(*np.reshape(np.array(split, dtype=object), (-1, 2))))
    train, test = [{k: v for k, v in zip(data_items, vs)} for vs in split]

    return {**train, **other_items}, {**test, **other_items}


# convert dict of arbitrary number datasets (must contain keys 'y', and 'target_names') to dataframe
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
        if len(ds) > 1:
            prev = {k: pd.concat((v, v.agg(["sum"]))) for k, v in prev.items()}
        prev = pd.concat(prev, axis=0).T
        prev = pd.concat((prev, prev.agg(["sum"])))
    else:
        prev = pd.concat(prev, axis=0).T
    return prev


# short wrapper for model fit, prediction and metrics.
def model_fit_evaluate(mdl, X, y, X_ho, y_ho, names, n_dimred=0, print_cr=False):
    if n_dimred:
        assert 0 < n_dimred < X.shape[1]
        # for reproducible result use svd_solver='full' or set random state
        dimred = decomposition.PCA(n_components=n_dimred, svd_solver='full').fit(X)
        X = dimred.transform(X)
        X_ho = dimred.transform(X_ho)

    print(str(mdl))
    mdl.fit(X, y)
    yh = mdl.predict(X_ho)

    if print_cr:
        print("Holdout test")
        cr = file_classification_report(y_ho, y_pred=yh, names=names)
        cm = pd.crosstab(pd.Series(names[y_ho]), names[yh])
        print(f"{'file'} level")
        print(cr)
    return yh


def test_third_octave_bands():
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


def test_adsim_features():
    # (multiple) labels to load
    label_subset = [
        "Kreischen",
        "Quietschen",
    ]

    ds = load_acramos(multiclass=True, label_subset=label_subset,
                      # n=200,
                      dir=test_dir, split=1.0, aggregate=np.mean, spectral_pool=0,
                      # feat_cache="clustering")
                      feat_cache=Path(test_dir) / Path(__file__).stem
                      )
    ds = dataset_class_subset(ds, subset_labels=subset, reindex=True)
    names = np.array(ds['target_names'])
    # aggregate time axis for each file
    # aggregate = partial(np.quantile, q=1)
    ds['y'] = [v[0] for v in ds['y']]
    # ds['X'] = [power_to_db(v, top_db=None) for v in ds['X']]
    ds['X'], *_ = power_to(ds['X'], to="db", abs_top_range=abs_top_range)
    ds['X'] = np.array([aggregate_feats(v, axis=0) for v in ds['X']])
    print("")
    print("Train holdout split")
    ds_trn, ds_ho = train_test_split_dataset(ds, stratify=ds['y'])

    # build DataFrame for display
    prev = prevalence(dict(train=ds_trn, holdout=ds_ho), margin=True)
    # prev = prev.rename(columns=dict(file="slice"))
    # prev.columns = [('slice', b) for a, b in prev.columns]
    print(prev.to_string(col_space=([*[7] * (len(prev.columns) // 2), 15] * 2)[:len(prev.columns)]))

    X, y = ds_trn['X'], ds_trn['y']
    X_ho, y_ho = ds_ho['X'], ds_ho['y']

    mdl = discriminant_analysis.LinearDiscriminantAnalysis()
    # fit model on train data
    model_fit_evaluate(mdl, X=X, y=y, X_ho=X_ho, y_ho=y_ho, names=names, print_cr=True, n_dimred=0)
    return


def test_slice_processing():
    reindex = True
    use_dimred = False
    do_plot_spectra = False
    ds = load_acramos_slices(subset=subset, multiclass=True, label_subset=["Kreischen", "Quietschen"],
                             # n=200,
                             reindex=reindex, label_report=True,
                             dir=test_dir, split=1.0, aggregate=np.mean, spectral_pool=0,
                             # feat_cache="clustering")
                             feat_cache=Path(test_dir) / f"{Path(__file__).stem}_slices"
                             )
    # labels = np.unique([*chain(*ds['y'])])
    names = np.array(ds['target_names'])

    # different amplitude representations
    X_db = [power_to_db(v, top_db=None) for v in ds['X']]
    # X_db = [power_to_db(v, top_db=vrange) for v in ds['X']]
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
    vmin = vmax - abs_top_range
    X_db = [np.clip(v, a_min=vmin, a_max=vmax) for v in X_db]
    ds['X'] = X_db
    if do_plot_spectra:
        # vmin, vmax = 0, 40
        y_files = [v[0] for v in ds['y']]
        for i, v in enumerate(X_db):
            title = f"{ds['target_names'][ds['y'][i][0]]} {ds['files'][i]}"
            plot_spectrogram(v.T,
                             Fs=ds['fs'], n_fft=1024, stride=32e3, title=title,
                             scale="linear", cmap="jet",  # vrange=vrange,
                             # vmin=vmin, vmax=vmax,
                             )
            # plt.ylim([0, 1e4])

    print("")
    print("Train holdout split")
    ds_trn, ds_ho = train_test_split_dataset(ds)
    # build DataFrame for display
    prev = prevalence(dict(train=ds_trn, holdout=ds_ho), margin=True)
    # prev = prev.rename(columns=dict(file="slice"))
    # prev.columns = [('slice', b) for a, b in prev.columns]
    print(prev.to_string(col_space=([*[7] * (len(prev.columns) // 2), 15] * 2)[:len(prev.columns)]))

    # flatten nested files
    X, y = [[*chain(*ds_trn[k])] for k in ['X', 'y']]
    X_ho, y_ho = [[*chain(*ds_ho[k])] for k in ['X', 'y']]

    print()
    print("Train on slices")
    # fit model on train data
    mdl = discriminant_analysis.LinearDiscriminantAnalysis(priors=None)
    # mdl = RandomForestClassifier()
    if use_dimred:
        # for reproducible result use svd_solver='full' or set random state
        dimred = decomposition.PCA(n_components=64, svd_solver='full').fit(X)
        X = dimred.transform(X)
        X_ho = dimred.transform(X_ho)
    print(str(mdl))
    mdl.fit(X,
            # dimred.transform(X),
            y=y)
    print("Holdout test")
    yh = mdl.predict(X_ho)
    yh = unchain(yh, ls=[len(v) for v in ds_ho['y']])
    if "use_dataframe":
        # reshape 1d list to ragged file structure
        # cr = pd.concat(slice_classification_report(ds_ho['y'], y_pred=yh, names=names, as_dataframe=True), axis=1)
        aggregate = dict(file=partial(conv_slice2file, split_sizes=[len(v) for v in ds_ho['y']]), slice=lambda x: x)
        cr = agg_classification_report(y_ho, y_pred=[*chain(*yh)], names=names, as_dataframe=True, aggregate=aggregate)
        cr = pd.concat({k: cr.loc[k] for k in cr.index.unique(0)}, axis=1)
        print(cr.to_string(col_space=[9, 9, 9, 9, 20, 9, 9, 9],
                           float_format='{:0.2f}'.format, na_rep='').replace("<NA>", "    "))
        print()
    else:
        # don't use dataframe
        cr = slice_classification_report(ds_ho['y'], y_pred=yh, names=names)
        [(print(f"{k} level"), print(v)) for k, v in cr.items()]
    er = slice_error_report(ds_ho['y'], y_pred=yh, )

    print("Holdout test against file level original acramos label")
    # -not- the ground truth used for training
    cr = file_classification_report(ds_ho['y_acramos'], y_pred=conv_slice2file(yh), names=names, as_dataframe=True,
                                    zero_division=0)
    # "display.precision", 2
    # with pd.option_context('display.float_format', '{:0.2f}'.format):
    print(cr.to_string(col_space=9, float_format='{:0.2f}'.format, na_rep='').replace("<NA>", "    "))
    print()

    print("Train on file level: aggregated from slices")
    # aggregate time axis
    # X, X_ho = [np.array([aggregate_feats(v, axis=0) for v in vs]) for vs in [ds_trn['X'], ds_ho['X']]]
    X = np.array([aggregate_feats(v, axis=0) for v in unchain(X, ls=list(map(len, ds_trn['X'])))])
    X_ho = np.array([aggregate_feats(v, axis=0) for v in unchain(X_ho, ls=list(map(len, ds_ho['X'])))])
    y = ds_trn['y_file']
    y_ho = ds_ho['y_file']

    # fit model on train data
    mdl = discriminant_analysis.LinearDiscriminantAnalysis(priors=None)
    model_fit_evaluate(mdl, X=X, y=y, X_ho=X_ho, y_ho=y_ho, names=names, print_cr=True)

    print("Train on file level: original acramos label")
    y = ds_trn['y_acramos']
    y_ho = ds_ho['y_acramos']

    # fit model on train data
    mdl = discriminant_analysis.LinearDiscriminantAnalysis(priors=None)
    model_fit_evaluate(mdl, X=X, y=y, X_ho=X_ho, y_ho=y_ho, names=names, print_cr=True)


def test_metrics():
    from sklearn.metrics import multilabel_confusion_matrix
    y_true = np.array([[1, 0, 1],
                       [0, 1, 0]])
    y_pred = np.array([[1, 0, 0],
                       [0, 1, 1]])
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred)
    return


if __name__ == "__main__":
    # feature aggregator over time axis in STFT representation
    # aggregate_feats = np.mean
    aggregate_feats = partial(np.quantile, q=1)
    # feature value range
    abs_top_range = 40
    # class label subset
    subset = [
        "Negativ",
        "Kreischen",
        "Quietschen",  # for label_subset ['Quietschen'] only, this is the union of Quietschen and Kreischen,Quietschen
        "Kreischen,Quietschen",
    ]

    test_dir = Path.home() / 'prj' / 'acrDb'
    #test_dir = r"D:\ADSIM\Import-2023-04"

    # ##################################################################################################################
    # Test third octave bands ##########################################################################################
    # test_third_octave_bands()

    test_metrics()

    test_adsim_features()

    # ##################################################################################################################
    # ##################################################################################################################
    test_slice_processing()

    pass
