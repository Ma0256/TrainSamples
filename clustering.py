from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torchaudio
from librosa import power_to_db
from sklearn import cluster, datasets, mixture, decomposition, discriminant_analysis, naive_bayes, svm, tree, ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, adjusted_mutual_info_score, mutual_info_score, \
    silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import pickle
from itertools import chain
from functools import partial
from textwrap import indent

from convert_labels import acr_dir
from features import load_acramos_slices, train_test_split_dataset, prevalence, unchain, file_classification_report, \
    slice_classification_report, agg_classification_report, conv_slice2file
from TrainSamples import plot_spectrogram
from TrainSoundExploration import plot_normal_prominent


# quietschen simulation
def sim_squeal(n, d, d_sqeal=None, random_state=None, ):
    rng = np.random.default_rng(seed=random_state)
    d_sqeal = d_sqeal or d
    if hasattr(n, "__len__"):
        npos, nn = n
    else:
        npos, nn = n - n // 2, n // 2
    Xp = np.zeros((npos, d))
    Xp[:, :d_sqeal] = np.tile(np.eye(d_sqeal), (npos, 1))[:len(Xp)]
    Xp += 0.3 * rng.uniform(size=(npos, d))
    # broadcast along features
    Xp *= rng.uniform(size=(npos, 1)) + .5
    # Xn = np.abs(np.random.randn(nn * d, d))
    cov = d ** -1 * (0.8 * np.ones((d, d)) + .2 * np.eye(d))
    Xn = np.abs(
        rng.multivariate_normal(mean=np.zeros(d), cov=cov, size=nn))
    X = np.concatenate((Xn, Xp))
    y = np.repeat([0, 1], (len(Xn), len(Xp)))
    target_names = ["negative", "q"]
    return dict(X=X, y=y, target_names=target_names)


def plot_error_spectrograms(X, y, yh, len_files, label=None):
    if len_files:
        split = lambda x: np.split(x, np.cumsum(len_files)[:-1])
        yh = split(yh)
        y = split(y)
        X = split(X)
    label = label or sorted({v for vv in y for v in vv})
    errs = [i for i in range(len(y)) if any(y[i] != yh[i])]
    yf = [any(v) * 1 for v in y]
    yhf = [any(v) * 1 for v in yh]
    err_file = [i for i in range(len(y)) if yf[i] != yhf[i]]
    for i in errs:
        t = np.where(y[i] != yh[i])[0]
        title = f"file#{i} {'predicted ' + str(label[yhf[i]]) + ' vs.' if i in err_file else ''} label {label[yf[i]]}:"
        title += f"\nerror at {t}, yh {yh[i][t]} vs. label {y[i][t]}"
        plot_spectrogram(X[i].T,
                         Fs=fs, n_fft=1024, stride=32e3, title=title,
                         scale="linear", cmap="jet",  # vrange=vrange,
                         # vmin=vmin, vmax=vmax,
                         )
        [plt.axvline(v + .5, color='r') for v in t]
    return


# plot ragged sequence of recordings of slices. Markers in 'y' highlight specific slices.
def plot_recording_trajectories(Xr: list, y=(), record_label=(), ax=None, projection: str = None):
    ax = ax or plt.figure().add_subplot(111, projection=projection)
    record_label = record_label if any(record_label) else (None,) * len(Xr)
    assert all(np.array([len(v) for v in [y, record_label] if len(v) > 0]) == len(Xr)), "Parameter lengths don't match"
    if projection == '3d':
        Xr = [v[:, :3] for v in Xr]
        pass
        # for i, v in enumerate(Xr_files):
        #     l = ax.plot(v[:, 0], v[:, 1], v[:, 2], label=files[i], marker='.', alpha=0.3, linestyle='dashed',
        #                 linewidth=0.5)
        #     # select start and end
        #     v = v[[0, -1], :]
        #     ax.scatter(v[0, 0], v[0, 1], v[0, 2], color=l[0].get_color(), marker=r'*')
        #     ax.scatter(v[-1, 0], v[-1, 1], v[-1, 2], color=l[0].get_color(), marker=r'+')
    else:
        Xr = [v[:, :2] for v in Xr]
    for i, v in enumerate(Xr):
        l = ax.plot(*v.T, label=record_label[i], marker='.', alpha=0.6,  # linestyle='dashed',
                    linewidth=0.5)
        # plot event markers
        markersize = 10
        zorder = l[0].zorder
        lab = dict(label="event over background") if i == len(Xr) - 1 else dict(label="")
        if y:
            ev = ax.scatter(*v[y[i] > 0].T, marker='o', color='k', facecolor='none', s=markersize ** 2, zorder=zorder,
                            **lab)
        # plot start and end prominent
        v = v[[0, -1], :]
        ax.scatter(*v[0], color=l[0].get_color(), marker=r'*', s=markersize ** 2, zorder=zorder)
        ax.scatter(*v[1], color=l[0].get_color(), marker=r'+', s=markersize ** 2, zorder=zorder)
    ax.set_title("recording trajectories")
    #ax.legend()
    return


# link 3d axis view
def link_elev_azim(fig, axs):
    def on_move(event):
        if event.inaxes in axs:
            i = axs.index(event.inaxes)
            if axs[i].button_pressed in axs[i]._rotate_btn:
                [axs[j].view_init(elev=axs[i].elev, azim=axs[i].azim) for j in range(len(axs)) if j != i]
            fig.canvas.draw_idle()
        return
        # if event.inaxes == axs[0]:
        #     if axs[0].button_pressed in axs[0]._rotate_btn:
        #         axs[1].view_init(elev=axs[0].elev, azim=axs[0].azim)
        #     # elif axs[0].button_pressed in axs[0]._zoom_btn:
        #     #     axs[1].set_xlim3d(axs[0].get_xlim3d())
        #     #     axs[1].set_ylim3d(axs[0].get_ylim3d())
        #     #     axs[1].set_zlim3d(axs[0].get_zlim3d())
        # elif event.inaxes == axs[1]:
        #     if axs[1].button_pressed in axs[1]._rotate_btn:
        #         axs[0].view_init(elev=axs[1].elev, azim=axs[1].azim)
        #     # elif axs[1].button_pressed in axs[1]._zoom_btn:
        #     #     axs[0].set_xlim3d(axs[1].get_xlim3d())
        #     #     axs[0].set_ylim3d(axs[1].get_ylim3d())
        #     #     axs[0].set_zlim3d(axs[1].get_zlim3d())
        # else:
        #     return
        # fig.canvas.draw_idle()

    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)
    return c1


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    pd.options.display.float_format = '{:,.2f}'.format
    cluster_methods = ["kmeans", "ac ward", "ac single", "ac average"]
    cluster_method = cluster_methods[0]
    data_selection = ["iris", "acramos", "simulation"][1]
    holdout = True
    relabel_clusters = False
    do_plot_spectra = False
    do_plot_training_errs_spectrograms = False
    projection = None
    projection = '3d'
    spectral_pool = 0  # kernel for spectral pooling

    print(f"Dataset is {repr(data_selection.upper())}")
    len_files_ho = len_files = None
    if data_selection == "acramos":
        # use uniform subset
        subset = ["[Negativ]", "[Quietschen]"]
        subset = ["[Quietschen]"]
        # subset = ["[Kreischen]", "[Quietschen]"]
        # classes = [label.index("[Kreischen]"), label.index("[Quietschen]"), label.index("[Kreischen][Quietschen]")]
        # classes = [label.index("[Kreischen]"), label.index("[Kreischen][Quietschen]")]
        # classes = [label.index("[Kreischen][Quietschen]")]
        reindex = True
        ds = load_acramos_slices(subset=subset, n=100,
                                 reindex=reindex, label_report=True,
                                 dir=acr_dir, split=1.0, aggregate=lambda x: x, spectral_pool=0,
                                 # feat_cache="clustering")
                                 feat_cache=Path(__file__).stem)
        # labels = np.unique([*chain(*ds['y'])])
        label = np.array(ds['target_names'])

        print("")
        if holdout:
            ds_trn, ds_ho = train_test_split_dataset(ds)
            len_files_ho = list(map(len, ds_ho['y']))
            X_ho, y_ho = [[*chain(*ds_ho[k])] for k in ['X', 'y']]
        else:
            ds_trn = ds
        len_files = list(map(len, ds_trn['y']))
        files = ds_trn['files']
        # flatten nested files
        X, y = [np.array([*chain(*ds_trn[k])]) for k in ['X', 'y']]

        # # build DataFrame for display
        # prev = prevalence(dict(train=ds_trn, holdout=ds_ho), margin=True)
        # print(prev.to_string(col_space=([*[7] * (len(prev.columns) // 2), 15] * 2)[:len(prev.columns)]))

    elif data_selection == "iris":
        X, y = datasets.load_iris(return_X_y=True)
        label = datasets.load_iris().target_names
        if holdout:
            X, X_ho, y, y_ho = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    elif data_selection == "simulation":
        # simulation of sqeal events, which have dirac-like spectral shape
        # frequency stair
        d = 4
        d_sqeal = 3  # d - d//2
        n = 2000  # acramos has 1665 + 142 files of varying length
        # r = 142/(1665 + 142)
        r = 0.5
        ds = sim_squeal(n=(int(r * n), int((1 - r) * n),), d=d, d_sqeal=d_sqeal, random_state=0)
        X, y, label = [ds[k] for k in "X, y, target_names".split(", ")]
        # show first 3 dimensions of generated data
        ax = plt.figure().add_subplot(111, projection='3d')
        [ax.scatter(*X[y == k, :3].T, marker='o', s=10 ** 2, label=label[k])
         for k in np.unique(y)]
        plt.title(f"First 3 dims of {'x'.join(map(str, X.shape))} generated data")

        # don't run under 'holdout' test
        print("Check linear separability, via linear vs. non-linear classification")
        Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
        s = {}
        for k, mdl in dict(LDA=discriminant_analysis.LinearDiscriminantAnalysis(priors=None),
                           RF=ensemble.RandomForestClassifier()).items():
            mdl = mdl.fit(Xtrn, ytrn)
            s[k] = {k: file_classification_report(v[1], y_pred=mdl.predict(v[0]), names=label, as_dataframe=True)
                    for k, v in dict(test=(Xtst, ytst), train=(Xtrn, ytrn), ).items()}

        # stack experiments (classification reports) vertically
        s = pd.concat({k: pd.concat(v, axis=0) for k, v in s.items()}, axis=0)
        # bring inner level front for simple hierarchical selection
        print(indent(s.reorder_levels([2, 0, 1]).loc["macro avg"].to_string(float_format=None), ' ' * 4))
        print()
        if holdout:
            X, X_ho, y, y_ho = Xtrn, Xtst, ytrn, ytst
    else:
        raise ValueError

    print("Train / holdout split")
    prev = dict(train=dict(y=y, target_names=label))
    if holdout:
        prev.update(holdout=dict(y=y_ho, target_names=label))
    prev = prevalence(prev, margin=True)
    # don't display multi index, since at this point the data could be either slice or file
    prev.columns = [v[-1] for v in prev.columns]
    print(prev.to_string(col_space=[7] * len(prev.columns)), '\n')

    cluster_params = dict(
        n_clusters=max(2, len(np.unique(y))),
        n_neighbors=3,
    )

    if cluster_params["n_clusters"] > 1:
        print(f"Clustering validation")
        # DB validation
        # https://stats.stackexchange.com/questions/448988/which-metrics-are-suitable-for-density-based-clustering-validation
        if cluster_method.startswith("ac"):
            linkage = cluster_method.split(" ")[1]
            # connectivity matrix for structured Ward
            connectivity = kneighbors_graph(X, n_neighbors=cluster_params["n_neighbors"], include_self=False)
            ac = cluster.AgglomerativeClustering(n_clusters=cluster_params["n_clusters"],
                                                 linkage=linkage,
                                                 affinity="euclidean",
                                                 connectivity=connectivity,
                                                 )
            ac.fit(decomposition.PCA().fit_transform(X))
            y_pred = ac.labels_.astype(int)
        elif cluster_method.startswith("kmeans"):
            km = cluster.KMeans(n_clusters=cluster_params["n_clusters"])
            km.fit(decomposition.PCA().fit_transform(X))
            y_pred = km.labels_.astype(int)
        else:
            raise ValueError("unknown cluster method")
        cs = pd.DataFrame({("external", "ami"): [adjusted_mutual_info_score(y, y_pred)],
                           ("external", "s_y"): [silhouette_score(X, labels=y)] if len(set(y)) > 1 else np.NaN,
                           ("internal", "s_yh"): [silhouette_score(X, labels=y_pred)], },
                          index=[cluster_method])
        # cs.index.name = "clusterer"
        # using plural for multiindex required
        # cs.columns.names = (None, "clusterer")
        print(cs.to_string(col_space=7), "\n")

        # # as side info on AMI: random AMI to check robustness against chance
        # n = 1000
        # ami = np.array([adjusted_mutual_info_score(y, np.random.randint(len(set(y)), size=len(y))) for i in range(n)])
        # mi = np.array([mutual_info_score(y, np.random.randint(len(set(y)), size=len(y))) for i in range(n)])
        # h = round(max(abs(ami)), ndigits=3) + .005
        # plt.hist(mi, range=(-h, h), bins=19)
        # plt.title(f"Random AMI (n = {n} of length {len(y)})")
        # plt.xlabel("adjusted mutual information")
        # plt.ylabel("frequency")
        # plt.grid(which="major", axis="x")
        # plt.gca().hist(ami, range=(-h, h), bins=19)
        # plt.legend(["MI", "AMI"])

        print("Contingency matrix")
        cm = contingency_matrix(y, labels_pred=y_pred)  # confusion_matrix(y, y_pred=y_pred)
        print(pd.DataFrame(cm, index=pd.Series(label),  # name="true"),
                           columns=pd.Series(np.unique(y_pred), name=r"true\clusters")))
        print()
        if relabel_clusters:
            raise NotImplementedError
            # process in order of the largest recalls first
            rcm = cm / cm.sum(axis=1)[:, None]
            # index of most likely true class in order of pred class
            i_y = np.argmax(rcm, axis=0)
            # ... and values
            v_y = np.array([rcm[i_y[i], i] for i in range(len(i_y))])
            # map to true class
            y_pred = i_y[y_pred]
            print("reassigned cluster labels")
            cm = confusion_matrix(y, y_pred=y_pred)
            print(pd.DataFrame(cm, index=label, columns=np.unique(y_pred)))
            print(classification_report(y.astype(int), y_pred.astype(int), labels=np.arange(len(label)),
                                        target_names=label))

    # PCA
    n_components = 3
    pca = decomposition.PCA(n_components=n_components).fit(X)
    lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=min(n_components, len(np.unique(y)) - 1)).fit(X,
                                                                                                                      y)
    Xp = pca.transform(X)
    Xl = lda.transform(X)
    if Xl.shape[1] < 3:
        if True:
            # complete orthonormal basis
            scalings = np.linalg.svd(lda.scalings_)[0][:, lda.scalings_.shape[1]:]
            #scalings = scalings[:, np.argsort(np.max(scalings, axis=0))[10:-1:50]]
            assert np.allclose(lda.scalings_.T@scalings, 0), "orthonormal base extension failed"
            scalings = np.c_[lda.scalings_/np.linalg.norm(lda.scalings_), scalings]
            X_new = np.dot(X - lda.xbar_, lda.scalings_)
            assert np.allclose(Xl, X_new)
            Xl = np.dot(X - lda.xbar_, scalings[:, :3])
        else:
            # random scatter for one missing dim
            Xl = np.concatenate((Xl, np.random.rand(len(Xl), 1), np.zeros((len(Xl), 2 - Xl.shape[1]))), axis=1)
    Xr = dict(PCA=Xp, LDA=Xl)
    if holdout:
        print("Holdout test")
        s = {}
        for k, mdl in dict(LDA=discriminant_analysis.LinearDiscriminantAnalysis(priors=None),
                           RF=ensemble.RandomForestClassifier()).items():
            mdl = mdl.fit(X, y)
            for ki, v in dict(test=(X_ho, y_ho, len_files_ho), train=(X, y, len_files), ).items():
                yh = mdl.predict(v[0])
                aggregate = dict(file=lambda x: x)
                if len_files_ho:
                    aggregate = dict(file=partial(conv_slice2file, split_sizes=v[2]), slice=lambda x: x)
                for kii, f in aggregate.items():
                    cr = agg_classification_report(v[1], y_pred=yh, aggregate=f, names=label, as_dataframe=True)
                    s[(k, ki, kii)] = cr
        # vertical join a Dataframe
        s = pd.concat(s, axis=0)
        # split by multiindex level to dict of frames
        splitlevel = lambda x, level, axis: {k: v.droplevel(level, axis) for k, v in x.groupby(level=level, axis=axis)}
        # move innermost index to top column hierarchy for wide format
        s = pd.concat(splitlevel(s, -2, 0), axis=1)
        # select scores
        # s.reorder_levels(np.roll(range(4), 1)).loc["macro avg"]
        # s.loc[(*[slice(None)]*3, 'macro avg')].droplevel(-1)
        # dict(list(s.groupby(level=-1)))['macro avg'].droplevel(-1)
        # s.groupby(level=-1).get_group('macro avg').droplevel(-1)
        # s.loc[s.groupby(level=-1).groups['macro avg']].droplevel(-1)
        # [s.reorder_levels(np.roll(range(3), 1)).loc[k] for k in dict.fromkeys([k[-1] for k in s.index])]
        # s = splitlevel(s, -1, 0)['macro avg']
        df = s.groupby(level=-1).get_group('macro avg').reorder_levels([2, 0, 1])
        print(df.to_string(col_space=([*[8] * 4, 16] * 2)[:s.shape[1]], ))
        print()

        yh = lda.predict(X_ho)
        if do_plot_training_errs_spectrograms:
            plot_error_spectrograms(X_ho, y_ho, yh=yh, len_files=len_files_ho)

    plt.figure()
    [plt.scatter(*Xl[y == k, :2].T, marker='o', s=10 ** 2, alpha=.3, label=label[k]) for k in np.unique(y)]
    yh = lda.predict(X)
    e = np.where(yh != y)[0]
    # yf, yhf = [np.array([any(v) * 1 for v in np.split(vs, np.cumsum(len_files)[:-1])]) for vs in [y, yh, ]]
    # ef = np.where(yhf != yf)[0]
    plt.scatter(*Xl[e, :2].T, s=14 ** 2, color='r', facecolors='none', linewidths=2, label="error")
    plt.legend()
    plt.title(f"LDA projection, n = {len(Xl)}")

    if do_plot_training_errs_spectrograms:
        plot_error_spectrograms(X, y, yh=yh, len_files=len_files)
        # error (file, second, label)
        ixs = [(np.where(v > 0)[0][-1] + 1, v[v > 0][-1], k) for v, k in zip(np.c_[e] - np.cumsum(len_files), y[e])]
        ixs = list(zip(*zip(*ixs), Xl[e, 0]))

    kdr = "LDA"
    Xr = Xr[kdr]
    if True:
        def scatter(X, y, ax, labels=(), **kwargs):
            for i, k in enumerate(np.unique(y)):
                ax.scatter(*X[k == y].T, label=labels[i] if any(labels) else None, **kwargs)

        # set up a figure twice as wide as it is tall
        r, c, fig = 1, 2, plt.figure(figsize=plt.figaspect(0.5))
        axs, ax = [], None
        for i in range(r * c):
            # use z axis only when 3d
            shareax = ["sharex", "sharey", "sharez"][:3 if projection else 2]
            ax = fig.add_subplot(r, c, i + 1, projection=projection, **dict(zip(shareax, [ax]*len(shareax))))
            axs.append(ax)
        if projection:
            link_elev_azim(fig, axs=axs)
            Xr = Xr[:, :3]
        else:
            [v.set_aspect('equal') for v in axs]
            Xr = Xr[:, :2]

        scatter(Xr, y_pred, ax=axs[0], alpha=.3)
        axs[0].set_title(f"{cluster_method} {cluster_params['n_clusters']} clusters {kdr} projection")
        # axs[0].legend()
        scatter(Xr, y, ax=axs[1], labels=label, alpha=.3)
        axs[1].set_title(f"truth {kdr} projection")
        axs[1].legend()
        if len_files:
            # file-wise plot (3D / 2D)
            plot_recording_trajectories(*[unchain(v, len_files) for v in [Xr, y]],
                                        record_label=files, projection=projection)
            #plt.legend()
            #plt.legend([])

    plt.show()

    print("done")
