# legacy code, file renamed to *_V1 to differentiate from newer code (V2)
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
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.neighbors import kneighbors_graph
from functools import partial
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import pickle

from features import acr_dirs, load_acramos
from TrainSamples import plot_spectrogram


def robust_max(a, **kwargs):
    return np.quantile(a, q=0.95, **kwargs)


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    acr_dir = acr_dirs['V1']
    use_iris = False
    class_weight = None
    reps = 3
    plot_scores = False
    #class_weight = 'balanced'
    scores = []
    qs = ["mean", "ravel", .5, .7, .9, 0.95, 0.98, .99, .995, .999, 1]
    trimlr_secss = [(0, 0), (2, 0), (0, 2), (2, 2)]
    experiments = list(product(trimlr_secss[:1], 2**np.arange(10, 11), [qs[-1]], [0, .1, .2, .5, 1, 2][:],
                               [125, 250, 125*5, None][-1:]))
    print(f"running {len(experiments)} experiments in {len(experiments[0])} hyperparameters")
    for trimlr_secs, n_fft, q, split, crop_n in experiments:
        if use_iris is False:
            print("dataset is acramos")
            fs = 32e3
            #n_fft = 512
            size = None
            stride = n_fft*800//1024
            #stride = n_fft // 2
            tf_trans = "Spectrogram"
            n_mels = 512
            n_mfcc = 512
            #tf_trans = "MFCC"

            if q == "mean":
                aggregate = np.mean
            elif q == "ravel":
                aggregate = lambda x, axis: np.ravel(x)
            else:
                aggregate = partial(np.quantile, q=q)
            if split == "min":
                split = stride/fs
            ds = load_acramos(acr_dir, n_fft=n_fft, stride=stride, tf_trans=tf_trans,
                              trimlr_secs=trimlr_secs, crop_n=crop_n, aggregate=aggregate, split=split,
                              feat_cache=acr_dir / Path(__file__).stem,
                              )
            X, y, label = [ds[k] for k in "X, y, target_names".split(", ")]
            #X = [power_to_db(v, top_db=None) for v in X]
            #X = [(v ** .5) / np.max(v ** .5) for v in X]
            #X = [(v ** .5)  for v in X]
            vmin, vmax = [(np.min(v), np.max(v)) for v in [np.concatenate(X)]][0]
            #vmin = vmax - 40
            X = [np.clip(v, a_min=vmin, a_max=vmax) for v in X]
            if split:
                # aggregate time axis
                X = np.array([aggregate(v, axis=0) for v in X])
                y = np.array([v[0] for v in y])
        else:
            print("dataset is IRIS")
            X, y = datasets.load_iris(return_X_y=True)
            label = datasets.load_iris().target_names

        Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
        #w = len(ytrn) / (4 * np.bincount(ytrn))
        if class_weight is None:
            wc = np.ones(len(label))
        elif class_weight.startswith('balanced'):
            wc = compute_class_weight(class_weight='balanced', classes=np.unique(ytrn), y=ytrn)
            #assert all(wc[ytrn] == compute_sample_weight(class_weight='balanced', y=ytrn))
        else:
            raise ValueError
        with np.printoptions(precision=2):
            print(f"class weights: {wc}")
        ws = wc[ytrn]
        res = []
        for rep in 1 + np.arange(reps):
            # mdl = discriminant_analysis.LinearDiscriminantAnalysis().fit(Xtrn, ytrn)
            # name = "LDA"
            # mdl = discriminant_analysis.QuadraticDiscriminantAnalysis().fit(Xtrn, ytrn)
            # name = "QDA"
            # mdl = naive_bayes.MultinomialNB().fit(Xtrn, ytrn)
            # name = "MultinomialNB"
            # mdl = svm.LinearSVC(dual=False,  # n_samples > n_features
            #                     class_weight='balanced',
            #                     C=1e9,
            #                     #verbose=1,
            #                     max_iter=5000).fit(Xtrn, ytrn)  # failed to converge
            # name = "SVM")
            # mdl = tree.DecisionTreeClassifier().fit(Xtrn, ytrn)
            # name = "Decision tree"
            mdl = ensemble.RandomForestClassifier(n_estimators=100, random_state=rep, class_weight=class_weight).fit(Xtrn, ytrn)
            name = "Random Forest"
            # mdl = ensemble.ExtraTreesClassifier(n_estimators=100, class_weight=class_weight).fit(Xtrn, ytrn)
            # name = ("Extra Trees")
            # mdl = ensemble.AdaBoostClassifier(n_estimators=100).fit(Xtrn, ytrn, sample_weight=ws)
            # name = ("Ada Boost w. Decision Tree")
            # mdl = ensemble.GradientBoostingClassifier(n_estimators=100, ).fit(Xtrn, ytrn, sample_weight=ws)
            # name = ("Gradient Boosting")
            # mdl = ensemble.HistGradientBoostingClassifier().fit(Xtrn, ytrn, sample_weight=ws)
            # name = ("Hist Gradient Boosting")

            print(name)
            yh = mdl.predict(Xtst).astype(int)
            cr = classification_report(ytst, y_pred=yh, labels=np.arange(len(label)), output_dict=True)
            res.append(dict(accuracy=cr["accuracy"], **cr["macro avg"]))
            print(classification_report(ytst,
                                        y_pred=yh,
                                        labels=np.arange(len(label)),
                                        target_names=label,
                                        output_dict=False,
                                        digits=2))
        df = pd.DataFrame(res).drop(columns=["support"])
        scores.append(dict(classifier=name, trimlr_secs=trimlr_secs, q=q, n_fft=n_fft, stride=stride, split=split,
                           crop_n=crop_n,
                           reps=reps, support=len(ytst), **df.mean()))
        if plot_scores:
            df.melt().plot(kind="scatter", x="variable", y="value")
            plt.grid(axis='y')
            #plt.title()
    pd_opts = ('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.precision', 3,
               )
    with pd.option_context(*pd_opts):  # more options can be specified also
        print(pd.DataFrame(scores).to_string(index=False))
    plt.show()
    print("done")
