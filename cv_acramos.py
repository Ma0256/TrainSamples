from pathlib import Path

import pandas as pd
import numpy as np
import os
from sklearn import datasets, mixture, decomposition, discriminant_analysis, naive_bayes, svm, tree, ensemble
from sklearn.preprocessing import label_binarize, MultiLabelBinarizer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, adjusted_mutual_info_score, mutual_info_score, \
    silhouette_score, f1_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pickle
from features import load_acramos, power_to, dataset_class_subset, prevalence, confusion_matrix_df, conv_slice2file, \
    agg_classification_report, file_classification_report, load_acramos_slices, train_test_split_dataset, score_agg, \
    ap_class_score, acr_dirs
from functools import partial
import itertools


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    pd.options.display.float_format = '{:,.2f}'.format

    label_subset = ["Kreischen", "Quietschen"]
    #label_subset = ["Flachstelle"]
    to = "db"
    abs_top_range = 60
    scoring = "f1_macro"
    acramos = "V2"
    use_iris = False
    proc_slices = False
    #proc_slices = True
    agg_slices = False
    agg_slices = True
    to_binary_for = ""
    # to_binary_for = "Kreischen"
    # to_binary_for = "Quietschen"
    split_random_state = 42

    n_cv = 3
    class_weight = None
    # class_weight = 'balanced'
    # function to marginalize time axis
    aggregate_feats = partial(np.quantile, q=1)
    # aggregate_feats = np.mean
    model = "LDA"
    scale = False
    models = dict(
        LDA=discriminant_analysis.LinearDiscriminantAnalysis(),
        # mdl = discriminant_analysis.QuadraticDiscriminantAnalysis()
        # mdl = naive_bayes.MultinomialNB()
        SVC=make_pipeline(StandardScaler(),
                          svm.LinearSVC(dual=False,  # n_samples > n_features
                                        class_weight='balanced',
                                        C=1e-1,
                                        verbose=True,
                                        max_iter=500),
                          ),
        DT=tree.DecisionTreeClassifier(),
        # mdl = ensemble.RandomForestClassifier(n_estimators=100, class_weight=class_weight)
        # mdl = ensemble.RandomForestClassifier(n_estimators=100, class_weight=class_weight, random_state=0)
        RF=ensemble.RandomForestClassifier(n_estimators=100, class_weight=class_weight,
                                           # random_state=None,
                                           # random_state=42,
                                           # random_state=np.random.RandomState(0),
                                           ),
        ETC=ensemble.ExtraTreesClassifier(n_estimators=100, class_weight=class_weight),
        ABC=ensemble.AdaBoostClassifier(n_estimators=100),
        GBC=ensemble.GradientBoostingClassifier(n_estimators=100, ),
        HGBC=ensemble.HistGradientBoostingClassifier(),
    )
    mdl = models["DT"]
    acr_dir = acr_dirs[acramos]
    load_acramos_params = dict(dir=acr_dir, label_subset=label_subset, multiclass=True, split=1.0, )
    subset_labels = None
    reindex_subset = True
    postfix = "".join([v[0].upper() for v in sorted(load_acramos_params['label_subset'])])
    ds = load_acramos(**load_acramos_params,
                      feat_cache=Path(acr_dir) / f"{Path(__file__).stem}_{postfix}")
    if subset_labels is not None:
        ds = dataset_class_subset(ds, subset_labels=subset_labels, reindex=reindex_subset)
    # X, *_ = power_to(X, to="linear_magnitude")
    ds['X'], *_ = power_to(ds['X'], to=to, abs_top_range=abs_top_range)

    target_names = ds['target_names']
    X, y = ds['X'], ds['y']
    # does data have a time axis?
    if hasattr(y[0], "__len__"):
        # no time slice label available -> no supervised slice training -> marginalize time
        X = np.array([aggregate_feats(v, axis=0) for v in X])
        # pick first since rest ist identical
        y = np.array([v[0] for v in y])
        ds.update(X=X, y=y)

    prev = prevalence(dict(data=ds), margin=True)
    # prev = prev.rename(columns=dict(file="slice"))
    # prev.columns = [('slice', b) for a, b in prev.columns]
    print(prev.to_string(col_space=([*[7] * (len(prev.columns) // 2), 15] * 2)[:len(prev.columns)]))

    # split in folds
    # cv = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
    # randomize before split (result here: reduces variance between folds)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    yh = np.array([cross_val_score(mdl.set_params(
        # random_state=i
    ), X, y, cv=cv, scoring=scoring)
        # for i in [None]*n_cv])  # default - non reproducible
        for i in range(n_cv)])  # reproducible model training
    # for i in [0]*3])
    # for i in [1]*3])
    # yh = np.random.random((3, 5))
    groups = 1 + np.arange(yh.shape[1])
    ax = plt.subplots()[1]
    plt.grid(axis='y')
    sns.stripplot(  # data=pd.DataFrame(yh, columns=1 + np.arange(yh.shape[1])).melt(), x="variable", y="value",
        data=pd.DataFrame(yh, columns=groups),
        ax=ax, palette='dark:k', jitter=True)
    [ax.hlines(y, i - .25, i + .25, zorder=2) for i, y in zip(range(yh.shape[1]), yh.mean(axis=0))]
    ax.hlines(yh.mean(), -.25, yh.shape[1] - 1 + .25, 'k',
              linestyles='--', zorder=2)
    # ax.boxplot(yh, positions=groups,
    #            showmeans=True, meanline=True, meanprops={'color': 'k', 'ls': '-', 'lw': 2},
    #            medianprops={'visible': False}, whiskerprops={'visible': False}, showfliers=False, showbox=False, showcaps=False
    #            )
    # plt.ylim([.75, .90])

    plt.show()
    print("done")
