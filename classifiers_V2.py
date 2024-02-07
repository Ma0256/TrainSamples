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


def models_fit_evaluate(mdls: dict, print_score: str = "macro avg", **kwargs):
    s = {k: model_fit_score(mdl, **kwargs) for k, mdl in mdls.items()}
    s = pd.concat(s)

    if print_score:
        print(cr_to_string(s, metric=print_score))
        print()
    return s


# fit 'mdl' and predict on test data. Return classification report and optionally predictions
def model_fit_score(mdl, X, y, X_ho, y_ho, shape=None, shape_ho=None, names=None, target_names=None,
                    n_dimred=0, eval_on_trainset=False, return_yh=False, return_ph=False):
    if hasattr(y[0], "__len__"):
        # flatten ragged array
        shape = list(map(len, y))
        y = np.concatenate(y, axis=0)
        X = np.concatenate(X, axis=0)
        shape_ho = list(map(len, y_ho))
        y_ho = np.concatenate(y_ho, axis=0)
        X_ho = np.concatenate(X_ho, axis=0)

    if n_dimred:
        # reproducibility requires 'full' solver or defined random state
        dimred = decomposition.PCA(n_components=n_dimred, svd_solver='full').fit(X)
        X = dimred.transform(X)
        X_ho = dimred.transform(X_ho)
    pred_data = dict(test=(X_ho, y_ho, shape_ho))
    if eval_on_trainset:
        pred_data.update(train=(X, y, shape))
    mdl = mdl.fit(X, y)
    s = {}
    yhs = {}
    phs = {}
    for ki, v in pred_data.items():
        if return_ph:
            mdl_classes = sorted(set(y))
            try:
                phs[ki] = mdl.predict_proba(v[0])
            except AttributeError as e:
                # does not work with StdScaler phs[ki] = mdl._predict_proba_lr(v[0])
                phs[ki] = mdl.predict(v[0])
                phs[ki] = np.stack([phs[ki] == l for l in mdl_classes]).T.astype(int)
            yh = phs[ki].argmax(axis=1)
            # transform to fitted classes
            yh = np.take(mdl_classes, yh)
        else:
            yh = mdl.predict(v[0])
        yhs[ki] = yh

        if "use_score_agg":
            cr = score_agg(y_true=v[1], y_pred=yh, shape=v[2],
                           score_fun=file_classification_report, as_dataframe=True, target_names=target_names,
                           label_agg=conv_slice2file, labels=None)
            keys = ["flat"] if len(cr) == 1 else ["slice", "file"]
            s.update({(ki, kii): v for kii, v in zip(keys, cr)})
        else:
            # legacy 'agg_classification_report': API more cumbersome
            aggregate = dict(file=lambda x: x)
            labels = None
            # avoid micro avg output labels = sorted({0} | set(conv_slice2file(v[1], v[2])))
            if shape_ho:
                aggregate = dict(file=partial(conv_slice2file, split_sizes=v[2]), slice=lambda x: x)
            for kii, f in aggregate.items():
                # l = sorted(set(f(v[1])))  # report only labels from ground truth (avoid 0.0 f1-score)
                cr = agg_classification_report(v[1], y_pred=yh, aggregate=f,
                                               labels=labels, names=names, as_dataframe=True)
                s[(ki, kii)] = cr

    # split by multiindex level to dict of frames
    # {ko: pd.concat({ki: pd.concat(v, axis=1) for ki, v in ds.items()}) for ko, ds in s2.items()}
    if "split_via_indexing":
        # for case of different indices take the longest and restore it
        index = max([v.index for v in s.values()], key=len)
        # keys to take from columns for final index
        keys = {k[:-1]: 0 for k in s}
        s = pd.concat({k: pd.concat(s, axis=1).reindex(index)[k] for k in keys})
    else:
        # use groupby to split multiindex
        splitlevel = lambda x, level, axis: {k: v.droplevel(level, axis) for k, v in x.groupby(level=level, axis=axis)}
        # vertical concat a Dataframe and move innermost index to top column hierarchy for wide format
        # s = pd.concat(splitlevel(pd.concat(s, axis=0), -2, axis=0), axis=1)
        # horizontal concat - split - vert. concat: avoid missing 'support' for some tests
        s = pd.concat(splitlevel(pd.concat(s, axis=1), [0, 1], axis=1), axis=0)
        # select scores
        # s.reorder_levels(np.roll(range(4), 1)).loc["macro avg"]
        # s.loc[(*[slice(None)]*3, 'macro avg')].droplevel(-1)
        # dict(list(s.groupby(level=-1)))['macro avg'].droplevel(-1)
        # s.groupby(level=-1).get_group('macro avg').droplevel(-1)
        # s.loc[s.groupby(level=-1).groups['macro avg']].droplevel(-1)
        # [s.reorder_levels(np.roll(range(3), 1)).loc[k] for k in dict.fromkeys([k[-1] for k in s.index])]
        # s = splitlevel(s, -1, 0)['macro avg']
    if return_yh or return_ph:
        s = [s] + ([yhs] if return_yh else []) + ([phs] if return_ph else [])
    return s


def cr_to_string(df, metric: str = "macro avg", col_space=8, cr_gap=None):
    cr_gap = cr_gap or col_space
    if metric:
        df = df.groupby(level=-1).get_group(metric)
        df = df.reorder_levels(np.roll(np.arange(len(df.index.levels)), 1))
    return df.to_string(col_space=(col_space, *[*[col_space] * 3, cr_gap + col_space] * 3)[:df.shape[1]],
                        na_rep="").replace("<NA>", "")


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


# def load_data(use_iris, load_acramos_params, split_random_state, proc_slices, agg_slices):
#     if use_iris:
#         print("dataset: IRIS")
#         X, y = datasets.load_iris(return_X_y=True)
#         names = datasets.load_iris().target_names
#         labels = sorted(set(y))
#         target_names = names
#
#         ds = dict(X=X, y=y, target_names=target_names)
#         ds_trn, ds_ho = train_test_split_dataset(ds, stratify=y, shuffle=True, random_state=split_random_state)
#         X_trn, y_trn = ds_trn['X'], ds_trn['y']
#         X_ho, y_ho = ds_ho['X'], ds_ho['y']
#         # build DataFrame for display
#         prev = prevalence(dict(train=dict(y=y_trn, target_names=target_names),
#                                holdout=dict(y=y_ho, target_names=target_names)), margin=True)
#         print(prev.to_string(col_space=7))
#     else:
#         print("dataset: acramos")
#         #all_class_labels = ["Negativ", "Quietschen", "Kreischen", "Kreischen,Quietschen", ]
#         all_class_labels = [','.join(v) for v in powerset(load_acramos_params['label_subset'])]
#         all_class_labels = ["Negativ", *all_class_labels[1:]]
#
#         # load only classes that contain 'pick_label'
#         pick_label = ''
#         # pick_label = to_binary_for
#         subset_labels = [v for v in all_class_labels if
#                          pick_label in v or v == "Negativ"] if pick_label else all_class_labels
#         reindex_subset = True
#         bin_label = False
#         if proc_slices:
#             ds = load_acramos_slices(**load_acramos_params,
#                                      subset=subset_labels,
#                                      # n=100,
#                                      reindex=reindex_subset, label_report=True,
#                                      # feat_cache="clustering")
#                                      feat_cache=Path(acr_dir) / f"{Path(__file__).stem}_slices")
#             # X, *_ = power_to(X, to="linear_magnitude")
#             ds['X'], *_ = power_to(ds['X'], to=to, abs_top_range=abs_top_range)
#             if agg_slices:
#                 ds['X'] = np.array([aggregate_feats(v, axis=0) for v in ds['X']])
#                 ds['y'] = np.array(conv_slice2file(ds['y']))
#                 assert (ds['y'] == ds['y_file']).all()
#             stratify = ds['y_file']
#             # split on hierarchy (not on flattened files)
#             ds_trn, ds_ho = train_test_split_dataset(ds, stratify=stratify, shuffle=True,
#                                                      random_state=split_random_state)
#             if agg_slices:
#                 X_trn, y_trn = ds_trn['X'], ds_trn['y']
#                 X_ho, y_ho = ds_ho['X'], ds_ho['y']
#             else:
#                 # flatten hierarchy
#                 X_trn, y_trn = [np.concatenate(ds_trn[k]) for k in ['X', 'y']]
#                 X_ho, y_ho = [np.concatenate(ds_ho[k]) for k in ['X', 'y']]
#         else:
#             postfix = "".join([v[0].upper() for v in sorted(load_acramos_params['label_subset'])])
#             ds = load_acramos(**load_acramos_params,
#                               feat_cache=Path(acr_dir) / f"{Path(__file__).stem}_{postfix}")
#
#             # these are the data settings for the project report 3 (first LDA results).
#             # For identical results: need random_state = 0 and stratification for TT-split, magnitude as power, no clipping.
#             # ds = load_acramos(dir=acr_dir, label_subset=["Kreischen", "Quietschen"], multiclass=True,
#             #                   feat_cache=Path(acr_dir) / f"{Path(__file__).stem}1")
#             ds = dataset_class_subset(ds, subset_labels=subset_labels, reindex=reindex_subset)
#             # X, *_ = power_to(X, to="linear_magnitude")
#             ds['X'], *_ = power_to(ds['X'], to=to, abs_top_range=abs_top_range)
#
#             target_names = ds['target_names']
#             X, y = ds['X'], ds['y']
#             # does data have a time axis?
#             if hasattr(y[0], "__len__"):
#                 # no time slice label available -> no supervised slice training -> marginalize time
#                 X = np.array([aggregate_feats(v, axis=0) for v in X])
#                 # pick first since rest ist identical
#                 y = np.array([v[0] for v in y])
#
#             if to_binary_for:
#                 y = np.array([to_binary_for in v for v in target_names], dtype=int)[y]
#                 target_names = [target_names[0], to_binary_for]
#
#             stratify = y
#             if bin_label:
#                 y = MultiLabelBinarizer().fit_transform([target_names[v].split(',') if v else [] for v in y])
#                 code = [([(v in k) * 1 for v in load_acramos_params['label_subset']]) for k in target_names]
#                 assert (np.array([code.index(v) for v in y.tolist()]) == stratify).all()
#                 target_names = load_acramos_params['label_subset']
#                 mdl = OneVsRestClassifier(mdl)
#             ds.update(X=X, y=y, target_names=target_names)
#             ds_trn, ds_ho = train_test_split_dataset(ds, stratify=stratify, shuffle=True,
#                                                      random_state=split_random_state)
#             X_trn, y_trn = ds_trn['X'], ds_trn['y']
#             X_ho, y_ho = ds_ho['X'], ds_ho['y']
#             # X_trn, X_ho, y_trn, y_ho = train_test_split(X, y, test_size=0.2, stratify=y,
#             #                                             # choose random_state such, that test set performance is near CV mean
#             #                                             shuffle=True, random_state=0)
#             # # build DataFrame for display
#             # prev = prevalence(dict(train=dict(y=y_trn, target_names=target_names),
#             #                        holdout=dict(y=y_ho, target_names=target_names)), margin=True)
#
#         labels = sorted(set(y_trn) | set(y_ho)) if len(y_trn.shape) == 1 else np.arange(2)
#         names = ds['target_names']
#         # sklearn API semantics
#         target_names = np.take(names, labels)
#
#         # build DataFrame for display
#         prev = prevalence(dict(train=ds_trn, holdout=ds_ho), margin=True)
#         # prev = prev.rename(columns=dict(file="slice"))
#         # prev.columns = [('slice', b) for a, b in prev.columns]
#         print(prev.to_string(col_space=([*[7] * (len(prev.columns) // 2), 15] * 2)[:len(prev.columns)]))
#     return ds_trn, ds_ho


# return scores on hold out data
def load_split_fit_score(mdl):
    if use_iris:
        print("dataset: IRIS")
        X, y = datasets.load_iris(return_X_y=True)
        names = datasets.load_iris().target_names
        labels = sorted(set(y))
        target_names = names

        ds = dict(X=X, y=y, target_names=target_names)
        ds_trn, ds_ho = train_test_split_dataset(ds, stratify=y, shuffle=True, random_state=split_random_state)
        X_trn, y_trn = ds_trn['X'], ds_trn['y']
        X_ho, y_ho = ds_ho['X'], ds_ho['y']
        # build DataFrame for display
        prev = prevalence(dict(train=dict(y=y_trn, target_names=target_names),
                               holdout=dict(y=y_ho, target_names=target_names)), margin=True)
        print(prev.to_string(col_space=7))
    else:
        print("dataset: acramos")
        #all_class_labels = ["Negativ", "Quietschen", "Kreischen", "Kreischen,Quietschen", ]
        all_class_labels = [','.join(v) for v in powerset(load_acramos_params['label_subset'])]
        all_class_labels = ["Negativ", *all_class_labels[1:]]

        # load only classes that contain 'pick_label'
        pick_label = ''
        # pick_label = to_binary_for
        subset_labels = [v for v in all_class_labels if
                         pick_label in v or v == "Negativ"] if pick_label else all_class_labels
        reindex_subset = True
        bin_label = False
        if proc_slices:
            ds = load_acramos_slices(**load_acramos_params,
                                     subset=subset_labels,
                                     # n=100,
                                     reindex=reindex_subset, label_report=True,
                                     # feat_cache="clustering")
                                     feat_cache=Path(acr_dir) / f"{Path(__file__).stem}_slices")
            # X, *_ = power_to(X, to="linear_magnitude")
            ds['X'], *_ = power_to(ds['X'], to=to, abs_top_range=abs_top_range)
            if agg_slices:
                ds['X'] = np.array([aggregate_feats(v, axis=0) for v in ds['X']])
                ds['y'] = np.array(conv_slice2file(ds['y']))
                assert (ds['y'] == ds['y_file']).all()
            stratify = ds['y_file']
            # split on hierarchy (not on flattened files)
            ds_trn, ds_ho = train_test_split_dataset(ds, stratify=stratify, shuffle=True,
                                                     random_state=split_random_state)
            if agg_slices:
                X_trn, y_trn = ds_trn['X'], ds_trn['y']
                X_ho, y_ho = ds_ho['X'], ds_ho['y']
            else:
                # flatten hierarchy
                X_trn, y_trn = [np.concatenate(ds_trn[k]) for k in ['X', 'y']]
                X_ho, y_ho = [np.concatenate(ds_ho[k]) for k in ['X', 'y']]
        else:
            postfix = "".join([v[0].upper() for v in sorted(load_acramos_params['label_subset'])])
            ds = load_acramos(**load_acramos_params,
                              feat_cache=Path(acr_dir) / f"{Path(__file__).stem}_{postfix}")

            # these are the data settings for the project report 3 (first LDA results).
            # For identical results: need random_state = 0 and stratification for TT-split, magnitude as power, no clipping.
            # ds = load_acramos(dir=acr_dir, label_subset=["Kreischen", "Quietschen"], multiclass=True,
            #                   feat_cache=Path(acr_dir) / f"{Path(__file__).stem}1")
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

            if to_binary_for:
                y = np.array([to_binary_for in v for v in target_names], dtype=int)[y]
                target_names = [target_names[0], to_binary_for]

            stratify = y
            if bin_label:
                y = MultiLabelBinarizer().fit_transform([target_names[v].split(',') if v else [] for v in y])
                code = [([(v in k) * 1 for v in load_acramos_params['label_subset']]) for k in target_names]
                assert (np.array([code.index(v) for v in y.tolist()]) == stratify).all()
                target_names = load_acramos_params['label_subset']
                mdl = OneVsRestClassifier(mdl)
            ds.update(X=X, y=y, target_names=target_names)
            ds_trn, ds_ho = train_test_split_dataset(ds, stratify=stratify, shuffle=True,
                                                     random_state=split_random_state)
            X_trn, y_trn = ds_trn['X'], ds_trn['y']
            X_ho, y_ho = ds_ho['X'], ds_ho['y']
            # X_trn, X_ho, y_trn, y_ho = train_test_split(X, y, test_size=0.2, stratify=y,
            #                                             # choose random_state such, that test set performance is near CV mean
            #                                             shuffle=True, random_state=0)
            # # build DataFrame for display
            # prev = prevalence(dict(train=dict(y=y_trn, target_names=target_names),
            #                        holdout=dict(y=y_ho, target_names=target_names)), margin=True)

        labels = sorted(set(y_trn) | set(y_ho)) if len(y_trn.shape) == 1 else np.arange(2)
        names = ds['target_names']
        # sklearn API semantics
        target_names = np.take(names, labels)

        # build DataFrame for display
        prev = prevalence(dict(train=ds_trn, holdout=ds_ho), margin=True)
        # prev = prev.rename(columns=dict(file="slice"))
        # prev.columns = [('slice', b) for a, b in prev.columns]
        print(prev.to_string(col_space=([*[7] * (len(prev.columns) // 2), 15] * 2)[:len(prev.columns)]))

    if len(y_trn.shape) == 1:
        # w = len(ytrn) / (4 * np.bincount(ytrn))
        # len(wc) == len(labels)
        classes = sorted(set(y_trn))
        if class_weight is None:
            wc = np.ones(len(labels))
        elif class_weight.startswith('balanced'):
            wc = compute_class_weight(class_weight='balanced', classes=classes, y=y_trn)
            # assert all(wc[ytrn] == compute_sample_weight(class_weight='balanced', y=ytrn))
        else:
            raise ValueError
        with np.printoptions(precision=2):
            print(f"class weights: {wc}")
        # sample weight: class indices don't
        ws = wc[list(map(classes.index, y_trn))]

    # test transform to 1 hot code
    # y_trn1h = [target_names[v].split(',') if v else [] for v in y_trn]
    # mlb = MultiLabelBinarizer().fit(y_trn1h)
    # y_trn1h = mlb.transform(y_trn1h)
    # y_ho1h = mlb.transform([target_names[v].split(',') if v else [] for v in y_ho])
    print("")
    print("Holdout test")
    print(f"model: {str(mdl)}")

    # # use flat data, no scoring label aggregation
    # s2, yh2 = model_fit_score(mdl, X=X_trn, y=y_trn, X_ho=X_ho, y_ho=y_ho,
    #                           names=names, target_names=target_names, eval_on_trainset=True, return_yh=True)
    # use ragged data, aggregated label scoring internally
    s, yh, ph = model_fit_score(mdl, X=ds_trn['X'], y=ds_trn['y'], X_ho=ds_ho['X'], y_ho=ds_ho['y'],
                                target_names=target_names, eval_on_trainset=True, return_ph=True, return_yh=True)
    # #y_ho_names = np.take(names, y_ho)
    # y_ho_names = np.take(target_names, list(map(labels.index, y_ho)))
    # cm = pd.crosstab(index=y_ho_names, columns=yh, margins=True).loc[[*target_names, "All"], :]
    yh = yh['test']
    ph = ph['test']
    cm = confusion_matrix_df(y_ho, y_pred=yh, target_names=target_names, margins=True)
    print(cm.set_index(cm.index + " ").to_string(index_names=False, col_space=7))
    print(cr_to_string(s, metric="macro avg"))
    print(cr_to_string(s[s.index.isin(np.concatenate((["macro avg"], names)), level=-1)], metric=""))

    # ap = ap_class_score(y_ho, y_score=ph, labels=labels)
    s1 = score_agg(y_true=ds_ho['y'], y_pred=yh,
                   # score_fun=f1_score, average="macro",
                   # score_fun=average_precision_score, y_score=ph, pos_label=2,
                   # score_fun=classification_report, target_names=target_names,
                   # score_fun=confusion_matrix_df, target_names=target_names, margins=True,
                   score_fun=file_classification_report, as_dataframe=True, target_names=target_names,
                   label_agg=conv_slice2file, labels=None)
    s1 = dict(zip(["flat", "agg"], s1))
    if isinstance(list(s1.values())[0], pd.DataFrame):
        s1 = pd.concat(s1, axis=1)
    return s


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    pd.options.display.float_format = '{:,.2f}'.format
    # acr_dirs = dict(
    #     V1=Path.home() / 'prj' / 'acrDb',
    #     V2=Path(r"D:\ADSIM\Import-2023-04"),
    # )

    label_subset = ["Kreischen", "Quietschen"]
    #label_subset = ["Flachstelle"]
    to = "db"
    abs_top_range = 60
    endpoint = "macro avg"
    acramos = "V1"
    use_iris = False
    proc_slices = False
    #proc_slices = True
    agg_slices = False
    agg_slices = True
    to_binary_for = ""
    # to_binary_for = "Kreischen"
    # to_binary_for = "Quietschen"
    split_random_state = 42

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

    if label_subset == ["Kreischen", "Quietschen"]:
        load_acramos_params = dict(dir=acramos, label_subset=label_subset, multiclass=True, split=1.0, )
        # hyper param setting for slice processing. Slice labels are only available for Quietschen
        data_slices = pd.DataFrame(
            [[True, True, ""],
             [True, False, ""],
             [False, False, "Quietschen"]
             ],
            columns=["proc_slices", "agg_slices", "to_binary_for"])
        # reorder columns
        data_slices = data_slices[data_slices.columns[np.roll(np.arange(data_slices.shape[1]), 1)]]
        data_slices = pd.DataFrame(dict(acramos=["V1", "V2"])).merge(data_slices, how="cross")
        # hyper param setting for Kreischen
        data_K = pd.DataFrame(
            [[False, False, "Kreischen"],
             ],
            columns=["proc_slices", "agg_slices", "to_binary_for"])
        # reorder columns
        data_K = data_K[data_K.columns[np.roll(np.arange(data_K.shape[1]), 1)]]
        data_K = pd.DataFrame(dict(acramos=["V1", "V2"])).merge(data_K, how="cross")

        # hyper param setting for Kreischen+Quietschen
        data_KQ = pd.DataFrame(
            [[False, False, ""],
             ],
            columns=["proc_slices", "agg_slices", "to_binary_for"])
        # reorder columns
        data_KQ = data_KQ[data_KQ.columns[np.roll(np.arange(data_KQ.shape[1]), 1)]]
        data_KQ = pd.DataFrame(dict(acramos=["V1",
                                             #"V2"
                                             ])).merge(data_KQ, how="cross")

        settings = data_slices
        #settings = data_K
        settings = data_KQ

        # settings = settings.merge(pd.DataFrame(dict(scale=["std"])), how="cross")
        settings = settings.merge(#pd.DataFrame(dict(model=["LDA", "SVC", "RF"])),
                                  pd.DataFrame(dict(model=["LDA"])),
                                  how="cross")
    elif label_subset == ["Flachstelle"]:
        load_acramos_params = dict(dir=acramos, label_subset=label_subset, multiclass=True, split=1.0, )
        settings = pd.DataFrame(dict(acramos=["V2"])).merge(pd.DataFrame(dict(model=list(models))), how="cross")
    #settings = settings[14:15]
    s = {}
    print(f"{len(settings)} experiments:")
    print(settings)
    for d in settings.to_dict(orient='index').values():
        # unpack d in local scope
        exec(f'{", ".join(d)} = d.values()')
        acr_dir = acr_dirs[acramos]
        load_acramos_params.update(dir=acr_dir)
        mdl = models[model]
        # try:
        #     mdl = mdl.set_params(verbose=True)
        # except ValueError as e:
        #     pass
        if scale == "std":
            mdl = make_pipeline(
                StandardScaler(),
                mdl
            )
        scores = load_split_fit_score(mdl=mdl)

        target_names = scores.loc['test'].index[:-3]
        scores = scores[scores.index.isin(np.concatenate(([endpoint], target_names)), level=-1)]
        # reorder cols
        if "slice" in scores:
            scores = scores[["file", "slice"]]
        else:
            scores = scores.rename(columns=dict(flat="file"))
        s[tuple(d.values())] = scores

    # summary = pd.concat({k: (v["file"] if "file" in v else v["flat"]) for k, v in s.items()})
    summary = pd.concat(s)
    summary.index = summary.index.rename(list(settings) + ["infer", "metric"])
    with pd.option_context('expand_frame_repr', False):
        print(cr_to_string(summary, metric=endpoint, col_space=6))
    summary.to_excel('results/summary.xlsx', merge_cells=False)