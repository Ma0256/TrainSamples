# save labels from exported acramos db tables to CSV file
from pathlib import Path

import pandas as pd
import numpy as np
import torchaudio
from statsmodels.stats import inter_rater
from sklearn import metrics
from pprint import pprint
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import re
import itertools

matplotlib.use('TkAgg')


def find_ties(ml, n_raters):
    # counts for maj. voting
    # to integer via pd categorical
    cats = np.unique(ml)
    # cats = lmap
    cat_type = pd.CategoricalDtype(categories=cats, ordered=False)
    ml_intlabel = ml.astype(cat_type).apply(lambda x: x.cat.codes)
    # to integer via pd Series mapping
    # code = pd.Series({k: v for v, k in enumerate(cats)})
    # code = pd.Series(range(len(cats)), cats)
    # must index Series with 1D array
    # ml_intlabel = code[ml.values.reshape(-1)].values.reshape(ml.shape)
    # to integer via numpy unique
    ml_intlabel = np.unique(ml, return_inverse=True)
    assert all(ml_intlabel[0] == cats)
    # ml_intlabel = pd.DataFrame(ml_intlabel[1].reshape(ml.shape), index=ml.index, columns=ml.columns)
    # prefer 3D tensor to matrix
    ml_intlabel = ml_intlabel[1].reshape(len(ml), -1, n_raters)
    counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=3), axis=-1, arr=ml_intlabel)
    # counts = {k: ml[k].apply(lambda x: x.value_counts(), axis=1) for k in labels?}
    ties = np.sum(counts == counts.max(-1, keepdims=True), axis=-1) > 1
    return ties


def get_mode(x, minlength=None, return_ties=False, return_counts=False):
    # switch original representation to integer codes
    code, coded = np.unique(x, return_inverse=True)
    # avoid side effect on caller
    x1 = coded.reshape(x.shape)
    minlength = minlength or len(np.unique(x1))
    counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=minlength), axis=-1, arr=x1)
    # consensus from ordered categories (0 < 0.5 < 1.)
    # count_hist = pd.DataFrame(counts).value_counts()
    cons_mode = np.argmax(counts, -1)
    # go back to original representation
    cons_mode = code.astype(type(np.array(x)[0, 0]))[cons_mode]
    # assert all(counts[np.arange(len(counts)), cons_mode] == counts.max(-1))
    ties = np.sum(counts == counts.max(-1, keepdims=True), axis=-1) > 1
    assert set(counts.sum(1)) == {x1.shape[1]}
    if return_ties or return_counts:
        return (cons_mode,) + ((ties,) if return_ties else ()) + ((counts,) if return_counts else ())
    else:
        return pd.Series(cons_mode, index=x.index)[~ties]


def interrater_agreement(x, f_score):
    x.loc[:, :] = np.unique(x, return_inverse=True)[1].reshape(x.shape)
    irr = {(a, b): f_score(x[a], x[b]) for a, b in itertools.combinations(x.columns, 2)}
    irr = pd.Series(irr).sort_values(ascending=False)
    # counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=3), axis=-1, arr=x)
    # cons_mode = np.argmax(counts, -1)
    # ties = np.sum(counts == counts.max(-1, keepdims=True), axis=-1) > 1
    cons_mode, ties, counts = get_mode(x, return_ties=True, return_counts=True)
    n_ties = sum(ties)
    # all ties are {0, 1, 2}, because: 3 votes, 3 categories
    # x.loc[ties, :]
    # consensus from ordered categories (0 < 0.5 < 1.)
    # count_hist = pd.DataFrame(counts).value_counts()
    # mean_hist = x.mean(axis=1).value_counts()
    cons_ordered = x.mean(axis=1).round().astype(int)
    cons = pd.DataFrame(dict(**dict(x), counts=list(counts),
                             ties=ties, cons_mode=cons_mode, cons_ordered=cons_ordered))
    # disregard undefined modes
    # cons['cons_mode'] = cons['cons_mode'].astype('Int32')
    # cons.loc[ties, 'cons_mode'] = pd.NA
    cons = cons[~ties]
    # rater vs. BIASED consensus: all raters vs. same consensus
    rcr = {(a, b): f_score(cons[a], cons[b])
           for a, b in itertools.combinations(list(x.columns) + ['cons_mode'], 2)}
    # unbiased consensus from remaining raters: different cons.
    cons_unb = {f'cons_{k}': get_mode(x[sorted(set(x.columns) - {k})], return_ties=False) for k in x.columns}
    # common subset of no-ties for identical support (--> results in all identical votes, i.e., IRR 1.0)
    # cons_unb = pd.concat(cons_unb, axis=1, sort=True)
    # cons_unb = cons_unb.dropna()
    irr_cons_ub = {k: f_score(x.loc[cons_unb[f'cons_{k}'].index, k], cons_unb[f'cons_{k}'])
                   for k in x.columns}
    rcr = pd.DataFrame(dict(
        cons_b={list(set(k) - {'cons_mode'})[0]: rcr[k] for k in rcr if 'cons_mode' in k},
        n_b=len(cons),
        cons_ub=irr_cons_ub,
        n_ub={k: len(cons_unb[f'cons_{k}']) for k in x.columns}
    ))
    return irr, rcr


if __name__ == "__main__":
    if Path.cwd().name == Path(__file__).parent.name:
        # default CWD

        # dataset dir for this project
        prj_root = Path.home() / "prj"
        acr_dir = prj_root / "acrDb"
        # sound file dir
        data_dir = acr_dir
        # data_dir = acr_dir
        # data_dirs = [v for v in acr_dir.glob("*") if v.is_dir()]
        # files = [list(data_dir.glob("*.wav")) for data_dir in data_dirs]
        wavs = list(data_dir.glob("**/*.wav"))

        # get metadata, takes a lot of time
        print("Reading audio metadata ...")
        wav_info = {k.stem[:-3]: vars(torchaudio.info(k)) for k in tqdm(wavs) if k.stem.endswith("01")}
        wav_info = pd.DataFrame(wav_info).T.infer_objects()
        wav_info["dur"] = wav_info["num_frames"] / wav_info["sample_rate"]
        print("First file metadata:")
        print(wav_info.iloc[0])

        # make sure ids are unique, to not lose files which may overlap in different subdirs
        assert len(wavs) == len({v.stem for v in wavs})
        wav_ids = {str(v.stem).rsplit("_", 1)[0]: v.parts[-2] for v in wavs}

        cols = ["FileDatum", "FileZeit", "MkID", "Rohdaten", "Tag"]
        print("Opening table file ...")
        db_ids_df = pd.read_excel(data_dir / "TabZug.xlsx", usecols=cols)[cols]
        print("...completed")
        db_ids_df["ID"] = [f"{r[0].strftime('%Y-%m-%d')}_{r[1].strftime('%H-%M-%S')}" for i, r in db_ids_df.iterrows()]
        # convert labels
        print(f"Found tags: {db_ids_df['Tag'].unique()}")
        label = []
        for i, v in enumerate(db_ids_df["Tag"]):
            if str(v).lower() == "nan":
                v = "[Negativ]"
            else:
                v = "".join(f"[{k}]" for k in ["Kreischen", "Quietschen"] if k in v)
            label.append(v)
        db_ids_df["Label"] = label
        print(f"Converted labels: {db_ids_df['Label'].unique()}")
        db_ids_df["Kreischen"] = ["Kreischen" in str(v) for v in db_ids_df["Tag"]]
        db_ids_df["Quietschen"] = ["Quietschen" in str(v) for v in db_ids_df["Tag"]]

        db_ids = dict(db_ids_df[["ID", "MkID"]].values)

        if len(db_ids) != len(db_ids_df):
            ovid = {k: v for k, v in zip(*(np.unique(db_ids_df["ID"], return_counts=True))) if v > 1}
            print(f"Warning: loosing ids {ovid} from database due to id overlap")

        print("Database entries which should have existing raw data according 'Rohdaten' flag")
        print(dict(zip(*np.unique(db_ids_df[db_ids_df["Rohdaten"] == 1]["MkID"], return_counts=True))))

        print("Samples with both database entry AND raw data found")
        db_ids_df["WAV"] = [k in wav_ids for k in db_ids_df["ID"]]
        inboth = {k: wav_ids[k] for k in db_ids_df["ID"] if k in wav_ids}
        print(dict(zip(*np.unique(list(inboth.values()), return_counts=True))))

        campaigns = Path.cwd() / "TabMess.xlsx"
        campaigns = pd.read_excel(campaigns)
        campaigns = campaigns[["MkID", "MkName"]]
        mkids = dict(campaigns.values)
        db_ids_df["campaign"] = [mkids[k] if k in mkids else k for k in db_ids_df["MkID"]]

        db_ids_df["dur"] = [(wav_info.loc[k, "dur"] if k in wav_info.index else None) for k in db_ids_df["ID"]]
        db_df = db_ids_df[db_ids_df["dur"] > 0]
        print("Saving label CSV file ...")
        db_df[["campaign", "ID", "Label"]].to_csv(f"adsim_label.csv", index=False)
        print("...completed")

        _, axs = plt.subplots(nrows=2, ncols=2, sharey='row', )
        axs[0, 0].hist(db_ids_df["Label"])
        plt.setp(axs[0, 0].get_xticklabels(), rotation=10, ha="right", rotation_mode="anchor")
        axs[0, 0].set_title(f"All {len(db_ids_df)} database entries")
        axs[0, 1].hist(db_df["Label"])
        plt.setp(axs[0, 1].get_xticklabels(), rotation=10, ha="right", rotation_mode="anchor")
        axs[0, 1].set_title(f"Entries with WAV file: {len(db_df)}")

        # axs[1, 0].hist(db_ids_df["dur"], bins=40)
        # axs[1, 0].set_title("No duration found", backgroundcolor="white")
        axs[1, 0].text(x=0.5, y=0.5, s="No duration found", ha="center", transform=axs[1, 0].transAxes)
        axs[1, 1].set_xlabel("audio duration [s]")
        axs[1, 1].hist(db_df["dur"], bins=np.arange(0, max(db_df["dur"]) + 1, 2))

        plt.show()
    else:
        # "new" version for DB restored from SQL dump
        # modified CWD
        db_dir = Path.cwd() / 'db'
        data_dir = Path.cwd() / 'data'
        tasks = [
            'tabzug',
            'tabhoerprobe',
        ]
        read_wavmetadata = True
        voting_scheme = "unanimous"
        #voting_scheme = "majority"
        overrule_ties = False
        label_file = f"adsim_label.csv"
        wav_relative_to_label_file = True
        export_labels = ["Flachstelle",
                         #"Wagenaufbau",
                         ]

        dfs = {k.stem: pd.read_csv(k, sep=';', decimal=',')  # , parse_dates=['FileDatum'])
               for k in db_dir.glob('*.csv')}
        if 'tabzug' in tasks:
            dfs['tabmess'] = dfs['tabmess'].set_index('MkID')
            # trains = pd.concat((dfs['tabmess']['MkName'], dfs['tabzug']['MkID'].value_counts()), axis=1)
            campaigns = pd.DataFrame(dict(ProjNr=dfs['tabmess']['ProjNr'],
                                          MkName=dfs['tabmess']['MkName'],
                                          Züge=dfs['tabzug']['MkID'].value_counts().astype('Int64')))
            dfs['tabmess'].insert(2, 'Züge', dfs['tabzug']['MkID'].value_counts())
            has_data = dfs['tabzug']['Rohdaten'] == True
            campaigns['ZügeRohdaten'] = dfs['tabzug'].loc[has_data, 'MkID'].value_counts().astype('Int64')

            # pick channel 1 only
            wavs = [v.relative_to(data_dir)  # .with_suffix('')
                    for v in data_dir.glob("**/*_01.wav")]
            folder_counts = pd.DataFrame(v.parent.parts for v in wavs).value_counts().rename('Wavs')
            print(f'Found {len(wavs)} WAV file train passings (*_01.wav)')
            print(folder_counts)
            campaigns['WavOrdner'] = campaigns['ProjNr'].isin({v.parts[-3] for v in wavs}) & \
                                     campaigns['MkName'].isin({v.parts[-2] for v in wavs})
            campaigns = campaigns.join(folder_counts.astype('Int64'), on=['ProjNr', 'MkName'], how='left', sort=False)
            if "use dictionary to simplify index building in place of DataFrame column ops":
                idx = pd.Series({v: (v.parts[-2], '_'.join(v.parts[-1].split('_')[-3:-1])) for v in wavs})
                # idx = pd.DataFrame([dict(Wav=v, MkName=v.parts[-2], UID='_'.join(v.parts[-1].split('_')[-3:-1])) for v in wavs]).set_index('Wav')
                prefixes = ['_'.join(v.name.split('_')[:-3]) for v in wavs]
                print("Get all duplicate instances, i.e., at least 2 per unique item")
                dup = idx[idx.duplicated(keep=False)]
                print(dup)
                idx = idx[~idx.duplicated(keep='last')]
                print(
                    f"Removing {len(wavs) - len(idx)} duplicates, keep last occurence (alternative: keep most frequent)")
                # MultiIndex from two columns would be easier ...
                # wavs = pd.Series(idx.index, index=pd.MultiIndex.from_tuples(idx.values, names=['MkName', 'UID']), name='Wav')
                wavs = pd.DataFrame(dict(Wav=idx.index),
                                    index=pd.MultiIndex.from_tuples(idx.values, names=['MkName', 'UID']))

            else:
                wavs = pd.DataFrame(v.parts for v in wavs)
                # treat file name: extract the date_time portion
                # wavs.iloc[:, -1] = wavs.iloc[:, -1].str.replace(r'_01$', '')
                # file name has variable number of delimited substrings: 111_222_..._date_time_01
                prefixes = wavs.iloc[:, -1].str.split('_').str[:-3].str.join('_').value_counts()
                # save original WAV file name in index
                wavs = wavs.set_index(wavs.columns[-1], drop=False)
                # wavs = pd.DataFrame(dict(wavs) | dict(FileName=wavs.iloc[:, -1])).set_index('FileName')
                wavs.columns = [*wavs.columns[:-1], 'UID']
                wavs['UID'] = wavs['UID'].str.split('_').str[-3:-1].str.join('_')
                print('Removing duplicates:')
                print(wavs[wavs.duplicated()].groupby([0, 1]).size())
                wavs = wavs[~wavs.duplicated()]
                wavs['UID'] = wavs.iloc[:, 1:].apply('/'.join, axis=1)
                wavs = wavs.reset_index().set_index('UID')
                wavs = wavs[sorted(wavs.columns, key=str)]
                wavs = pd.DataFrame(dict(Wav=wavs.apply(lambda r: Path(*r), axis=1)))

            assert not any(wavs.index.duplicated()), "WAV index not deduped"

            cols = ["FileDatum", "FileZeit", "MkID", "IDZug", "Rohdaten", "Tag", "TAG2"]
            # DB 'ProjNR' 'N81_501DW' has entries which do not map to a unique key. Further, it does not have data folders.
            # --> use only campaigns with data (i.e. folders)
            rows = dfs['tabzug']['MkID'].isin(campaigns.index[campaigns['WavOrdner']])
            # rows = slice(None)
            trains = dfs['tabzug'].loc[rows, cols]
            trains['Rohdaten'] = trains['Rohdaten'] > 0
            trains = trains.convert_dtypes()
            # dfs['tabzug'].loc[:, dfs['tabzug'].columns.str.contains('[zZ]eit$|[dD]atum$')]#, flags=re.IGNORECASE)]
            # for k in trains.filter(regex='[zZ]eit$|[dD]atum$'):
            #    trains[k] = pd.to_datetime(trains[k])
            trains['FileDatum'] = trains['FileDatum'].str.split().str[0]
            trains['FileZeit'] = trains['FileZeit'].str.split().str[-1]
            trains['UID'] = trains['FileDatum'] + '_' + trains['FileZeit'].str.replace(':', '-')
            # trains['WavName'] = trains.apply(lambda r: '_'.join((r['FileDatum'], r['FileZeit'].replace(':', '-'))), axis=1)
            trains['MkName'] = np.array(dfs['tabmess']['MkName'][trains['MkID']])
            # trains['UID'] = np.array(dfs['tabmess']['MkName'][trains['MkID']]) + '/' + trains['UID']
            trains = trains.set_index(['MkName', 'UID'])
            assert not any(trains.index.duplicated()), "No unique file paths in database"

            # join database and WAV paths from left
            # trains = pd.concat((trains.set_index('UID'), wavs), join='inner', axis=1)
            trains = trains.join(wavs)
            # campaigns.join(train_nowav.reset_index()['MkName'].value_counts().rename(r'Züge\WAV').astype('Int64'), on='MkName', how='left')
            campaigns[r'Züge\Wav'] = trains[trains['Wav'].isna()]['MkID'].value_counts().astype('Int64')
            campaigns[r'Züge&Wav'] = trains[trains['Wav'].notna()]['MkID'].value_counts().astype('Int64')
            # sum trains with and without WAV
            checksum = campaigns.filter(regex='e.Wav').sum(axis=1).astype('Int64') - campaigns['Züge']
            trains = trains[trains['Wav'].notna()]

            # indicate wavs from defined projects and campaigns
            wavs['isin_db_campaigns'] = wavs['Wav'].map(lambda x: x.parts[-3]).isin(dfs['tabmess']['ProjNr'])
            wavs['isin_db_campaigns'] &= wavs.index.get_level_values(0).isin(dfs['tabmess']['MkName'])

            print("")
            print(f'Found   {len(dfs["tabzug"]):>10} trains in acramos DB')
            print(f'Thereof {sum(rows):>10} trains belong to campaigns with a WAV folder')
            print(f'Thereof {len(trains):>10} trains have a WAV file')

            # get metadata, takes a lot of time
            if read_wavmetadata:
                print("Reading audio metadata ...")
                wav_info = {i: vars(torchaudio.info(data_dir / k))
                            for i, k in tqdm(trains['Wav'].items()) if k.stem.endswith("01")}
                wav_info = pd.DataFrame(wav_info).T.infer_objects().rename_axis(trains.index.names)
                wav_info["dur"] = wav_info["num_frames"] / wav_info["sample_rate"]
                print("First file metadata:")
                print(wav_info.iloc[0])
                trains = trains.join(wav_info[["dur"]], how='left')

            tag_counts = pd.DataFrame(dict(Tag=dfs['tabzug']['Tag'].value_counts(),
                                           Tag2=dfs['tabzug']['TAG2'].value_counts())).astype('Int32')
            campaigns['Tag'] = trains[trains['Tag'].notna()]['MkID'].value_counts().astype('Int64')
            campaigns['TAG2'] = trains[trains['TAG2'].notna()].groupby('MkID').size().astype('Int64')

            # remove trains without label
            # trains = trains[trains['TAG2'].notna()]
            # process TAG2
            # mapping for the individual classes
            l = [{f"[kein-{k}]": .0, "": .5, f"[{k}]": 1.0} for k in ("Kreischen", "Quietschen")]
            # combination to labels (i.e., tags)
            tags = {f"{ko}{ki}": (l[0][ko], l[1][ki],) for ko in l[0] for ki in l[1]}
            assert set(trains['TAG2'].value_counts().index) <= set(tags), "Not all DB labels are known"
            kq = list(zip(*[tags.get(k, [None] * 2) for k in trains['TAG2']]))
            trains['Kreischen'], trains['Quietschen'] = kq
            # trains['K'], trains['Q'] = trains['TAG2'].apply(lambda x: tags[x] if x in tags else (np.nan, np.nan)).str

        if 'tabhoerprobe' in tasks:
            # #### manual labels #######
            ml = dfs['tabhoerprobe'].sort_values(by=['ZUGID', 'Proband'])
            # each train should have at least 3 labeling passes, i.e., rows
            assert all(ml['ZUGID'].value_counts() >= 3), "too few labelings"
            # example: all flat spots
            dfs['tabhoerprobe'][dfs['tabhoerprobe']['Flachstelle'] > 0].sort_values(by='ZUGID')
            # picked trains which should have 3 manual labels
            ws = dfs['tabwavauswahl']
            ids = set(ml['ZUGID'])
            # it appears that
            # set(ml['ZUGID']) == set(ws['ZUGID'])

            # get sections for rating from contiguous IDs
            i = ws['WAID'].diff().abs() > 1
            i = i[i].index
            ws_tid = [v.sort_values() for v in np.split(ws['ZUGID'], i)]
            assert all([~any(v.duplicated()) for v in ws_tid]), "selection of trains for labeling not unique"
            mls = [ml[ml['ZUGID'].isin(v)] for v in ws_tid]
            # summary of sections
            ss = pd.DataFrame({f"WA{i}": v.iloc[:, 2:].sum(numeric_only=True) for i, v in enumerate(mls)})
            ss = pd.concat((pd.DataFrame(dict(Züge=map(len, ws_tid), Reviews=map(len, mls)), index=ss.columns).T, ss))
            ss['Hörprobe'] = (len(set(ml['ZUGID'])), len(ml), *ml.iloc[:, 2:].sum(numeric_only=True),)

            # make consensus - combine reviews
            labels = ml.columns[5:-3]
            labels = labels[~labels.str.contains('unsicher', case=False)]
            # labels = [(v, labels[labels.str.contains(v) & labels.str.contains('unsicher', case=False)])
            #          for v in labels[~labels.str.contains('unsicher', case=False)]]
            ml = dfs['tabhoerprobe']
            ml['Proband'] = ml['Proband'].apply(lambda x: ''.join([v[0] for v in x.split()]))
            ml = ml.set_index(['ZUGID', 'Proband']).sort_index(kind='stable')
            # convert to 3 state logic
            # assert not any(ml.filter(regex='uffaellig').all(axis=1))
            # delete redundant column
            (1 - ml['Auffaellig'] == ml['Unauffaellig']).value_counts()
            del ml['Unauffaellig']
            for k in labels:
                ku = f"{k}Unsicher"
                # check mutual exclusivity of True and Maybe ('unsicher')
                assert not any((ml[ku] != 0) & (ml[k] != 0))
                if not "3VL":
                    # only allowed keys in dict, others will raise
                    lmap = "FUT"
                else:
                    # lmap = np.arange(3) / 2
                    lmap = np.linspace(0, 1, 3)
                d = dict(zip(itertools.product(*[range(2)] * 2), lmap))
                mlk = ml[[k, ku]].apply(lambda x: d[tuple(x)], axis=1)
                ml[k] = mlk
                ml = ml.drop(columns=ku)
            print("")
            print(f'Found   {len(dfs["tabhoerprobe"]):>10} ratings from {len(ml.index.unique(level=1))} raters ' \
                  f'for {len(labels)} categories for {len(ml.index.unique(level=0))} trains in tabhoerprobe')

            lmap = dict(zip(lmap, ["Negativ", "Unsicher", "Positiv"]))
            # same file same rater duplicated - rating could be identical or non-identical
            dup_ratings = ml[ml.index.duplicated(False)].reset_index().set_index(['HPID'])
            dup_ratings_info = dict(n_trains=len(dup_ratings['ZUGID'].value_counts()))
            # ... and the subset with non-identical rating
            dups_conflict = dup_ratings[~dup_ratings.iloc[:, :-2].duplicated(keep='last')]
            dups_conflict = dups_conflict[dups_conflict.duplicated(subset=['ZUGID', 'Proband'], keep=False)]
            dup_ratings_info['trains_conflicting'] = dups_conflict['ZUGID'].value_counts()
            dup_ratings_info['n_trains_conflicting'] = len(dups_conflict['ZUGID'].value_counts())
            # sort the highest train counts to top
            # dups_conflict = dups_conflict.iloc[(-dups_conflict.groupby('ZUGID').transform('size')).argsort(kind='stable')]
            # same result via sort_values key
            dups_conflict = dups_conflict.sort_values(by='ZUGID', key=lambda x: x.map(x.value_counts()),
                                                      ascending=False, kind='stable')
            # enumerate dups
            ml = ml.set_index(ml.groupby(level=[0, 1]).cumcount(ascending=False).rename('WH'), append=True)
            # remove raters with low count
            i = ml.index.get_level_values(level=1).value_counts()
            i = list(i[i < 100].index)
            ml = ml.drop(ml.loc[(slice(None), i, slice(None))].index)
            print(f'Removing {len(i)} rater(s): {i} with too few votes')

            # the most recent repetition is 'WH' = 0, sort chronologically
            ml = ml.unstack('Proband').convert_dtypes().sort_index(level=[0, 1], ascending=[1, 0])
            # discard trains rated more than 2 times by the same reviewer
            ml = ml[ml.index.get_level_values('WH') < 2]
            # keep only train rating repetitions with count greater threshold
            # ml.groupby('WH').filter(lambda x: len(x) > 10)
            na_rows = ml[ml.columns.unique(level=0)[:-1]].isna().any(axis=1)
            ml = ml[~na_rows]

            raters = ml.columns.unique('Proband')
            cons = pd.DataFrame({k: ml[k].mean(axis=1) for k in labels})
            # ties on frame level
            ties = find_ties(ml[labels], n_raters=len(raters))
            # ties = np.nonzero(ties)
            ties = pd.DataFrame(ties, index=ml.index, columns=labels)
            # example selection: all flat spots p=.5 which are not ties
            ml[(cons['Flachstelle'] == .5) & ~ties['Flachstelle']]['Flachstelle']
            tie_trains = ties[ties.index.isin([0], level=1) & ties.any(axis=1)].reset_index()['ZUGID']
            # ties with conflicting duplicates: maybe duplicate resolves tie
            tie_trains_dups = set(tie_trains) & set(dups_conflict['ZUGID'])
            tie_trains_dups = (dups_conflict[dups_conflict['ZUGID'].isin(tie_trains_dups)],
                               ties[ties.index.isin(set(tie_trains) & set(dups_conflict['ZUGID']), level=0)])

            x = ml[ml.index.isin([0], level=1)]
            ntrains = len(x)
            counts = {k: pd.DataFrame(get_mode(x[k],
                                               minlength=3,
                                               return_counts=True)[-1]).value_counts() for k in labels}
            # INFO: statsmodels.inter_rater has this function ...
            # inter_rater.aggregate_raters(x)[0] == get_mode(x, return_counts=True)[1]
            counts = pd.concat(counts, axis=1).astype('Int32')
            x = np.array(list(itertools.product(lmap, repeat=3)))
            x = pd.DataFrame(dict(x=list(x), M=np.mean(x, axis=1), SD=np.std(x, axis=1),
                                  cnt=map(tuple, get_mode(x, return_counts=True)[1])))
            counts.insert(0, 'SD', pd.DataFrame({k: v['SD'].unique() for k, v in x.groupby('cnt')}).T)
            counts.insert(0, 'M', pd.DataFrame({k: v['M'].unique() for k, v in x.groupby('cnt')}).T)
            if voting_scheme == "unanimous":
                counts.insert(0, 'cons', counts['M'].where(counts['SD'] == 0, list(lmap)[1]))
            elif voting_scheme == "majority":
                # consenus computation: rounding
                counts.insert(0, 'cons', (counts.M * 2).round() / 2)
            else:
                raise ValueError
            counts.insert(3, 'valid', (counts['SD'] < 0.41))
            counts = counts.sort_values(by='SD', kind='stable')
            counts = counts.sort_values(by='cons', kind='stable', ascending=False,
                                        key=lambda x: np.isin(x, [0.0, 1.0]))
            unanimous_sure = counts.loc[counts['cons'].isin([0, 1]) & (counts['SD'] == 0), labels]
            unanimous_sure.index = ["EinhelligNeg", "EinhelligPos"]

            # # agreement between raters
            f_score = metrics.cohen_kappa_score
            # get IRR and consensus-reliability, handle each label independently
            irr = {k: interrater_agreement(ml[ml.index.isin([0], level=1)][k], f_score=f_score)
                   for k in labels}
            # rater-consensus-reliability
            rcr = pd.concat({k: v[1]['cons_ub'] for k, v in irr.items()}, axis=1).dropna(axis=1).T
            irr = pd.concat({k: v[0] for k, v in irr.items()}, axis=1).T
            # counts for highest score
            best_count = pd.Series(np.bincount((-rcr.T).apply(pd.Series.argsort, axis=0).values[0]), index=rcr.columns)
            best_count = best_count.sort_values(ascending=False)
            # rank scores ascending: higher score is higher rank is better
            rr = (rcr.T.rank() - 1).sum(axis=1).sort_values(ascending=False)
            fk = {k: inter_rater.fleiss_kappa(inter_rater.aggregate_raters(ml[ml.index.isin([0], level=1)][k])[0])
                  for k in labels}

            print(f'Votes for {ntrains} trains:')
            summary = counts.groupby('cons')[labels].sum().T.rename(columns=lmap)
            summary = summary.sort_values('Positiv', ascending=False)
            summary = summary.join((counts[labels].sum().rename("Summe"),
                                    unanimous_sure.iloc[1],
                                    pd.Series(fk, name="FleissKappa"),
                                    #pd.Series(raters[np.argsort(-rcr.values, axis=1)[:, 0]], rcr.index,)
                                    rcr.apply(lambda x: raters[np.argmax(x)], axis=1).rename("BesterBewerter"),
                                    ))
            with pd.option_context('display.float_format', '{:0.2f}'.format, 'expand_frame_repr', False):
                print(summary)

            counts.insert(0, 'sel', counts['valid']
                          # & counts['cons'].isin([0., 1.])
                          )

            cons = pd.DataFrame()
            contradictory = pd.DataFrame()     # files that should be reviewed
            for k in export_labels:
                x = ml[ml.index.isin([0], level=1)][k].reset_index(1, drop=True)
                _, ties, cnts = get_mode(x, return_ties=True, return_counts=True)
                #c = pd.Series(map(tuple, cnts), x.index, name=k).map(counts['cons'])
                c = pd.Series([counts['cons'][tuple(v)] for v in cnts], x.index, name=k)
                if overrule_ties:
                    # resolve ties with vote from the highest ranked rater
                    c[ties] = x.loc[ties, rr.index[0]]
                sel = [tuple(v) in counts.index[counts['sel']] for v in cnts]
                c = c[sel]
                cons[k] = c
                contradictory[k] = pd.Series(~np.array([tuple(v) in counts.index[counts['valid']] for v in cnts]), x.index)

        assert trains['Wav'].notna().all()
        # join on columns
        # trains = trains.join(c, on='IDZug', how='outer')    # could lead to nan indices
        trains = trains.reset_index().rename(columns=dict(IDZug='ZUGID')).set_index('ZUGID')
        # trains['Flachstelle'] = c   # uses left join: discards right indices not in left
        trains = pd.concat((trains, cons), join='outer', axis=1)
        # no new NAs should have occurred
        assert trains['Wav'].notna().all()
        contradictory = contradictory[contradictory.any(axis=1)].join(trains['Wav'])
        # improve readability
        lmap_short = dict(zip(np.linspace(0, 1, 3), "-U+"))
        # this .join switches result index to different argument index automatically (why? --> is deprecated)
        #contradictory = contradictory.join(ml[cons.columns].applymap(lambda x: lmap_short[x]), how='left')
        # same result: index without change to multi-index
        x = ml[cons.columns].applymap(lambda x: lmap_short[x][0]).reset_index('WH')
        # but columns get multi-index
        contradictory = pd.concat(dict(contradictory=contradictory), axis=1).join(x, how='left')
        contradictory.set_index('WH', append=True).to_csv(f"ratings_contradictory.csv")
        # only use hard values
        # trains[~trains.iloc[:, -3:].isin([0., 1.])] = np.nan
        trains.iloc[:, -3:] = trains.iloc[:, -3:][trains.iloc[:, -3:].isin([0., 1.])]
        # trains.iloc[:, -3:] = trains.iloc[:, -3:].where(trains.iloc[:, -3:].isin([0., 1.]))
        if wav_relative_to_label_file:
            trains['Wav'] = [data_dir.relative_to(Path.cwd()) / v for v in trains['Wav']]
        print("Saving label CSV file ...")
        # remove rows with empty (all NA) labels
        trains = trains[~trains.loc[:, 'Kreischen':].isna().all(axis=1)]
        trains.loc[:, 'Wav':].to_csv(label_file, index=True)
        print("...completed")
    print("done")
