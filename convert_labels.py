from pathlib import Path

import pandas as pd
import numpy as np
import torchaudio
from pprint import pprint
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# data dir for this project
prj_root = Path.home() / "prj"
acr_dir = prj_root / "acrDb"


if __name__ == "__main__":
    data_dir = acr_dir
    #data_dir = acr_dir
    #data_dirs = [v for v in acr_dir.glob("*") if v.is_dir()]
    #files = [list(data_dir.glob("*.wav")) for data_dir in data_dirs]
    files = list(data_dir.glob("**/*.wav"))

    # get metadata, takes a lot of time
    print("Reading audio metadata ...")
    info = {k.stem[:-3]: vars(torchaudio.info(k)) for k in tqdm(files) if k.stem.endswith("01")}
    info = pd.DataFrame(info).T.infer_objects()
    info["dur"] = info["num_frames"]/info["sample_rate"]
    print("First file metadata:")
    print(info.iloc[0])

    # make sure ids are unique, to not lose files which may overlap in different subdirs
    assert len(files) == len({v.stem for v in files})
    wav_ids = {str(v.stem).rsplit("_", 1)[0]: v.parts[-2] for v in files}

    cols = ["FileDatum", "FileZeit", "MkID", "Rohdaten", "Tag"]
    print("Opening table file...")
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

    db_ids_df["dur"] = [(info.loc[k, "dur"] if k in info.index else None) for k in db_ids_df["ID"]]
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

    #axs[1, 0].hist(db_ids_df["dur"], bins=40)
    #axs[1, 0].set_title("No duration found", backgroundcolor="white")
    axs[1, 0].text(x=0.5, y=0.5, s="No duration found", ha="center", transform=axs[1, 0].transAxes)
    axs[1, 1].set_xlabel("audio duration [s]")
    axs[1, 1].hist(db_df["dur"], bins=np.arange(0, max(db_df["dur"]) + 1, 2))

    plt.show()

    print("done")
