from pathlib import Path
import pandas as pd
from creds import api_key
from datetime import datetime, timedelta
import requests
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


class MinerMeta:
    def __init__(self):
        self.data = None
        self.ml_ds = None

    def compile_lc_data(
        self, num_days=180, read_csv=False, write_csv=False, coins="ETH"
    ):
        file = Path.cwd() / "lc_data.csv"
        if read_csv is True:
            df = pd.read_csv(file, index_col=0)
            return df
        intervals = ["1d", "1w", "1m", "3m", "6m", "1y", "2y"]
        finish = datetime.now()
        start = finish - timedelta(days=num_days)
        delta = timedelta(hours=720)
        df = pd.DataFrame()
        while finish > start:
            payload = {
                "key": api_key,
                "symbol": coins,
                "change": intervals,
                "data_points": "720",
                # "start": datetime.timestamp(start),
            }

            r = requests.get(
                "https://api.lunarcrush.com/v2?data=assets", params=payload
            )

            data = pd.DataFrame.from_dict(r.json()["data"][0])
            ts = data.timeSeries.to_dict()
            new = pd.DataFrame.from_dict(ts, orient="index")
            new.pop("asset_id")
            new.pop("search_average")
            new["time"] = pd.to_datetime(new["time"], unit="s")
            new.set_index("time", inplace=True)
            new.sort_index(ascending=True, inplace=True)
            new["month"] = [new.index[i].month for i in range(len(new))]
            new["day"] = [new.index[i].day for i in range(len(new))]
            new["hour"] = [new.index[i].hour for i in range(len(new))]
            new.fillna(new.mean(), inplace=True)

            df = pd.concat([df, new])
            start = start + delta

        if write_csv is True:
            df.to_csv(file)
        self.df = df
        return self.df

    def ttsplit_norm(self, df, feature_plot=False):
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set(
            font="Franklin Gothic Book",
            rc={
                "axes.axisbelow": False,
                "axes.edgecolor": "lightgrey",
                "axes.facecolor": "None",
                "axes.grid": False,
                "axes.labelcolor": "dimgrey",
                "axes.spines.right": False,
                "axes.spines.top": False,
                "figure.facecolor": "white",
                "lines.solid_capstyle": "round",
                "patch.edgecolor": "w",
                "patch.force_edgecolor": True,
                "text.color": "dimgrey",
                "xtick.bottom": False,
                "xtick.color": "dimgrey",
                "xtick.direction": "out",
                "xtick.top": False,
                "ytick.color": "dimgrey",
                "ytick.direction": "out",
                "ytick.left": False,
                "ytick.right": False,
            },
        )
        sns.set_context(
            "notebook", rc={"font.size": 16, "axes.titlesize": 20, "axes.labelsize": 18}
        )

        # train_df Test Split
        n = len(df)
        train_df = df[0 : int(n * 0.7)]
        val_df = df[int(n * 0.7) : int(n * 0.9)]
        test_df = df[int(n * 0.9) :]
        # Normalize the Data
        train_df_mean = train_df.mean()
        train_df_std = train_df.std()

        train_df = (train_df - train_df_mean) / train_df_std
        val_df = (val_df - train_df_mean) / train_df_std
        test_df = (test_df - train_df_mean) / train_df_std

        # Create Feature Plot if wanted
        if feature_plot is True:
            df_std = (df - train_df_mean) / train_df_std
            df_std = df_std.melt(var_name="Column", value_name="Normalized")
            plt.figure(figsize=(12, 6))
            ax = sns.violinplot(x="Column", y="Normalized", data=df_std)
            ax.set_xticklabels(df.keys(), rotation=90)
            ax.set_title("Training Data Feature Dist with whole DF Mean")

            return ax

        return train_df, val_df, test_df
