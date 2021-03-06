from pathlib import Path
import pandas as pd
from creds import api_key
from datetime import datetime, timedelta
import requests
import xgboost as xgb
import ta
import pandas as pd

pd.plotting.register_matplotlib_converters()
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


class MinerMeta:
    def __init__(self):
        self.df = self.compile_lc_data(read_csv=True)
        self.train_mean = None
        self.x_train = None
        self.y_train = None
        self.xg_model = None
        self.pred_df = None

    def compile_lc_data(
        self, num_days=180, read_csv=False, write_csv=False, coins="ETH"
    ):
        file = Path.cwd() / "lc_data.csv"
        if read_csv is True:
            self.df = pd.read_csv(file, index_col=0)
            self.df.index = pd.to_datetime(self.df.index)
            return self.df
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

        df = ta.add_all_ta_features(
            df,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            fillna=True,
        )

        if write_csv is True:
            df.to_csv(file)
        self.df = df
        return self.df

    def ttsplit_norm(self):
        # train Test Split
        n = len(self.df)
        # self.val_df = self.df[int(n * 0.7) : int(n * 0.9)]
        self.train = self.df[0 : int(n * 0.7)]
        self.test = self.df[int(n * 0.7) :]
        # Normalize the Data
        self.train_mean = self.train.mean()
        self.train_std = self.train.std()

        self.x = self.train.loc[:, self.train.columns != "close"]
        self.y = self.train.close

        self.x_test = self.test.loc[:, self.test.columns != "close"]
        self.y_test = self.test.close

        # self.train = (train - self.train_mean) / self.train_std
        # self.val_df = (val_df - self.train_mean) / self.train_std
        # self.test_df = (test_df - self.train_mean) / self.train_std

        # self.x_train = self.train.copy()
        # self.y_train = self.x_train.pop("close")

        # self.x_val = self.val_df.copy()
        # self.y_val = self.x_val.pop("close")

        # self.x_test = self.test_df.copy()
        # self.y_test = self.x_test.pop("close")

    def build_xgb(self):
        if self.x_train is None or self.y_train is None:
            self.ttsplit_norm()

        # dtrain = xgb.DMatrix(self.x_train, label=self.y_train)
        # dtest = xgb.DMatrix(self.x_test, label=self.y_test)
        dtrain = xgb.DMatrix(self.x, label=self.y)
        dtest = xgb.DMatrix(self.x_test, label=self.y_test)

        # param = {
        #     "max_depth": 100,
        #     "eta": 1,
        #     "objective": "reg:squarederror",
        #     "booster": "gbtree",
        # }
        param = {
            "colsample_bynode": 0.8,
            "learning_rate": 1,
            "max_depth": 6,
            "num_parallel_tree": 110,
            "objective": "reg:squarederror",
            "subsample": 0.8,
            "tree_method": "gpu_hist",
        }
        # param["eval_metric"] = ["auc", "ams@0"]
        evallist = [(dtest, "eval"), (dtrain, "train")]

        num_round = 2
        self.xg_model = xgb.train(param, dtrain, num_round, evals=evallist)

    def predict_and_plot(self, plot=True):
        df_features = self.df.loc[:, self.df.columns != "close"]
        # self.test_predictions["xgb"] = self.xg_model.predict(
        #     xgb.DMatrix(tmp, label=self.df.close))
        self.df["predictions"] = self.xg_model.predict(
            xgb.DMatrix(df_features, label=self.df.close)
        )
        self.pred_df = self.df.copy()
        if plot:

            plt.figure(figsize=(15, 10))
            plt.scatter(
                x=self.df.index,
                y=self.df.close,
                color="r",
                marker=".",
                label="real data",
            )
            plt.scatter(
                x=self.df.index,
                y=self.df.predictions,
                marker="X",
                label="predictions",
            )
            plt.xlabel("time")
            plt.ylabel("price")
            plt.title("Red is predictions, Blue is real data")
            plt.show()

    def plot_feature_dist(self):
        if self.train_mean is None or self.train is None:
            self.ttsplit_norm()
        # Creates Feature Plot of main DF keys compared to train mean and train std
        df_std = (self.df - self.train_mean) / self.train_std
        df_std = df_std.melt(var_name="Columns", value_name="Normalized")
        plt.figure(figsize=(15, 9))
        ax = sns.violinplot(x="Columns", y="Normalized", data=df_std)
        ax.set_xticklabels(self.df.keys(), rotation=90)
        ax.set_title("Training Data Feature Dist with whole DF Mean")

    def plot_day(self, num_days: int):
        if self.pred_df is None:
            self.predict_and_plot(plot=False)
        tmp = self.pred_df.copy()
        tmp.index = pd.to_datetime(tmp.index)
        start = tmp.index.max() - timedelta(days=num_days)
        tmp = tmp[tmp.index >= start]
        self.num_days = num_days
        tmp["dif"] = tmp.close - tmp.predictions
        avg_diff = tmp.dif.mean()
        print("Average Difference of Prediction from Close:", avg_diff)
        tmp.plot(y=["predictions", "close"], use_index=True, figsize=(15, 9))

    def frequency_plots(self):
        self.df.plot.hist(y=["close"], use_index=True, figsize=(15, 9))
        self.df.plot.hist(y=["predictions"], use_index=True, figsize=(15, 9))
        from statsmodels.tsa.stattools import adfuller

        x = self.df.close.values
        result = adfuller(x)
        print("AD Fuller Tests of Close")
        print("ADF Statistic: %f" % result[0])
        print("p-value: %f" % result[1])
        print("Critical Values:")
        for key, value in result[4].items():
            print("\t%s: %.3f" % (key, value))

    def feature_importance(self):
        if self.xg_model is None:
            self.build_xgb()
        from xgboost import plot_importance

        figsize = (15, 9)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        return plot_importance(self.xg_model, ax=ax)

    # def deploy_sagemaker(self):
    #     boto3.Session().resource("s3").Bucket(bucket_name).Object(
    #         os.path.join(prefix, "train/train.csv")
    #     ).upload_file("train.csv")
    #     s3_input_train = sagemaker.inputs.TrainingInput(
    #         s3_data="s3://{}/{}/train".format(bucket_name, prefix), content_type="csv"
    #     )
    #     sess = sagemaker.Session()
    #     xgb = sagemaker.estimator.Estimator(
    #         xgboost_container,
    #         role,
    #         instance_count=1,
    #         instance_type="ml.m4.xlarge",
    #         output_path="s3://{}/{}/output".format(bucket_name, prefix),
    #         sagemaker_session=sess,
    #     )
    #     xgb.set_hyperparameters(
    #         max_depth=5,
    #         eta=0.2,
    #         gamma=4,
    #         min_child_weight=6,
    #         subsample=0.8,
    #         silent=0,
    #         objective="binary:logistic",
    #         num_round=100,
    #     )
    #     xgb.fit({"train": s3_input_train})
