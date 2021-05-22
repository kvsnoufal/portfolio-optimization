import pandas as pd
FILE = "cleaned.csv"
COINS = ["DASH","LTC","STR"]
COLS = ['high', 'low', 'open', 'close', 'volume', 'quoteVolume','weightedAverage']
SCOLS = ["vh","vl","vc","open_s","volume_s","quoteVolume_s","weightedAverage_s"]
OBS_COLS = ['vh', 'vl', 'vc', 'open_s', 'volume_s', 'quoteVolume_s', 'weightedAverage_s', 'vh_roll_7', 'vh_roll_14', 'vh_roll_30', 'vl_roll_7', 'vl_roll_14', 'vl_roll_30', 'vc_roll_7', 'vc_roll_14', 'vc_roll_30', 'open_s_roll_7', 'open_s_roll_14', 'open_s_roll_30', 'volume_s_roll_7', 'volume_s_roll_14', 'volume_s_roll_30', 'quoteVolume_s_roll_7', 'quoteVolume_s_roll_14', 'quoteVolume_s_roll_30', 'weightedAverage_s_roll_7', 'weightedAverage_s_roll_14', 'weightedAverage_s_roll_30']
EPISODE_LENGTH = 500


df = pd.read_csv(FILE)

        
df["date"] = df["date"].apply(lambda x: pd.Timestamp(x, unit='s', tz='US/Pacific'))
df = df[df["coin"].isin(COINS)].sort_values("date")
df["vh"] = df["high"]/df["open"]
df["vl"] = df["low"]/df["open"]
df["vc"] = df["close"]/df["open"]
df["open_s"] = df.groupby("coin")["open"].apply(lambda x: x - x.shift(1))
df["volume_s"] = df.groupby("coin")["volume"].apply(lambda x: x - x.shift(1))
df["quoteVolume_s"] = df.groupby("coin")["quoteVolume"].apply(lambda x: x - x.shift(1))
df["weightedAverage_s"] = df.groupby("coin")["weightedAverage"].apply(lambda x: x - x.shift(1))

new_cols = []

for col in SCOLS:
    print(col)
    df[col+"_roll_7"] = df.groupby("coin")[col].apply(lambda x: x.rolling(7).mean().bfill())
    new_cols.append(col+"_roll_7")
    df[col+"_roll_14"] = df.groupby("coin")[col].apply(lambda x: x.rolling(14).mean().bfill())
    new_cols.append(col+"_roll_14")
    df[col+"_roll_30"] = df.groupby("coin")[col].apply(lambda x: x.rolling(30).mean().bfill())
    new_cols.append(col+"_roll_30")
    
SCOLS.extend(new_cols)
print(SCOLS)

df.to_csv("cleaned_preprocessed.csv")
