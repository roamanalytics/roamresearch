import numpy as np
import pandas as pd

from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
outlier = importr('outlierDetection')
anomaly = importr('AnomalyDetection')


def make_time_series(start_dt, end_dt, time_step, functions, random_state=None):
    """
    Sklearn-style dataset creation for time series.
    Allows for arbitrary functions, adding noise, and outlier points.

    Parameters
    ----------
    start_dt : string
        Parsable by numpy
        (e.g., "2000-01-01" or "2000-01-01T00:00:00")
    end_dt : string
        Parsable by numpy
        (e.g., "2000-01-01" or "2000-01-01T00:00:00")
    time_step : string
        Dates: Y - year, M - month, W - week, D - day
        Times: h - hour, m - minute, s - second
               ms - millisecond, us - microsecond, ns - nanosecond,
               ps - picosecond, fs - femtosecond, as - attosecond
        Fully described here http://docs.scipy.org/doc/numpy/reference/arrays.datetime.html#datetime-units
    functions : List of tuples
        As in [(function_A, noise_A, outlier_pct_A),
               (function_B, noise_B, outlier_pct_B)].
        function is a function that takes outputs an array.
        noise is a float.
        outlier_pct is a float.
    random_state : int
        Seed if required.

    Returns
    -------
    (dt, ys)
        The date(time)s `dt` and target(s) `ys`.
    """
    if isinstance(random_state, int):
        np.random.seed(random_state)
    elif random_state is not None:
        raise ValueError('random_state has to be `None` or `int`.')

    dt = np.arange(start_dt, end_dt, dtype="datetime64[{}]".format(time_step))
    rows = len(dt)
    ind = range(rows)

    tmp_ys = []
    for function, noise, outlier_pct in functions:
        # Add noise
        random_noise = np.random.choice([-1, 0, 1], size=rows) * noise
        y = [function(i) + j for i, j in zip(ind, random_noise)]

        # Add outliers
        outlier_total = int(outlier_pct * rows)
        outlier_pts = np.random.choice(ind, size=outlier_total, replace=False)
        for pt in outlier_pts:
            outlier_size = np.random.choice([0.5, 2])
            y[pt] *= outlier_size

        tmp_ys.append(y)

    ys = np.column_stack(tmp_ys)
    return dt, ys


def np_to_df(dt, ys, cols):
    wide_df = pd.DataFrame(ys, index=dt, columns=cols).reset_index().rename(
        columns={'index': 'dt'})
    long_df = pd.melt(wide_df, id_vars=['dt'], var_name='series')
    long_df.to_csv("./data/long_df_{}.csv".format("".join(cols)), index=False)
    return wide_df


def r_stl_sequential_grubbs_test_df(
        df, var_in, var_out, frequency_season,
        max_outlier_pct=0.10, alpha=0.05):
    save_dt = df['dt']
    df = df.drop('dt', axis=1)
    pandas2ri.activate()
    tmp_df = outlier.stl_sequential_grubbs_test_df(
        df=df, var_in=var_in, var_out=var_out,
        frequency_season=frequency_season,
        max_outlier_pct=max_outlier_pct, alpha=alpha)
    out_df = pandas2ri.ri2py(tmp_df)
    pandas2ri.deactivate()
    out_df = pd.concat([save_dt.reset_index(drop=True),
                        out_df.reset_index(drop=True)], axis=1)
    return out_df


def find_outliers_for_examples(df):
    wide_outlier_df = df.copy()
    wide_outlier_df = r_stl_sequential_grubbs_test_df(
        wide_outlier_df, 'A', 'A_outlier', 'hour_day')
    wide_outlier_df = r_stl_sequential_grubbs_test_df(
        wide_outlier_df, 'B', 'B_outlier', 'hour_day')
    wide_outlier_df = r_stl_sequential_grubbs_test_df(
        wide_outlier_df, 'C', 'C_outlier', 'hour_day')
    wide_outlier_df.to_csv("./data/wide_outlier_df_ABC.csv", index=False)
    return wide_outlier_df


def find_outliers_for_benchmarks(df):
    wide_outlier_df = df.copy()
    wide_outlier_df = r_stl_sequential_grubbs_test_df(
        wide_outlier_df, 'A', 'A_outlier', 'minute_day')
    return wide_outlier_df


def r_anomaly_detection(df, period, max_outlier_pct=0.10, alpha=0.05):
    save_dt = df['dt']
    df = df.drop('dt', axis=1)
    pandas2ri.activate()
    tmp_res = anomaly.AnomalyDetectionVec(
        x=df, direction='both', period=period,
        max_anoms=max_outlier_pct, alpha=alpha)
    res = tmp_res[0]
    out_df = pandas2ri.ri2py(res)
    pandas2ri.deactivate()
    out_df['index'] = out_df['index'] - 1
    out_df = out_df.set_index(['index'])
    out_df = pd.concat([save_dt.reset_index(drop=True), df, out_df], axis=1)
    return out_df


def find_anomalies_for_benchmarks(df):
    wide_outlier_df = df.copy()
    wide_outlier_df = r_anomaly_detection(df, period=1440)
    return wide_outlier_df
