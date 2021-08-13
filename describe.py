import sys
import pandas as pd
import numpy as np
from dataset import Dataset
import math


def none_handler(func):
    def wrap(data, *args, **kwargs):
        if len(data) == 0:
            return None
        return func(data, *args, **kwargs)

    return wrap


def sort_handler(func):
    return lambda data, *args, **kwargs: func(sort_data(data), *args, **kwargs)


def calc_mean(data):
    total = 0
    for x in data:
        if np.isnan(x):
            continue
        total = total + x
    return total / len(data)


def calc_std(data):
    mean = calc_mean(data)
    total = 0
    for x in data:
        if np.isnan(x):
            continue
        total = total + (x - mean) ** 2
    return (total / len(data)) ** 0.5


@none_handler
def find_max_min(data, comparer):
    m = data[0]
    for t in data[1:]:
        if comparer(t, m):
            m = t
    return m


def minimum(data):
    return find_max_min(data, lambda x, y: x < y)


def maximum(data):
    return find_max_min(data, lambda x, y: x > y)


def sort_data(data):
    if len(data) < 2:
        return data
    data = list(data)
    pivot = data[0]
    body = data[1:]
    return (sort_data([x for x in body if x < pivot])
            + [pivot]
            + sort_data([x for x in body if x >= pivot]))


@none_handler
@sort_handler
def quantile25(data):
    return data[len(data) // 4]


@none_handler
@sort_handler
def quantile50(data):
    return data[len(data) // 2]


@none_handler
@sort_handler
def quantile75(data):
    return data[3 * (len(data) // 4)]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Как юзать: script.py dataset.csv")
        exit()
    dataset = Dataset(sys.argv[1])
    result_df = pd.DataFrame(
        dtype=np.float64,
        columns=[c for c, t in zip(dataset.df.columns, dataset.df.dtypes) if t == np.float64],
        index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    )
    for column in result_df.columns:
        result_df.loc['count', column] = len(dataset.df[column])
        result_df.loc['mean', column] = calc_mean(dataset.df[column])
        result_df.loc['std', column] = calc_std(dataset.df[column])
        result_df.loc['min', column] = minimum(dataset.df[column])
        result_df.loc['25%', column] = quantile25(dataset.df[column])
        result_df.loc['50%', column] = quantile50(dataset.df[column])
        result_df.loc['75%', column] = quantile75(dataset.df[column])
        result_df.loc['max', column] = maximum(dataset.df[column])
    print(result_df)
