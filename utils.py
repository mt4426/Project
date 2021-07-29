import numpy as np


def calc_leq(data):
    return 10 * np.log10(np.mean(10 ** (data / 10)))


def calcl90(data):
    stat_percentile = 100 - 90
    return np.nanpercentile(data, stat_percentile)

def calcl10(data):
    stat_percentile = 100 - 10
    return np.nanpercentile(data, stat_percentile)

def calcl5(data):
    stat_percentile = 100 - 5
    return np.nanpercentile(data, stat_percentile)

def calcl1(data):
    stat_percentile = 100 - 1
    return np.nanpercentile(data, stat_percentile)
