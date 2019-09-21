# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pandas as pd
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews import opts

hv.extension("bokeh")


def loggers2df(loggers):
    dfs = [pd.DataFrame(x.data) for x in loggers]
    return pd.concat(dfs, keys=[x.exp_id for x in loggers], names=["exp_id"])


DEF_STASH = ["case", "time_seconds"]


def stash2df(stash, index=DEF_STASH):
    return pd.DataFrame(stash, columns=list(stash[0].keys())).set_index(index)


def plot_legacy(df, x, y, z):
    fig, ax = plt.subplots(figsize=(8, 6))
    df.groupby(z).plot(x=x, y=y, ax=ax)
    ax.legend(df.index.levels[0].tolist(), loc="lower right")
    return ax


def plot(df, x, y, z, width=600, height=600):
    ds = hv.Dataset(df, [x, y, z])
    grouped = ds.to(hv.Curve, x, y)
    ndoverlay = grouped.overlay(z)
    ndoverlay.opts(opts.Curve(width=width, height=height))
    return ndoverlay
