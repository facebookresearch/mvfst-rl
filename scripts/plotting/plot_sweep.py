# -*- coding: utf-8 -*-
# + {}
# %env HV_DOC_HTML=false
# %matplotlib inline

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# To install:
#   conda install -c pyviz holoviews bokeh
#   conda install nodejs
#   pip install jupyterlab
#   pip install jupytext
#   jupyter labextension install jupyterlab-jupytext

# To run (from current dir):
#   jupyter lab --port 8081

import functools
import json
import pathlib
import typing
import collections
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # noqa: F401
import matplotlib.cbook
import holoviews as hv
from holoviews import opts

hv.extension("bokeh")
hv.notebook_extension(width=90)  # For showing wide plots

# Get rid of matplotlib deprecation warnings.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


# +
def load_config(path: str) -> typing.Dict:
    """Loads run metadata."""
    with open(f"{path}/meta.json", "r") as f:
        try:
            config = json.load(f)
            return config
        except Exception as e:  # noqa: F841
            print(f"Error parsing {path}")


RunLog = typing.Dict[str, pd.DataFrame]


def load_logs(path: str) -> RunLog:
    """Loads run logs as pandas.DataFrame."""
    logs = dict(df=pd.read_csv(f"{path}/logs.csv").dropna())
    return path, logs


def load_configs(paths: typing.Iterable, config_filter: typing.Callable) -> typing.Dict:
    """Finds all configs recursively and reads their content."""
    configs = {}
    for path in paths:
        p = pathlib.Path(path)
        for file_path in p.rglob("meta.json"):
            config = load_config(file_path.parent)
            if config is not None and config_filter(config):
                configs[file_path.parent] = config
    print(f"Loaded {len(configs)} configs.")
    return configs


def find_args_diff(configs: typing.Dict) -> typing.Set[str]:
    """Finds parameters that are instantiate differently across runs."""
    # gathering args and adding _path to simplify data structure
    lconfigs = [
        {**{"_path": path}, **config["args"]} for path, config in configs.items()
    ]
    # in reality we would want the symmetric difference of all sets...
    lconfigs = [set(c.items()) for c in lconfigs]
    fixed_args = [a[0] for a in set.intersection(*lconfigs)]
    sweep_args = []
    for c in lconfigs:
        for a in c:
            a_name = a[0]
            if a_name in ["_path", "xpid"]:
                continue
            if a_name in fixed_args:
                continue
            sweep_args.append(a_name)
    return set(sweep_args)


RunPivot = str
ExpID = str


def cluster_runs(
    configs: typing.Dict, xp_pivot: typing.Callable, run_pivot: typing.Callable
) -> typing.Dict[ExpID, typing.Dict[RunPivot, typing.List]]:
    """Cluster runs by given experiment and pivot key."""
    # we want dict(experiment_id=dict(split_key=[run])
    new_configs = collections.defaultdict(
        functools.partial(collections.defaultdict, list)
    )
    for path, config in configs.items():
        experiment_id = xp_pivot(config)
        split_key = run_pivot(config)
        new_configs[experiment_id][split_key].append(path)
    for k in new_configs:
        new_configs[k] = dict(new_configs[k])
    return dict(new_configs)


def print_tree(nconfigs):
    """Prints the structure of the clustered runs."""
    for facet, split in nconfigs.items():
        print(facet)
        for k, runs in split.items():
            print(f"└── {k}: x{len(runs)}")


# NOTE str == 'df'
Metrics = typing.Dict[ExpID, typing.Dict[RunPivot, RunLog]]


def get_all_logs(nconfigs, experiment_id, lock=threading.Lock()):  # noqa: B008
    with lock:
        print(f"Collecting metrics for {experiment_id}")
    split = nconfigs[experiment_id]
    xp_logs = collections.defaultdict(dict)
    for split_key, runs in split.items():
        pool = ThreadPoolExecutor(min(10, len(runs)))
        metrics = pool.map(load_logs, runs)
        for r, m in metrics:
            xp_logs[split_key][r] = m
    return experiment_id, xp_logs


def load_metrics(nconfigs) -> Metrics:
    """Retrieves metrics into a clustered format (given clustered runs)."""
    experiments = collections.defaultdict(
        functools.partial(collections.defaultdict, dict)
    )
    pool = ThreadPoolExecutor(min(10, len(nconfigs)))
    pget = functools.partial(get_all_logs, nconfigs)
    xplogs = pool.map(pget, list(nconfigs.keys()))
    for xpid, logs in xplogs:
        experiments[xpid] = dict(logs)
    return dict(experiments)


def load_experiments(
    paths: typing.List[str],
    config_filter: typing.Callable = lambda _: True,
    xp_pivot: typing.Callable = None,
    run_pivot: typing.Callable = lambda _: "mvfstrl",
) -> Metrics:
    """Retrieves and clusters experiment data.

    By default, it'll attempt to discover sweep parameters and cluster
    experiments around them.
    """
    configs = load_configs(paths, config_filter)
    if xp_pivot is None:
        sweep_args = find_args_diff(configs)
        print(f"Found the following sweep parameters: {sweep_args}.")

        def auto_pivot(config):
            return "|".join(
                [f"{k}{v}" for k, v in config["args"].items() if k in sweep_args]
            )

        xp_pivot = auto_pivot
    nc = cluster_runs(configs, xp_pivot, run_pivot)
    print_tree(nc)
    return load_metrics(nc)


# -

# ## Plotting


# +
def plot_run(
    df,
    x="step",
    y="mean_episode_return",
    model="model",
    color="#ff0000",
    subsample=1000,
):
    df = df[[x, y]].dropna()
    df[y] = df[y].rolling(100, min_periods=0).mean()
    grid = np.linspace(0, df[x].max(), subsample)
    yvalues = np.interp(grid, df[x].values, df[y].values)
    df = pd.DataFrame({x: grid, y: yvalues})
    p = hv.Curve(hv.Dataset(df, kdims=[x], vdims=[y]))
    p.opts(opts.Curve("Curve", color=color))
    return p


def plot_model(
    runs,
    x="step",
    y="mean_episode_return",
    model="model",
    color="#ff0000",
    subsample=1000,
):
    hmap = {}

    # Interpolate the data on an even grid. With min(np.amin(...))
    # this starts where the first data starts and ends where the last
    # data ends. Interpolation of missing data will create artefacts.
    # An alternative would be to throw away data and the end and do
    # max(np.amin(...)) etc.
    xmin = min(np.amin(config["df"][x].values) for _, config in runs.items())
    xmax = max(np.amax(config["df"][x].values) for _, config in runs.items())
    xnum = max(len(config["df"][x]) for _, config in runs.items())

    grid = np.linspace(xmin, xmax, xnum)

    for run, config in runs.items():
        df = config["df"]
        yvalues = np.interp(grid, df[x].values, df[y].values)
        df = pd.DataFrame({x: grid, y: yvalues})
        p = plot_run(df, x, y, model, color, subsample)
        p.opts(opts.Curve(f"Curve", color=color, alpha=0.2))
        hmap[run] = p

    hmap = hv.HoloMap(hmap)
    p_runs = hmap.overlay().relabel("Runs")

    hmap_mean = hv.HoloMap(hmap)
    p_mean = hmap_mean.collapse(function=np.mean)
    p_mean = hv.Curve(p_mean).relabel(model)
    p_mean.opts(opts.Curve("Curve", color=color))

    p = p_runs * p_mean
    # p = p_runs * p_mean * p_std

    # Plot options
    p.opts(opts.NdOverlay("NdOverlay.Runs", show_legend=False))
    return p


def plot_facet(
    models,
    x="step",
    y="mean_episode_return",
    palette="Set1",
    model2color=None,
    kdims=["lstm"],
    subsample=1000,
):
    if model2color is None:
        colors = hv.Palette(palette).values
        colors = list(set(colors))
        model2color = {}
    hmap = {}
    for model, runs in models.items():
        if model not in model2color:
            model2color[model] = colors[len(model2color) - 1]
        color = model2color[model]
        hmap[model] = plot_model(runs, x, y, model, color, subsample)
    p = hv.HoloMap(hmap, kdims=kdims)
    p = p.overlay()
    return p


def plot(
    metrics,
    x="step",
    y="mean_episode_return",
    palette="Set1",
    kdims=["lr"],
    kdims_facet=["lstm"],
    subsample=1000,
    cols=3,
):
    hmap = {}
    model2color = {}
    colors = hv.Palette(palette).values
    colors = list(set(colors))
    for _facet, models in metrics.items():
        for model, _ in models.items():
            if model not in model2color:
                model2color[model] = colors[len(model2color) - 1]

    for facet, models in metrics.items():
        p = plot_facet(
            models, x, y, palette, model2color, kdims=kdims_facet, subsample=subsample
        )
        if facet not in hmap:
            hmap[facet] = {}
        hmap[facet] = p
    p = hv.HoloMap(hmap, kdims=kdims)
    p = hv.NdLayout(p).cols(cols)
    return p


def augment(data, results, x="step", y="mean_episode_return"):
    boundaries = {}
    for env, models in data.items():
        for _model, runs in models.items():
            xmin = min(np.amin(config["df"][x].values) for _, config in runs.items())
            xmax = max(np.amax(config["df"][x].values) for _, config in runs.items())
            if env not in boundaries:
                boundaries[env] = {"xmin": xmin, "xmax": xmax}
            if xmin < boundaries[env]["xmin"]:
                boundaries[env]["xmin"] = xmin
            if xmax > boundaries[env]["xmax"]:
                boundaries[env]["xmax"] = xmax

    for env, models in results.items():
        for model, result in models.items():
            data[env].update(
                {
                    model: {
                        "published_results": {
                            "df": pd.DataFrame(
                                {
                                    x: [
                                        boundaries[env]["xmin"],
                                        boundaries[env]["xmax"],
                                    ],
                                    y: [result, result],
                                }
                            )
                        }
                    }
                }
            )


# -

# ## Load data

# +
def get_paths(pattern):
    logdirs = glob.glob("/checkpoint/viswanath/mvfstrl/{}".format(pattern))
    all_paths = [os.path.join(logdir, "train/torchbeast/latest") for logdir in logdirs]

    paths = []
    for path in all_paths:
        logf = os.path.join(path, "logs.csv")
        if not os.path.exists(logf):
            print(f"logs.csv not found, skipping {path}")
            continue
        if os.stat(logf).st_size == 0:
            print(f"Empty logs.csv, skipping {path}")
            continue
        paths.append(path)
    return paths


pattern = "*19-09-03_09-11-19-133759*"

xp_filter = lambda c: c["args"]["cc_env_time_window_ms"] == 100
experiment_pivot = lambda c: (
    "rdelay-%s" % c["args"]["cc_env_reward_delay_factor"],
    "rlost-%s" % c["args"]["cc_env_reward_packet_loss_factor"],
)
cluster_by = lambda c: "actions-%s" % c["args"]["num_actions"]

kdims = ["rd", "rl"]
kdims_facet = ["actions"]

paths = get_paths(pattern)
data = load_experiments(paths, xp_filter, experiment_pivot, cluster_by)

y = "mean_episode_return"
# y = 'mean_episode_step'
# y = 'total_loss'
# y = 'pg_loss'
# y = 'baseline_loss'
# y = 'entropy_loss'
# y = 'learner_queue_size'

# %%time
# %%opts Curve {+axiswise}
# %%opts NdOverlay [legend_position='top_left']
plot(
    data,
    y=y,
    cols=4,
    palette="Colorblind",
    kdims=kdims,
    kdims_facet=kdims_facet,
    subsample=100,
)
