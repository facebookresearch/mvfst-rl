# -*- coding: utf-8 -*-
# +
# %env HV_DOC_HTML=false
# %matplotlib inline

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Use the commands below to install. Note that, as of 2021-01-04, the latest
# jupyterlab (3.0.0) is not compatible with jupytext.

#   conda install bokeh holoviews matplotlib pandas
#   conda install -c conda-forge "jupyterlab<3" jupytext "nodejs>=12"
#   jupyter lab build

# To run (from current dir):
#   jupyter lab --port 8081

import copy
import errno
import functools
import getpass
import hiplot
import itertools
import json
import typing
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings

from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

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
# -
# Config settings to ignore, when identifying which settings were varied across several experiments.
IGNORE_IN_DIFF = {'base_logdir', 'checkpoint', 'logdir', 'traced_model'}
# Maximum number of parallel experiment loaders.
MAX_XP_LOADERS = 10
# Maximum number of parallel run loaders within each experiment.
MAX_RUN_LOADERS = 10
# Rolling window for smoothing plots
ROLLING_WINDOW = 1000
# Shorter names to make output more readable
SHORT_NAMES = {
    "cc_env_reward_delay_log_offset": "dlo",
    "cc_env_reward_throughput_log_offset": "tlo",
    "end_of_episode_bootstrap": "boot",
    "learning_rate": "lr",
    "mean_avg_reward": "reward",
    "mean_episode_return": "return",
    "mean_episode_step": "steps",
    "mean_cwnd_mean": "cwnd",
    "mean_delay_mean": "dl",
    "mean_throughput_mean": "th",
    "reward_normalization": "rn",
    "reward_normalization_coeff": "rnc",
    "reward_normalization_stats_per_job": "rnj",
    "throughput_over_delay": "th/dl",
    "use_job_id_in_actor": "ja",
    "use_job_id_in_critic": "jc",
}
LONG_NAMES = {v: k for k, v in SHORT_NAMES.items()}  # inverse mapping


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


def load_logs(path_and_config: Tuple[Path, Any], split_by_job_id: bool = False, actors: str = "all") -> RunLog:
    """Loads run logs as pandas.DataFrame."""
    path, config = path_and_config
    df = pd.read_csv(f"{path}/logs.tsv", sep="\t")

    # Check that we have data from all actors and that it is sufficiently balanced.
    num_actors = config["flags"]["num_actors"]
    counts = df["actor_id"].groupby(df.actor_id).count()
    actor_ids = set(counts.index)
    missing_actors = [i for i in range(num_actors) if i not in actor_ids]
    if missing_actors:
        print(f"WARNING: no data from actors: {sorted(missing_actors)}")
    median_count = counts.median()
    not_enough = counts[counts < median_count * 0.8]
    if not not_enough.empty:
        print(f"WARNING: some actors have fewer episodes than others:\n{not_enough}")
    too_many = counts[counts > median_count * 1.2]
    if not too_many.empty:
        print(f"WARNING: some actors have more episodes than others:\n{too_many}")

    if actors != "all":
        df = df[df.job_type == actors]

    if split_by_job_id:
        all_job_ids = sorted(df.job_id.unique())
        dfs = [(job_id, df[df.job_id == job_id]) for job_id in all_job_ids]
    else:
        dfs = [(-1, df)]

    all_logs = []
    for job_id, df in dfs:
        # Compute the average reward over the episode. This is more informative than the
        # total return, which depends a lot on the number of steps.
        df["avg_reward"] = df["episode_return"] / df["episode_step"]

        # Compute mean/std stats at each learning step.
        mean_fields = ["avg_reward", "episode_step", "episode_return", "loss", "cwnd_mean", "delay_mean", "throughput_mean"]
        mean_df = df[["step"] + mean_fields].groupby("step").mean()
        mean_df.rename(columns={k: f"mean_{k}" for k in mean_fields}, inplace=True)
        mean_df.reset_index(inplace=True)  # set `step` back as column instead of index
         
        std_fields = ["cwnd_mean"]
        std_df = df[["step"] + std_fields].groupby("step").std()
        std_df.rename(columns={k: f"std_{k}" for k in std_fields}, inplace=True)
        std_df.reset_index(drop=True, inplace=True)  # get rid of `step`

        # Final dataframe concatenates mean + std stats.
        df = pd.concat([mean_df, std_df], axis=1)
        
        # Add a field with log(throughput / delay).
        df["log_th/dl"] = df["mean_throughput_mean"] / df["mean_delay_mean"]
        
        logs = dict(df=df, config=config)
        all_logs.append((job_id, path, logs))

    return all_logs

def load_configs(paths: Iterable, config_filter: Optional[Callable] = None) -> Dict:
    """Finds all configs recursively and reads their content."""
    configs = {}
    for path in paths:
        try:
            file_paths = list(path.rglob("meta.json"))
        except FileNotFoundError as exc:
            missing = Path(exc.filename).name
            if missing.startswith("train_tid"):
                pass  # expected: these are deleted during experiments
            else:
                # If this code is reached, see if other files need handling.
                assert False, (exc.args, exc.filename, exc.filename2)
        except OSError as exc:
            if exc.errno == errno.ESTALE:
                # This may happen when an experiment is still running.
                print(f"Ignoring due to stale file handle: {path}")
                continue
            # If this code is reached, see if there are other error codes that may be ignored.
            assert False, (str(exc), exc.errno)
        for file_path in file_paths:
            config = load_config(file_path.parent)
            if config is not None and (config_filter is None or config_filter(config)):
                configs[file_path.parent] = config
    cr = "\n  "
    # Show max 10 configs
    configs_to_show = list(map(str, sorted(configs)))
    if len(configs_to_show) > 10:
        configs_to_show = configs_to_show[0:5] + ["..."] + configs_to_show[-5:]
    print(f"Loaded {len(configs)} configs:{cr}{cr.join(configs_to_show)}")
    return configs


def find_args_diff(configs: Dict) -> Set[str]:
    """Find parameters that are instantiated differently across runs."""
    # Obtain experiment settings.
    lconfigs = [flags_to_set(config["flags"]) for path, config in configs.items()]
    # Ideally we would want the symmetric difference of all sets, but that is not in Python.
    fixed_args = [a[0] for a in set.intersection(*lconfigs)]
    return set(a for c in lconfigs for a, v in c
              if a not in IGNORE_IN_DIFF and a not in fixed_args)


def make_hashable(obj):
    """Return a hashable version of `item`"""
    # Is it already hashable?
    try:
        hash(obj)
    except TypeError:
        pass
    else:
        return obj
    if isinstance(obj, dict):
        return tuple((k, make_hashable(v)) for k, v in obj.items())
    elif isinstance(obj, list):
        return tuple(make_hashable(item) for item in obj)
    else:
        raise NotImplementedError(type(obj))


def flags_to_set(flags: Dict) -> Set:
    """Turn a dictionary of settings into a set of unique (key, value) pairs"""
    # The tricky part is that some flags may be lists, that are not hashable.
    # We convert them to tuples here.
    return set((k, make_hashable(v)) for k, v in flags.items())


RunPivot = str
ExpID = str


def cluster_runs(
    configs: Dict, xp_pivot: Callable, run_pivot: Optional[Union[Callable, dict]] = None
) -> Dict[ExpID, Dict[RunPivot, List]]:
    """
    Cluster runs by given experiment and pivot key.

    :param run_pivot: One of:
        - `None` -> auto-detect parameters that were varied
        - dict -> auto-detect parameters that were varied, but ignore those listed in
            the key "ignore" of this dict (if it exists)
        - callable -> called to obtain the run's key
    """
    if run_pivot is None:
        run_pivot = {}
    # Group runs by experiment ID.
    xp_key = lambda path_config: xp_pivot(path_config[1])
    configs_by_id = groupby(sorted(configs.items(), key=xp_key), key=xp_key)

    # Process experiments.
    new_configs = {}
    for experiment_id, path_configs in configs_by_id:
        path_configs = list(path_configs)

        if isinstance(run_pivot, dict):
            # Auto-detect which parameters were varied in the sweep, and use them as labels.
            sweep_args = find_args_diff(dict(path_configs))

            # Filter out parameters we want to ignore.
            assert all(k in ["ignore"] for k in run_pivot)  # validate dict keys (currently only "ignore" is supported)
            to_ignore = run_pivot.get("ignore", set())
            to_ignore = {to_ignore} if isinstance(to_ignore, str) else set(to_ignore)
            sweep_args = sweep_args - to_ignore

            print(f"Found the following sweep parameters for experiment `{experiment_id}`: {sweep_args}.")

            def auto_pivot(config):
                return tuple(
                    f"{SHORT_NAMES.get(k, k)}={v}" for k, v in sorted(config["flags"].items()) if k in sweep_args
                )

            run_pivot_for_exp = auto_pivot
        else:
            run_pivot_for_exp = run_pivot

        # Group configs within each run based on their run ID.
        run_key = lambda path_config: run_pivot_for_exp(path_config[1])
        new_configs[experiment_id] = {run_id: list(pcs) for run_id, pcs in groupby(sorted(path_configs, key=run_key), key=run_key)}

    # [experiment_id][run_id] -> list of (path, config) pairs
    return new_configs


def print_tree(nconfigs):
    """Prints the structure of the clustered runs."""
    for facet, split in nconfigs.items():
        print(facet)
        for k, runs in split.items():
            print(f"└── {k}: x{len(runs)}")


# NOTE str == 'df'
Metrics = typing.Dict[ExpID, typing.Dict[RunPivot, RunLog]]


def get_all_logs(nconfigs, experiment_id, lock=threading.Lock(), split_by_job_id=False, actors="all"):  # noqa: B008
    with lock:
        print(f"Collecting metrics for {experiment_id}")
    split = nconfigs[experiment_id]
    # The first key in `xp_logs` is used to differentiate jobs when `split_by_job_id` is True.
    xp_logs = defaultdict(lambda: defaultdict(dict))
    for split_key, runs in split.items():
        pool = ThreadPoolExecutor(min(MAX_RUN_LOADERS, len(runs)))
        log_loader = functools.partial(load_logs, split_by_job_id=split_by_job_id, actors=actors)
        metrics = pool.map(log_loader, runs)
        for metrics_per_job in metrics:
            for job_id, r, m in metrics_per_job:
                xp_logs[job_id][split_key][r] = m
    # Convert to regular dict.
    xp_logs_dict = {k: dict(v) for k, v in xp_logs.items()}
    return experiment_id, xp_logs_dict


def get_kdims(key):
    """
    Extract the relevant keywords from a sample `key`.

    `key` may either be a string, or a tuple of strings. Each of these strings should
    be of the form "option=value", and we return the list of all options.

    key=`None` is also accepted, in which case we simply return ["exp"].
    """
    if key is None:
        return ["exp"]
    if isinstance(key, str):
        key = (key,)
    return [item.split("=", 1)[0] for item in key]


def get_label(key):
    """
    Return a single string representing the given key (which has same format as in `get_kdims()`).
    """
    if key is None:
        return "exp"
    elif isinstance(key, str):
        return key
    else:
        return "|".join(key)


def load_metrics(nconfigs, split_by_job_id: bool = False, actors="all") -> Metrics:
    """Retrieve metrics into a clustered format (given clustered runs)."""
    pool = ThreadPoolExecutor(min(MAX_XP_LOADERS, len(nconfigs)))
    pget = functools.partial(get_all_logs, nconfigs, split_by_job_id=split_by_job_id, actors=actors)
    all_xp_logs = pool.map(pget, list(nconfigs.keys()))

    metrics = {}
    for experiment_id, xp_logs in all_xp_logs:
        if split_by_job_id:
            # Combine the job ID with `experiment_id` to obtain the final ID
            for job_id, logs in xp_logs.items():
                new_id = f"job_id={job_id}"
                if experiment_id == "ALL":
                    pass
                elif isinstance(experiment_id, str):
                    new_id = ",".join([experiment_id, new_id])
                else:
                    assert isinstance(experiment_id, tuple)
                    new_id = experiment_id + (new_id,)
                metrics[new_id] = logs
        else:
            assert len(xp_logs) == 1 and next(iter(xp_logs)) == -1
            metrics[experiment_id] = xp_logs[-1]            
        
    return metrics


def load_experiments(
    paths: typing.List[Path],
    config_filter: Optional[Callable] = None,
    xp_pivot: Optional[Callable] = None,
    run_pivot: Optional[Callable] = None,
    split_by_job_id: bool = False,
    actors="all",
) -> Metrics:
    """Retrieves and clusters experiment data.

    By default, it'll attempt to discover sweep parameters and cluster
    experiments around them.

    :param actors: One of "all", "train", "eval". If "train" then only metrics from training
        actors (whose data is used to update the model) are reported. If "eval" then only
        metrics from evaluation actors (whose data is not used for training) are reported.
    """
    configs = load_configs(paths, config_filter)
    if xp_pivot is None:
        sweep_args = find_args_diff(configs)
        print(f"Found the following sweep parameters: {sweep_args}.")

        def auto_pivot(config):
            return tuple(
                f"{SHORT_NAMES.get(k, k)}={v}" for k, v in sorted(config["flags"].items()) if k in sweep_args
            )

        xp_pivot = auto_pivot
    nc = cluster_runs(configs, xp_pivot, run_pivot)
    print_tree(nc)
    return load_metrics(nc, split_by_job_id=split_by_job_id, actors=actors)


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
    y_min=None,
    y_max=None,
):
    df = df[[x, y]].dropna()
    df[y] = df[y].rolling(ROLLING_WINDOW, min_periods=0).mean()
    grid = np.linspace(0, df[x].max(), subsample)
    yvalues = np.interp(grid, df[x].values, df[y].values)
    df = pd.DataFrame({x: grid, y: yvalues})
    p = hv.Curve(hv.Dataset(df, kdims=[x], vdims=[y]))
    p.opts(opts.Curve("Curve", color=color, ylim=(y_min, y_max)))
    return p


def plot_model(
    runs,
    x="step",
    y="mean_episode_return",
    model="model",
    color="#ff0000",
    **plot_run_args,
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
        p = plot_run(df, x, y, model, color, **plot_run_args)
        p.opts(opts.Curve(f"Curve", color=color, alpha=0.2))
        hmap[run] = p

    hmap = hv.HoloMap(hmap)
    p_runs = hmap.overlay().relabel("Runs")

    hmap_mean = hv.HoloMap(hmap)
    p_mean = hmap_mean.collapse(function=np.mean)
    p_mean = hv.Curve(p_mean).relabel(get_label(model))
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
    **plot_model_args,
):
    if model2color is None:
        colors = hv.Palette(palette).values
        colors = list(set(colors))
        model2color = {}
    hmap = {}
    for model, runs in models.items():
        if model not in model2color:
            model2color[model] = colors[(len(model2color) - 1) % len(colors)]
        color = model2color[model]
        hmap[model] = plot_model(runs, x, y, model, color, **plot_model_args)
    p = hv.HoloMap(hmap, kdims=get_kdims(next(iter(models))))
    p = p.overlay()
    return p


def plot(
    metrics,
    x="step",
    y="mean_episode_return",
    palette="Set1",
    subsample=1000,
    cols=3,
    y_min=None,
    y_max=None,
):
    hmap = {}
    model2color = {}
    colors = hv.Palette(palette).values
    colors = list(set(colors))
    for _facet, models in metrics.items():
        for model, _ in models.items():
            if model not in model2color:
                model2color[model] = colors[(len(model2color) - 1) % len(colors)]
    for facet, models in metrics.items():
        p = plot_facet(
            models, x, y, palette, model2color, subsample=subsample, y_min=y_min, y_max=y_max,
        )
        if facet not in hmap:
            hmap[facet] = {}
        hmap[facet] = p
    kdims = get_kdims(next(iter(metrics)))
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

# ## HiPlot

# +
def get_hiplot_data(data, all_y):
    """
    Obtain data for HiPlot plots.

    :param data: The raw experiment data, as obtained by `load_experiments()`.
    :param all_y: List of fields from the underlying dataframes, that are the main
        quantities of interest we want to analyze.

    :returns: A tuple `(global_avg_rank, per_experiment_data, all_y_keys)` where:
        - `global_avg_rank` is the HiPlot data one can use to visualize the average
          rank of a given run across all experiments (each experiment corresponds to
          a figure displayed by the `plot()` function). This makes it possible to
          identify what works best "on average".
        - `per_experiment_data` is the HiPlot data for each experiment. It is used to
          look more closely at what happens in a specific setting.
        - `all_y_keys` is the set of all HiPlot keys associated to the quantities of
          interest in `all_y`.
    """
    # Map experiment_id -> list of HiPlot dicts for each run in the experiment.
    # This HiPlot data is used to display per-experiment results.
    per_experiment_data = {}
    # Map run_id -> list of HiPlot dicts for each experiment the run appears in.
    # This HiPlot data is used to display global results (across all experiments) based
    # on the average rank of each run.
    run_to_hip = defaultdict(list)
    # All keys associated to fields found in `all_y` (which we want in all HiPlot plots).
    # This set is filled below as we add these keys.
    all_y_keys = set()

    # Gather HiPlot data for each split.
    for experiment_id, experiment_data in data.items():
        # The list containing the HiPlot data for each run in the experiment.
        hip_data = []
        # Map y -> list containing the corresponding value at the end of each run.
        all_y_val = defaultdict(list)

        # All options used to identify individual runs in this experiment.
        hip_args = list(set(option_val.split("=", 1)[0] for run_id in experiment_data for option_val in run_id))

        # List whose i-th element is a HiPlot dict for the i-th run in the experiment.
        # This HiPlot dict can also be found in `run_to_hip`, and will be used to plot
        # rank statistics across all experiments.
        run_index_to_hip = []

        for run_id, run_data in experiment_data.items():
            run_y = defaultdict(list)  # map y -> all values of y for this run (at end of training)
            for idx, (path, logs) in enumerate(run_data.items()):
                flags = logs["config"]["flags"]
                if idx == 0:
                    # This is the first job in the run: use it as reference.
                    # We store in `hip_data` the dict to be used in `per_experiment_data`.
                    hip_data.append({k: flags[LONG_NAMES.get(k, k)] for k in hip_args})
                    # And we make a copy to be used in `run_to_hip`.
                    h = copy.deepcopy(hip_data[-1])
                    # Ensure that for a given run_id, all HiPlot dictionaries share the same args.
                    for other_h in run_to_hip[run_id]:
                        assert all(other_h[k] == v for k, v in h.items()), (other_h, h)
                    run_to_hip[run_id].append(h)
                    run_index_to_hip.append(h)
                else:
                    # Make sure that other experiments in the same run are consistent in terms of args.
                    assert all(flags[LONG_NAMES.get(k, k)] == hip_data[-1][k] for k in hip_args)

                df = logs["df"][[x] + all_y].dropna()
                for y in all_y:
                    # Use a rolling window to extract a stable value for the quantity of interest.
                    df[y] = df[y].rolling(ROLLING_WINDOW, min_periods=0).mean()
                    run_y[y].append(df[y].iloc[-1])

            # The data we use for HiPlot is the mean across all jobs in the run.
            for y in all_y:
                y_mean = np.mean(run_y[y])
                y_key = SHORT_NAMES.get(y, y) + "_last"
                all_y_keys.add(y_key)
                hip_data[-1][y_key] = y_mean
                all_y_val[y].append(y_mean)

        # Compute the rank of each run within this experiment, for each quantity of interest y.
        for y, y_data in all_y_val.items():
            y_short = SHORT_NAMES.get(y, y)
            n = len(y_data)
            for rank, i in enumerate(np.argsort(y_data)):
                rank_norm = rank / (n - 1) if n > 1 else 0  # normalize rank between 0 and 1
                y_key = y_short + "_rank"
                all_y_keys.add(y_key)
                # Add the normalized rank to the HiPlot dictionary associated to the i-th run
                # in this experiment.
                run_index_to_hip[i][y_key] = rank_norm

        per_experiment_data[experiment_id] = hip_data

    # Compute the average rank across all experiments, for run IDs that appear in all experiments.
    for run_id, hip_dicts in run_to_hip.items():
        if len(hip_dicts) != len(data):
            # This run does not appear in all experiments: ignore it here.
            assert len(hip_dicts) < len(data)
            continue
        for y in all_y:
            y_key = SHORT_NAMES.get(y, y) + "_rank"
            # Compute average rank across all experiments.
            avg_rank = np.mean([h[y_key] for h in hip_dicts])
            # The first dict is taken as reference to hold the average rank. Others will not be used.
            hip_dicts[0][y_key] = avg_rank

    # List of the HiPlot data for each run containing their global average rank for each quantity
    # of interest. We keep only the first dict associated to each run since it is the one holding
    # this data.
    global_avg_rank = [hip_dicts[0] for hip_dicts in run_to_hip.values()]

    return global_avg_rank, per_experiment_data, all_y_keys


def display_hiplot(name, hiplot_data, max_cols_per_plot=0, mandatory_cols=None, sort_cols=False):
    """
    Utility function to display HiPlot data.

    The main job of this function is to create multiple plots when there are too
    many columns, so as to keep each plot readable.

    :param max_cols_per_plot: Maximum number of columns in each plot. Ignored if <= 0.
    :param mandatory_cols: List/set of columns that should be visible in each plot
        (when present in the data).
    :param sort_cols: Whether columns should be sorted (alphabetically).
    """
    if mandatory_cols is None:
        mandatory_cols = []

    all_data = list(hiplot_data)
    all_keys = list(set(k for hip_dict in all_data for k in hip_dict))
    if sort_cols:
        all_keys = sorted(all_keys)

    if max_cols_per_plot <= 0 or len(all_keys) <= max_cols_per_plot:
        # Easy case: we can fit everything in a single plot.
        print(f"*** {name} (1/1) ***")
        hiplot.Experiment.from_iterable(all_data).display()

    else:
        # Must split keys across multiple plots.
        must_have = [k for k in all_keys if k in mandatory_cols]
        assert max_cols_per_plot > len(must_have)  # since all keys in `y_keys` must be displayed
        other_keys = [k for k in all_keys if k not in must_have]
        n_other_keys_per_plot = max_cols_per_plot - len(must_have)
        n_plots = (len(other_keys) - 1) // n_other_keys_per_plot + 1  # total number of plots

        start = 0
        while start < len(other_keys):
            end = start + n_other_keys_per_plot
            plotted_keys = other_keys[start:end] + must_have
            to_display = [{k: h[k] for k in itertools.chain(plotted_keys, must_have)}
                          for h in all_data]
            plot_idx = start // n_other_keys_per_plot + 1
            print(f"*** {name} ({plot_idx} / {n_plots}) ***")
            hiplot.Experiment.from_iterable(to_display).display()
            start = end



# -
# ## Load data

# +
def get_paths(pattern):
    base_path = Path("/checkpoint/{}/mvfst-rl/multirun".format(getpass.getuser()))
    #base_path = Path("/checkpoint/{}/mvfst-rl/run".format(getpass.getuser()))
    if isinstance(pattern, str):
        pattern = [pattern]
    logdirs = [logdir for p in pattern for logdir in base_path.glob(p)]
    # This function identifies the folders of individual experiments. In multirun mode they
    # are just numbers, while in single run mode we can still identify them by the presence
    # of the "train" subfolder.
    is_individual_folder = lambda path: (path.is_dir() and path.name.isdigit()) or (path / "train").is_dir()
    all_paths = []
    for logdir in logdirs:
        if is_individual_folder(logdir):
            all_paths.append(logdir / "train")
        else:
            all_paths += [
                item / "train" for item in logdir.iterdir()
                if is_individual_folder(item)
            ]

    paths = []
    for path in all_paths:
        logf = path / "logs.tsv"
        if not logf.exists():
            print(f"logs.tsv not found, skipping {path}")
            continue
        if logf.stat().st_size == 0:
            print(f"Empty logs.tsv, skipping {path}")
            continue
        paths.append(path)
    return paths


pattern = "2020-12-08_05-29-41"  # cc_env_reward_log_ratio=false
pattern = ["2020-12-07_20-38-44", "2020-12-07_19-54-42"]
pattern = "2020-12-07_20-38-44"  # cc_env_reward_log_ratio=true | https://fb.quip.com/BolVAGQ6S8sN
pattern = "2020-12-11_16-31-35"  # varying a' / b'
pattern = "2020-12-12_17-01-02"  # env 0
pattern = "2020-12-28_07-00-40"  # 1+3
pattern = ["2020-12-28_06-56-55", "2020-12-28_07-00-40"]  # 1,3 & 1+3
pattern = "2020-12-21_06-07-08"  # reward normalization
pattern = "2020-12-29_05-26-54"  # varying gamma on envs 1 & 2
pattern = "2020-12-28_20-31-49"  # 1,2 hyper-parameter search
pattern = "2020-12-29_10-34-46"  # 0 + combination of other envs
pattern = "2020-12-29_10-49-19"  # 0 + combination of other envs (lr=1e-4)
pattern = "2020-12-30_05-19-03"  # hyper-param search #0
pattern = "2020-12-30_20-44-02"  # hyper-param search #1
pattern = ["2020-12-30_05-19-03", "2020-12-30_20-44-02"]  # hyper-param #0 & #1
pattern = ["2020-12-30_05-19-03", "2020-12-31_09-14-59"]  # hyper-param #0 & #2a
pattern = "2021-01-04_19-26-36"  # hyper-param search #2b (all jobs)
pattern = "2021-01-04_19-30-20"  # hyper-param search #2b (job 5 only)
pattern = ["2020-12-28_09-31-33", "2020-12-28_09-33-54"]  # 0,1,2,3,4,5 & 1+2+3+4+5
pattern = ["2020-12-30_05-19-03", "2020-12-30_20-44-02", "2020-12-31_09-14-59", "2021-01-04_19-26-36"]  # all hyper-params on 0,1,2,3,4,5
pattern = ["2020-12-31_09-14-59", "2021-01-04_19-26-36"]  # hyper-param 2a+b
pattern = ["2021-03-01_05-24-44", "2021-02-28_19-12-49"]  # individual models & multi-task models (common seeds: 6874,1234,2345,3456)
pattern = "2021-02-28_19-18-26"  # re-run best from 2021-01-04_19-26-36 with 1M/5M steps and combinations of job ID as input
pattern = "2021-02-28_19-12-49"  # re-run best from 2021-01-04_19-26-36 with 1M/5M steps
pattern = ["2021-03-01_16-40-01", "2021-02-28_19-12-49"]  # lr sweep for 5M steps (+ previous exp with 1M/5M steps and lr=5e-4)
pattern = "2021-03-02_05-11-55"  # checking if use_job_id_in_{actor,critic} helps
pattern = "2021-03-02_11-43-49"  # re-run of a=False,c=True on a seed that did not work
pattern = "2021-03-02_19-01-30"  # re-run of a=False,c=True but with lower learning rates to see if that helps
pattern = "2021-03-03_08-42-01"  # checking if different learning rates change stability for a=False,c=True
pattern = "2021-03-05_12-02-04"  # higher entropy cost
pattern = "2021-03-11_06-21-06"  # 5M steps with lower learning rate and higher entropy
pattern = ["2021-03-11_06-21-06", "2021-01-04_19-26-36/27"]  # 5M steps with lower lr & higher entropy + previous best 1M
pattern = "2021-03-03_14-27-22"  # checking if use_job_id_in_{actor,critic} helps (lower lr)
pattern = "2021-04-12_18-45-38"  # re-run for Arxiv paper
pattern = "2021-04-25_09-07-21"  # experimenting with entropy_cost at 1M steps
pattern = "2021-04-26_07-27-51"  # experimenting with entropy_cost at 5M steps for Arxiv paper's ablation result (use_lstm=False)
pattern = "2021-04-12_18-45-38"  # re-run for Arxiv paper
pattern = "2021-05-24_09-52-54"  # debugging: with/without normalization per job
pattern = ["2021-05-24_11-41-46", "2021-05-24_09-52-54"]  # debugging: with/without normalization per job
pattern = "2021-05-24_16-02-22"  # 10 seeds randomization
pattern = ["2021-05-24_16-02-22", "2021-05-24_19-18-14"]  # randomization vs baseline (providing job ID to critic)
pattern = ["2021-05-24_19-18-14", "2021-05-24_16-02-47"]  # baselines with/without normalization per job and job ID to critic
pattern = ["2021-05-24_16-02-22", "2021-05-24_16-02-47"]  # randomization vs baseline (*not* providing job ID to critic)
pattern = ["2021-04-25_11-31-53", "2021-04-26_07-27-51", "2021-04-26_08-10-52"]  # all entropy experiments
pattern = "2021-04-25_11-31-53"  # entropy experiments with lstm
pattern = "2021-05-25_15-09-20"  # baselines with higher entropy cost
pattern = "2021-05-26_13-22-45"  # check ability to generalize to very different scenario (low vs. high bandwidth)
pattern = "2021-05-24_07-09-41"  # debugging: normal run
pattern = "2021-05-24_07-22-51"  # debugging: with 10 eval actors
pattern = ["2021-05-24_07-09-41", "2021-05-24_07-22-51", "2021-05-24_11-41-46"]
pattern = "2021-05-24_19-18-14"  # baseline providing job ID to critic
pattern = ["2021-05-26_09-26-21", "2021-05-26_09-26-58"]  # generalization abilities
pattern = ["2021-05-24_16-02-47", "2021-05-24_16-02-22"]  # 10 seeds randomization vs baselines
pattern = ["2021-05-27_14-10-51", "2021-05-27_12-00-15"]  # varying the hidden size with randomization
pattern = ["2021-05-28_07-37-01"]  # varying the unroll length
pattern = ["2021-05-28_10-21-52"]  # varying gamma
pattern = "2021-05-28_12-29-13"  # varying learning_rate
pattern = "2021-05-28_14-36-35"  # varying alpha (RMSProp coefficient)
pattern = "2021-05-28_16-33-09"  # varying RMSProp momentum
pattern = "2021-05-28_18-33-41"  # varying history_size
pattern = "2021-05-29_06-11-27"  # varying reward_normalization_coeff
pattern = "2021-05-29_07-59-41"  # varying end_of_episode_bootstrap
pattern = "2021-06-02_19-37-54"  # narrower random distribution
pattern = ["2021-06-02_14-57-11", "2021-06-02_19-37-54"]
pattern = "2021-06-02_14-57-11"  # use_task_obs_in_critic=true,false
pattern = "2021-06-02_19-40-20"  # use_task_obs_in_critic=true,false over 5M steps
pattern = "2021-06-03_09-50-48"  # use reward or not in model
pattern = ["2021-06-03_13-44-24","2021-06-03_16-22-39"]  # varying entropy_cost
pattern = "2021-06-03_13-41-36"  # learning rate over 5M steps
pattern = "2021-06-03_19-55-20"  # shared_ator_critic (5M) -- bugged
pattern = "2021-06-04_05-17-03"  # shared_actor_critic (1M)
pattern = "2021-06-03_19-54-07"  # shared_actor_critic (1M) -- bugged
pattern = "2021-06-04_08-18-22"  # varying learning rate
pattern = "2021-06-04_05-18-20"  # shared_ator_critic (5M)

# Filter.
xp_filter = lambda c: c["flags"]["train_job_ids"] == [0] and c["flags"]["cc_env_reward_delay_log_offset"] == 1
xp_filter = lambda c: c["flags"]["learning_rate"] > 1e-7 and c["flags"]["seed"] in [1234, 4567, 6789]
xp_filter = lambda c: c["flags"]["train_job_ids"] in [[0,1,2,3,4,5]]
xp_filter = lambda c: c["flags"]["cc_env_reward_delay_factor"] == 1
xp_filter = lambda c: c["flags"]["cc_env_reward_delay_factor"] == 1 and len(c["flags"]["train_job_ids"]) == 1
xp_filter = lambda c: c["flags"]["train_job_ids"] in [[1],[2]]
xp_filter = lambda c: c["flags"]["train_job_ids"] == [1]
xp_filter = lambda c: c["flags"]["seed"] == 5937 or c["git"]["commit"] == "e681426054e133acbad1f6621da2f0bcff59c364"  # best h0 + h1
xp_filter = lambda c: c["flags"]["seed"] == 5937 or c["git"]["commit"] == "7399213029dd235049f122837dd86baa3dce4430"  # best h0 + h2
xp_filter = lambda c: (c["flags"]["seed"] == 5937 or c["git"]["commit"] == "7399213029dd235049f122837dd86baa3dce4430") and c["flags"]["hidden_size"] == 1024  # TMP
xp_filter = lambda c: c["flags"]["seed"] < 3000
xp_filter = lambda c: c["flags"]["seed"] in [6874,1234,2345,3456] and c["flags"]["total_steps"] == 1_000_000
xp_filter = lambda c: c["flags"]["total_steps"] == 5_000_000
xp_filter = lambda c: c["flags"]["learning_rate"] == 5e-4 and c["flags"]["seed"] in [33]
xp_filter = lambda c: c["flags"]["seed"] == 6874 or (c["flags"]["learning_rate"] == 1e-4 and not c["flags"]["use_job_id_in_critic"] and c["flags"]["entropy_cost"] == 0.005 and c["flags"]["seed"] == 4)
xp_filter = lambda c: c["flags"]["cc_env_history_size"] == 20
xp_filter = lambda c: f'{c["flags"]["entropy_cost"]:e}'.startswith("1") and c["flags"]["entropy_cost"] > 1e-5
xp_filter = lambda c: not c["flags"]["reward_normalization_stats_per_job"] or len(c["flags"]["train_job_ids"]) == 1
xp_filter = lambda c: c["flags"]["hidden_size"] in [128, 1024]
xp_filter = lambda c: c["flags"]["end_of_episode_bootstrap"] and c["flags"]["seed"] == 2
xp_filter = lambda c: c["flags"]["seed"] == 5
xp_filter = lambda c: c["flags"]["reward_normalization"]
xp_filter = lambda c: c["flags"]["entropy_cost"] < 1e-2
xp_filter = None



# Criterion to split plots.
# xp_pivot = lambda c: (
#     "rdelay=%s" % c["flags"]["cc_env_reward_delay_factor"],
#     "rlost=%s" % c["flags"]["cc_env_reward_packet_loss_factor"],
# )
# xp_pivot = lambda c: f"rdelay={c['flags']['cc_env_reward_delay_factor']}"
xp_pivot = lambda c: f'offsets={c["flags"]["cc_env_reward_delay_log_offset"]}, {c["flags"]["cc_env_reward_throughput_log_offset"]}'
xp_pivot = lambda c: f'seed={c["flags"]["seed"]},lr={c["flags"]["learning_rate"]:.0e}'
xp_pivot = lambda c: f'id={",".join(map(str, c["flags"]["train_job_ids"]))},s={c["flags"]["seed"]}'
xp_pivot = lambda c: f'job_ids={c["flags"]["train_job_ids"]}'
xp_pivot = lambda c: f'ids={c["flags"]["train_job_ids"]},b={c["flags"]["cc_env_reward_delay_factor"]}'
xp_pivot = lambda c: (f'ids={c["flags"]["train_job_ids"]},t={c["flags"]["cc_env_reward_throughput_log_offset"]},'
                      f'd={c["flags"]["cc_env_reward_delay_log_offset"]},eb={int(c["flags"]["end_of_episode_bootstrap"])}')
xp_pivot = lambda c: f'tlo={c["flags"]["cc_env_reward_throughput_log_offset"]}'
xp_pivot = None  # auto
xp_pivot = lambda c: (f'd={c["flags"]["cc_env_reward_delay_factor"]},p={c["flags"]["cc_env_reward_packet_loss_factor"]}')
xp_pivot = lambda c: f'seed={c["flags"]["seed"]}'
xp_pivot = lambda c: f'N={c["flags"]["total_steps"]}'
xp_pivot = lambda c: f'seed={c["flags"]["seed"]},N={c["flags"]["total_steps"]}'
xp_pivot = lambda c: f'a={c["flags"]["use_job_id_in_actor"]},c={c["flags"]["use_job_id_in_critic"]}'
xp_pivot = lambda c: f'e={c["flags"]["learning_rate"]},a={c["flags"]["use_job_id_in_actor"]},c={c["flags"]["use_job_id_in_critic"]}'
xp_pivot = lambda c: f'lstm={c["flags"]["use_lstm"]}'
xp_pivot = lambda c: f'task_obs={c["flags"]["use_task_obs_in_critic"]}'
xp_pivot = lambda c: "ALL"
xp_pivot = lambda c: f'lr={c["flags"]["learning_rate"]}'

# Criterion to split individual curves within each plot.
# run_pivot = lambda c: "actions=%s" % c["flags"]["num_actions"]
run_pivot = lambda c: f'j={c["flags"]["train_job_ids"]}'
run_pivot = lambda c: f'N={c["flags"]["total_steps"]}'
run_pivot = lambda c: f'lr={c["flags"]["learning_rate"]}'
run_pivot = lambda c: (f'id_in_actor={c["flags"]["use_job_id_in_actor"]}', f'id_in_critic={c["flags"]["use_job_id_in_critic"]}')
run_pivot = lambda c: f'entropy={c["flags"]["entropy_cost"]}'
run_pivot = lambda c: "ALL"
run_pivot = lambda c: f'lstm={c["flags"]["use_lstm"]}'
run_pivot = lambda c: (f'entropy_cost={c["flags"]["entropy_cost"]}',)
run_pivot = lambda c: f'h={c["flags"]["cc_env_history_size"]}'
run_pivot = {"ignore": ["jobs_random", "seed", "reward_normalization_stats_per_job", "train_job_ids", "use_job_id_in_critic"]}
run_pivot = lambda c: f'entropy={c["flags"]["entropy_cost"]:.5f}'
run_pivot = {"ignore": ["jobs_random", "seed", "num_actors_eval", "train_job_ids"]}
run_pivot = {"ignore": ["jobs_random", "seed", "train_job_ids"]}
run_pivot = {"ignore": ["seed", "observation_length"]}
run_pivot = None  # auto
run_pivot = {"ignore": ["train_jobs", "seed"]}
run_pivot = lambda c: f'J={c["flags"]["train_jobs"]["jobs"]["sampled_scenario"]["delay"]}'
run_pivot = {"ignore": ["seed", "use_reward"]}
run_pivot = {"ignore": "seed"}

paths = get_paths(pattern)
assert paths, f"no valid experiment data found for pattern '{pattern}'"
data = load_experiments(paths, xp_filter, xp_pivot, run_pivot, split_by_job_id=True, actors="eval")
# -

# ## Training curves

# +
# %%time
# %%opts Curve {+axiswise}
# %%opts NdOverlay [legend_position='bottom_right', show_legend=True]

y = "mean_avg_reward"
# y = "mean_episode_return"
# y = 'mean_episode_step'
# y = "mean_episode_return"
y = "mean_delay_mean"
y = "log_th/dl"
y = "mean_throughput_mean"
y = "mean_cwnd_mean"
y = "mean_avg_reward"
# y = 'total_loss'
# y = 'pg_loss'
# y = 'baseline_loss'
# y = 'entropy_loss'
# y = 'learner_queue_size'
plot(
    data,
    y=y,
    y_min=None,
    y_max=None,
    cols=4,
    palette="Colorblind",
    subsample=100,
)

# +
# %%opts Curve {+axiswise}
# %%opts NdOverlay [legend_position='bottom_right', show_legend=False]

# y = "mean_episode_return"
# y = 'mean_episode_step'
# y = "mean_episode_return"
y = "log_th/dl"
y = "mean_delay_mean"
y = "mean_cwnd_mean"
y = "mean_throughput_mean"
y = "mean_avg_reward"
# y = 'total_loss'
# y = 'pg_loss'
# y = 'baseline_loss'
# y = 'entropy_loss'
# y = 'learner_queue_size'
plot(
    data,
    y=y,
    y_min=None,
    y_max=None,
    cols=4,
    palette="Colorblind",
    subsample=100,
)

# +
# %%opts Curve {+axiswise}
# %%opts NdOverlay [legend_position='bottom_right', show_legend=False]

# y = "mean_episode_return"
# y = 'mean_episode_step'
# y = "mean_episode_return"
y = "mean_avg_reward"
y = "mean_delay_mean"
y = "mean_throughput_mean"
y = "mean_cwnd_mean"
y = "log_th/dl"
# y = 'total_loss'
# y = 'pg_loss'
# y = 'baseline_loss'
# y = 'entropy_loss'
# y = 'learner_queue_size'
plot(
    data,
    y=y,
    y_min=None,
    y_max=None,
    cols=4,
    palette="Colorblind",
    subsample=100,
)

# +
# %%opts Curve {+axiswise}
# %%opts NdOverlay [legend_position='bottom_right', show_legend=False]

# y = "mean_episode_return"
# y = 'mean_episode_step'
# y = "mean_episode_return"
y = "mean_avg_reward"
y = "mean_throughput_mean"
y = "mean_delay_mean"
y = "log_th/dl"
y = "mean_cwnd_mean"
# y = 'total_loss'
# y = 'pg_loss'
# y = 'baseline_loss'
# y = 'entropy_loss'
# y = 'learner_queue_size'
plot(
    data,
    y=y,
    y_min=None,
    y_max=None,
    cols=4,
    palette="Colorblind",
    subsample=100,
)

# +
# %%opts Curve {+axiswise}
# %%opts NdOverlay [legend_position='bottom_right', show_legend=False]

# y = "mean_episode_return"
# y = 'mean_episode_step'
# y = "mean_episode_return"
y = "mean_avg_reward"
y = "mean_cwnd_mean"
y = "mean_throughput_mean"
y = "log_th/dl"
y = "mean_delay_mean"
# y = 'total_loss'
# y = 'pg_loss'
# y = 'baseline_loss'
# y = 'entropy_loss'
# y = 'learner_queue_size'
plot(
    data,
    y=y,
    y_min=None,
    y_max=None,
    cols=4,
    palette="Colorblind",
    subsample=100,
)
# -

# ## HiPlot analysis

# +
max_cols_per_hiplot = 10
show_per_experiment = False
x = "step"
all_y = [
    #"mean_episode_return",
    #"mean_episode_step",
    #"mean_avg_reward",
    "mean_delay_mean",
    #"mean_cwnd_mean",
    "mean_throughput_mean",
    "log_th/dl",
]

global_avg_rank, per_experiment_data, all_y_keys = get_hiplot_data(data, all_y)

hip_display = functools.partial(display_hiplot, max_cols_per_plot=max_cols_per_hiplot, mandatory_cols=all_y_keys)

# Display global HiPlot combining all experiments to show average rank across experiments.
# This only makes sense if there are at least two experiments.
if len(data) >= 2:
    hip_display("GLOBAL AVG RANK ACROSS ALL EXPERIMENTS", global_avg_rank)

if show_per_experiment:
    # Display HiPlot for each experiment.
    for experiment_id, hip_data in per_experiment_data.items():
        hip_display(experiment_id, hip_data)
# -

