import argparse
import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from RL.common.args import str2bool

try:
    # Not necessary for monitor writing, but very useful for monitor loading
    import ujson as json
except ImportError:
    import json

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID',
                    default='BreakoutNoFrameskip-v4')
parser.add_argument(
    '--logdir', help='logs will be read from logdir/{env}/{run_no}/  . Defaults to os env variable OPENAI_LOGDIR', default=os.getenv('OPENAI_LOGDIR'))
parser.add_argument(
    '--run_ids', help='run ids to plot. seperated by comma. Dont specify to plot all', default=None)
parser.add_argument(
    '--metrics', help='seperated by comma', default='Reward')
parser.add_argument('--scales', help='seperated by comma per metric. each scale of form y1:y2 or auto', default=None)
parser.add_argument('--smoothing', type=int, default=250)
parser.add_argument('--live', type=str2bool, default=False)
parser.add_argument('--style', default='seaborn')
parser.add_argument('--title', default=None)
parser.add_argument('--update_interval', type=int, default=30)
args = parser.parse_args()

# refer:
# https://matplotlib.org/examples/color/named_colors.html
color_map = {
    'sddpg': 'gold',
    'sddpg_rmsac': 'orange',
    'sddpg_rmsac_wolpert': 'darkorange',
    'sddpg_logac': 'coral',
    'sddpg_logboth': 'indianred',
    'cddpg': 'lightgreen',
    'cddpg_wolpert': 'mediumseagreen',
    'cddpg_rmsac': 'turquoise',
    'cddpg_rmsac_wolpert': 'teal',
    'cddpg_logac': 'skyblue',
    'cddpg_logac_wolpert': 'steelblue',
    'cddpg_logboth': 'mediumpurple',
    'cddpg_logboth_wolpert': 'darkorchid',
    'static': 'silver',
    'uniform': 'silver',
    'no_repositioning': 'silver',
    'greedy': 'gray',
    'vehicle_repositioning': 'gray',
    'rtrailer_cap5': 'gray'
}

line_style_map = {
    'sddpg_rmsac_wolpert': '--',
    'cddpg_rmsac_wolpert': '-',
    'greedy': ':',
    'vehicle_repositioning': ':',
    'rtrailer_cap5': ':'
}

marker_map = {
    # 'sddpg_rmsac_wolpert': 'o',
    # 'cddpg_rmsac_wolpert': '+'
}

legend_sort_order = {
    'sddpg': 20,
    'sddpg_rmsac': 30,
    'sddpg_rmsac_wolpert': 35,
    'sddpg_logac': 40,
    'sddpg_logboth': 50,
    'cddpg': 60,
    'cddpg_wolpert': 70,
    'cddpg_rmsac': 80,
    'cddpg_rmsac_wolpert': 90,
    'cddpg_logac': 100,
    'cddpg_logac_wolpert': 110,
    'cddpg_logboth': 120,
    'cddpg_logboth_wolpert': 130,
    'static': 0,
    'uniform': 0,
    'no_repositioning': 0,
    'greedy': 10,
    'rtrailer_cap5': 15,
    'vehicle_repositioning': 10
}


def enter_figure(event):
    event.canvas.figure.autoscale = False
    # print("autoscale false")


def leave_figure(event):
    # global autoscale, time_mouse_leave
    event.canvas.figure.autoscale = True
    # time_mouse_leave = time.time()
    event.canvas.figure.time_mouse_leave = time.time()
    # print('leave_figure', event.canvas.figure)
    # event.canvas.figure.patch.set_facecolor('grey')
    # event.canvas.draw()
    # print("autoscale true")


def load_progress(fname, fields):
    progress_values = {}
    episodes = -1
    with open(fname, 'rt') as f:
        lines = f.readlines()
        for line in lines:
            log = json.loads(line)
            if log.get('Episode', 0) <= episodes:
                # break when episodes reset to zero. most probably due to end of training and start of testing
                break
            if log.get('Exploited', False):
                episodes = log.get('Episode', 0)
                for f in fields:
                    if f in progress_values.keys():
                        progress_values[f].append(log.get(f, 0))
                    else:
                        progress_values[f] = []
    return progress_values, episodes


def load_baseline(fname, fields):
    data = {}
    with open(fname, 'rt') as f:
        line = f.readline()
        log = json.loads(line)
        for f in fields:
            data[f] = log.get(f, 0)
    return data


def moving_average(arr, n=30):
    if len(arr) == 0:
        return np.array([])
    cumsum = np.cumsum(np.insert(arr, 0, np.zeros(n)))
    if len(arr) < n:
        div = np.arange(1, len(arr) + 1)
    else:
        div = np.insert(n * np.ones(len(arr) - n + 1), 0, np.arange(1, n))
    return (cumsum[n:] - cumsum[:-n]) / div


def read_all_data(dirs, metrics):
    episodes = -1
    data = {}
    for dir_name in dirs:
        plot_name = dir_name
        if ':' in dir_name:
            plot_name = dir_name.split(':')[0]
            dir_name = dir_name.split(':')[1]
        logs_dir = os.path.join(args.logdir, args.env, dir_name)
        if os.path.isdir(logs_dir):
            baseline_fname = os.path.join(logs_dir, 'baseline.json')
            progress_fname = os.path.join(logs_dir, 'progress.json')
            if os.path.exists(baseline_fname):
                try:
                    baseline_data = load_baseline(baseline_fname, metrics)
                    data[dir_name] = {
                        "dir_name": dir_name,
                        "plot_name": plot_name,
                        "baseline_data": baseline_data,
                    }
                except Exception as e:
                    print(
                        "Could not read baseline.json for {0}".format(dir_name))
                    print(type(e), e)
            elif os.path.exists(progress_fname):
                try:
                    plot_data, plot_episodes = load_progress(
                        progress_fname, metrics)
                    data[dir_name] = {
                        "dir_name": dir_name,
                        "plot_name": plot_name,
                        "plot_data": plot_data
                    }
                    episodes = max(episodes, plot_episodes)
                except Exception as e:
                    print(
                        "Could not read progress.json for {0}".format(dir_name))
                    print(type(e), e)
    return data, episodes


def get_x_y(dir_data, metric, episodes):
    if 'baseline_data' in dir_data:
        episodes = max(episodes, 10)
        x = np.array(list(range(0, episodes + 1)))
        y = np.array([dir_data['baseline_data'][metric]] * (episodes + 1))
    else:
        x = np.array(dir_data['plot_data']['Episode'])
        y = np.array(dir_data['plot_data'][metric])
        y = moving_average(y, args.smoothing)
    if len(x) > 1000:
        # keep no more than 1000 points, so subsample:
        k = int(len(x) / 1000)
        x = x[::k].copy()
        y = y[::k].copy()
    return x, y


def plot_figure(data, metric, metric_label, scale, episodes):
    fig = plt.figure(num=metric)  # type: plt.Figure
    curves = []
    label_to_dir_name_map = {}
    for dir_name, dir_data in data.items():
        try:
            x, y = get_x_y(dir_data, metric, episodes)
            curve, = plt.plot(x, y, label=dir_data['plot_name'], color=color_map.get(
                dir_name), linestyle=line_style_map.get(dir_name, '-'), linewidth=2, marker=marker_map.get(dir_name))
            curves.append(curve)
            label_to_dir_name_map[dir_data['plot_name']] = dir_data['dir_name']
        except Exception as e:
            print("Could not plot {metric} for {dir_name}".format(
                metric=metric, dir_name=dir_name))
            print(type(e), e)
    plt.xlabel('Episode no')
    plt.ylabel(metric_label + ' (Smoothed)')
    if scale == 'auto':
        plt.gca().set_ylim(auto=True)
    else:
        plt.gca().set_ylim(scale)
    plt.gca().tick_params(labelsize='large')
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: -
                                  legend_sort_order.get(label_to_dir_name_map[t[0]], 1000)))
    plt.legend(handles, labels)
    plt.title(args.title)
    if args.live:
        fig.canvas.mpl_connect('axes_enter_event', enter_figure)
        fig.canvas.mpl_connect('axes_leave_event', leave_figure)
        fig.autoscale = True
        fig.time_mouse_leave = time.time() - args.update_interval
    return fig, curves


def update_figure(fig: plt.Figure, curves, data, metric, episodes):
    plt.figure(num=fig.number)
    for dir_data, curve in zip(data.values(), curves):
        try:
            x, y = get_x_y(dir_data, metric, episodes)
            curve.set_xdata(x)
            curve.set_ydata(y)
        except Exception as e:
            print("Could not plot {metric} for {dir_name}".format(
                metric=metric, dir_name=dir_data['dir_name']))
            print(type(e), e)
    plt.gca().relim()
    plt.gca().autoscale(enable=plt.gcf().autoscale and time.time() -
                        plt.gcf().time_mouse_leave > args.update_interval)


def save_figure(fig, name):
    save_dir = os.path.join(args.logdir, args.env, "plots")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "{0}.png".format(name))
    fig.savefig(save_path, dpi=192)


if args.run_ids is None:
    dirs = os.listdir(os.path.join(args.logdir, args.env))
else:
    dirs = args.run_ids.split(',')

if args.title is None:
    args.title = args.env

metrics = args.metrics.split(',')
metrics = [m.strip('\"') for m in metrics]
metric_labels = []
metric_ids = []
for m in metrics:
    m_label = m
    m_id = m
    if ':' in m:
        m_label = m.split(':')[0]
        m_id = m.split(':')[1]
    metric_labels.append(m_label)
    metric_ids.append(m_id)
metrics = metric_ids

if args.scales is None:
    scales = ['auto' for m in metrics]
else:
    scales_strings = args.scales.split(',')
    scales = []
    assert len(scales_strings) == len(metrics), "Please specify one scale for each metric"
    try:
        for scales_string in scales_strings:
            if scales_string == 'auto':
                scales.append(scales_string)
            else:
                bounds = scales_string.split(':')
                y1 = float(bounds[0])
                y2 = float(bounds[1])
                scales.append([y1, y2])
    except Exception as e:
        print("Could not parse scales")
        print(type(e), e)

data, episodes = read_all_data(dirs, ["Episode"] + metrics)
last_update_at = time.time()
plt.style.use(args.style)

mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.titlesize'] = 'x-large'
mpl.rcParams['axes.labelsize'] = 'x-large'
mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['ytick.labelsize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'x-large'
mpl.rcParams['figure.titlesize'] = 'x-large'

if args.live:
    plt.ion()

figs = []
curve_sets = []
for metric, metric_label, scale in zip(metrics, metric_labels, scales):
    fig, curves = plot_figure(data, metric, metric_label, scale, episodes)
    figs.append(fig)
    curve_sets.append(curves)
    save_figure(fig, metric)

while args.live:
    plt.pause(10)
    if time.time() - last_update_at >= args.update_interval:
        data, episodes = read_all_data(dirs, ["Episode"] + metrics)
        last_update_at = time.time()
        for fig, curves, metric in zip(figs, curve_sets, metrics):
            update_figure(fig, curves, data, metric, episodes)
            save_figure(fig, metric)
