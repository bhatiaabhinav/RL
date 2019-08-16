import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Axes, Figure  # noqa: F401

import RL
from RL.common.plot_renderer import moving_average

from .base_pyglet_rendering_agent import BasePygletRenderingAgent


class MatplotlibPlotAgent(BasePygletRenderingAgent):
    def __init__(self, context: RL.Context, name: str, list_data: List[Tuple[List[float], List[float]]], line_fmts: List[str], episode_interval=1, auto_dispatch_on_render=None, title=None, xlabel=None, ylabel=None, legend=[], smoothing=None, style='seaborn', auto_save=False, save_path=None):
        super().__init__(context, name, episode_interval=episode_interval)
        with plt.style.context(style):
            self.fig = Figure(dpi=48)
            self.axes = self.fig.gca()  # type: Axes
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.legend = legend
        self.axes.set_title(title)
        self.canvas = FigureCanvas(self.fig)
        self.list_data = list_data
        self.line_fmts = line_fmts
        self.smoothing = smoothing
        self.save_path = save_path
        self.auto_save = auto_save
        self.redraw_needed = True
        self.auto_dispatch_on_render = auto_dispatch_on_render

    def start(self):
        self.lines = []
        for (xdata, ydata), fmt in zip(self.list_data, self.line_fmts):
            line, = self.axes.plot(xdata, ydata, fmt)
            self.lines.append(line)
        self.axes.legend(self.legend)
        self.redraw_needed = True
        if self.auto_save:
            self.save()

    def update(self):
        print('update called')
        for line, (xdata, ydata) in zip(self.lines, self.list_data):
            ydata_smooth = moving_average(ydata, self.smoothing)
            line.set_data(xdata, ydata_smooth)
        self.axes.relim()
        self.axes.autoscale()
        self.redraw_needed = True

    def post_episode(self, env_id_nos):
        super().post_episode(env_id_nos)
        self.update()
        if self.auto_save:
            self.save()

    def save(self):
        print('save called')
        path = self.save_path
        if path is None:
            path = os.path.join(self.context.logdir, self.name, 'plot.png')
        dirname = os.path.dirname(path)
        if len(dirname) > 0 and not os.path.exists(dirname):
            os.makedirs(dirname)
        self.fig.savefig(path)

    def render(self):
        if self.redraw_needed:
            self.canvas.draw()
            width, height = self.fig.get_size_inches() * self.fig.get_dpi()
            self.image = np.fromstring(self.canvas.tostring_rgb(), dtype='uint8').reshape(
                int(height), int(width), 3)
            self.redraw_needed = False
        self.window.set_image(self.image, self.auto_dispatch_on_render)
