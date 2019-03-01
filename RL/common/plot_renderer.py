import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Axes, Figure  # noqa: F401

from RL.common.utils import ImagePygletWingow


def moving_average(arr, n=30):
    if n is not None:
        if len(arr) == 0:
            return np.array([])
        cumsum = np.cumsum(np.insert(arr, 0, np.zeros(n)))
        if len(arr) < n:
            div = np.arange(1, len(arr) + 1)
        else:
            div = np.insert(n * np.ones(len(arr) - n + 1), 0, np.arange(1, n))
        return (cumsum[n:] - cumsum[:-n]) / div
    else:
        return arr


class PlotRenderer:
    def __init__(self, window_width=None, window_height=None, title=None, xlabel=None, ylabel=None, legend=[], window_caption='Plot', concat_title_with_caption=True, smoothing=None, style='seaborn', auto_save=False, save_path=None, default_init=False, auto_dispatch_on_render=True):
        if title is not None and concat_title_with_caption:
            window_caption += ': {0}'.format(title)
        # self.viewer = SimpleImageViewer(
        #     width=window_width, height=window_height, caption=window_caption)
        self.viewer = ImagePygletWingow(width=window_width, height=window_height, caption=window_caption, vsync=False)
        with plt.style.context(style):
            self.fig = Figure()
            self.axes = self.fig.gca()  # type: Axes
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.legend = legend
        self.axes.set_title(title)
        self.curves = None
        self.canvas = FigureCanvas(self.fig)
        self.data = []
        self.smoothing = smoothing
        self.auto_save = auto_save
        self.save_path = save_path
        self.auto_dispatch_on_render = auto_dispatch_on_render
        if default_init:
            self.plot([], [])

    def plot(self, *args, data=None, **kwargs):
        self.curves = self.axes.plot(*args, data=data, **kwargs)
        self.axes.legend(self.legend)
        for curve in self.curves:
            xs, ys = curve.get_data()
            self.data.append([list(xs), list(ys)])
        if self.auto_save:
            self.save()

    def update(self, list_data, autoscale=True):
        if not autoscale:
            self.axes.autoscale(enable=False)
        for curve, data, new_data in zip(self.curves, self.data, list_data):
            xs, ys = new_data
            data[0], data[1] = list(xs), list(ys)
            xs, ys = xs, moving_average(ys, self.smoothing)
            curve.set_data(xs, ys)
        if autoscale:
            self.axes.relim()
            self.axes.autoscale(enable=autoscale)
        if self.auto_save:
            self.save()

    def render(self):
        self.canvas.draw()
        width, height = self.fig.get_size_inches() * self.fig.get_dpi()
        image = np.fromstring(self.canvas.tostring_rgb(), dtype='uint8').reshape(
            int(height), int(width), 3)
        if self.auto_dispatch_on_render:
            self.viewer.imshow(image)
        else:
            self.viewer.set_image(image)

    def dispatch_events(self):
        self.viewer.dispatch_events()

    def update_and_render(self, list_data, autoscale=True):
        self.update(list_data, autoscale=autoscale)
        self.render()

    def append(self, list_y, starting_x=0, autoscale=True):
        for curve, data, y in zip(self.curves, self.data, list_y):
            xs, ys = data
            if len(xs) == 0:
                xs.append(starting_x)
            else:
                xs.append(xs[-1] + 1)
            ys.append(y)
            data[0], data[1] = xs, ys
            xs, ys = xs, moving_average(ys, self.smoothing)
            curve.set_data(xs, ys)
        if autoscale:
            self.axes.relim()
            self.axes.autoscale(enable=autoscale)
        if self.auto_save:
            self.save()

    def append_and_render(self, list_y, starting_x=0, autoscale=True):
        self.append(list_y, starting_x=starting_x, autoscale=autoscale)
        self.render()

    def clear(self):
        list_data = [[[], []] for curve in self.curves]
        self.update(list_data)

    def save(self, path=None):
        if path is None:
            path = self.save_path
        if path is None:
            raise FileNotFoundError('No save path given')
        dirname = os.path.dirname(path)
        if len(dirname) > 0 and not os.path.exists(dirname):
            os.makedirs(dirname)
        self.fig.savefig(path)

    def close(self):
        self.viewer.close()
