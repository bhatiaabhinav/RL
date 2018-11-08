from matplotlib.figure import Figure, Axes  # noqa: F401
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from RL.common.utils import SimpleImageViewer


class PlotRenderer:
    def __init__(self, window_width=None, window_height=None, title=None, xlabel=None, ylabel=None, window_caption='Plot'):
        if title is not None:
            window_caption += ': {0}'.format(title)
        self.viewer = SimpleImageViewer(
            width=window_width, height=window_height, caption=window_caption)
        self.fig = Figure()
        self.axes = self.fig.gca()  # type: Axes
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)
        self.curves = None
        self.canvas = FigureCanvas(self.fig)

    def plot(self, *args, data=None, **kwargs):
        self.curves = self.axes.plot(*args, data=data, **kwargs)

    def update(self, list_data, autoscale=True):
        if not autoscale:
            self.axes.autoscale(enable=False)
        for curve, data in zip(self.curves, list_data):
            curve.set_data(data[0], data[1])
        if autoscale:
            self.axes.relim()
            self.axes.autoscale(enable=autoscale)

    def render(self):
        self.canvas.draw()
        width, height = self.fig.get_size_inches() * self.fig.get_dpi()
        image = np.fromstring(self.canvas.tostring_rgb(), dtype='uint8').reshape(
            int(height), int(width), 3)
        self.viewer.imshow(image)

    def update_and_render(self, list_data, autoscale=True):
        self.update(list_data, autoscale=autoscale)
        self.render()

    def close(self):
        self.viewer.close()
