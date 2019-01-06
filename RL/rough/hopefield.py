import numpy as np
from RL.common.plot_renderer import PlotRenderer

plot_error = PlotRenderer(title="Deviation from total",
                          xlabel="iterations", ylabel="error", default_init=True)
plot_x = PlotRenderer(title="x", xlabel="index", default_init=True)

target_sum = 1
x = np.array([0.2, 0.3, 0.1, 0.15])
n_iterations = 100


def error(x, target_sum):
    return np.abs(np.sum(x) - target_sum)


def update_rule_hopefield(x, target_sum):
    for i in range(len(x)):
        x[i] = target_sum - (np.sum(x) - x[i])
    return x


def update_rule_custom(x, target_sum):
    all_hopefield_simultaneous = target_sum - np.sum(x) + x
    lr = 0.01
    x = x + lr * all_hopefield_simultaneous
    return x


def update_rule_slow_hopefield(x, target_sum):
    lr = 0.1
    for i in range(len(x)):
        x[i] = (1 - lr) * x[i] + lr * (target_sum - (np.sum(x) - x[i]))
    return x


def update_rule(x, target_sum):
    # return update_rule_hopefield(x, target_sum)
    # return update_rule_custom(x, target_sum)
    return update_rule_slow_hopefield(x, target_sum)


def update_plots(x, target_sum, plot_error: PlotRenderer, plot_x: PlotRenderer):
    plot_error.append_and_render([error(x, target_sum)])
    plot_x.update_and_render([[list(range(len(x))), x]])


def print_stats(x, target_sum):
    print("x: {x}\tError: {e}".format(x=x, e=error(x, target_sum)))


for iteration_id in range(n_iterations):
    x = update_rule(x, target_sum)
    print_stats(x, target_sum)
    update_plots(x, target_sum, plot_error, plot_x)

input("Press Enter to exit")
plot_x.close()
plot_error.close()
