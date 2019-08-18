from .sac_model import SACModel
import tensorflow as tf
import numpy as np


class SafeSACModel(SACModel):
    def tf_actor_loss(self, actor_loss_coeffs, actor_loss_alpha, actor_critics, actor_logpis, name):
        with tf.variable_scope(name):
            violation = actor_critics[-1] - self.context.safety_threshold
            c = self.context.safe_sac_penalty_max_grad  # max gradient
            b = self.context.beta
            x = violation
            if c:
                min_x = -b * np.log(b * c)  # at this point, grad of exp_penalty is c.
                exp_penalty = -tf.exp(-tf.maximum(x, min_x) / b)
                linear_penalty = c * x + b * c * (np.log(b * c) - 1)  # expression cx + const to make this function cont from where exp_penalty left off.
                step_fn = (1 + tf.sign(x - min_x)) / 2
                penalty = (1 - step_fn) * linear_penalty + step_fn * exp_penalty
            else:
                penalty = -tf.exp(-x / b)
            log_feasibility = penalty
            objective = sum([actor_loss_coeffs[i] * actor_critics[i] for i in range(self.num_critics - 1)]) - actor_loss_alpha * actor_logpis + actor_loss_alpha * actor_loss_coeffs[-1] * log_feasibility
            loss = -objective
            loss = tf.reduce_mean(loss)
            return loss
