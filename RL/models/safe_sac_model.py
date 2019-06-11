from .sac_model import SACModel
import tensorflow as tf
import numpy as np


class SafeSACModel(SACModel):
    def tf_actor_loss(self, actor_loss_coeffs, actor_loss_alpha, actor_critics, actor_logpis, name):
        with tf.variable_scope(name):
            violation = actor_critics[-1] - self.context.safety_threshold
            x = violation
            c = self.context.safe_sac_penalty_max_grad  # max gradient
            b = self.context.beta
            exp_penalty = -tf.exp(-x / b)
            linear_penalty = c * x + b * c * (np.log(b * c) - 1)  # expression cx + const to make this function cont from where exp_penalty left off.
            switch_at = -b * np.log(b * c)  # at this point, grad of exp_penalty is c. and the overall penalty should switch to a linear_penalty
            penalty = (tf.maximum(0.0, x - switch_at) / (x - switch_at)) * exp_penalty + (tf.minimum(0.0, x - switch_at) / (x - switch_at)) * linear_penalty
            log_feasibility = penalty
            objective = sum([actor_loss_coeffs[i] * actor_critics[i] for i in range(self.num_critics - 1)]) - actor_loss_alpha * actor_logpis + actor_loss_alpha * actor_loss_coeffs[-1] * log_feasibility
            loss = -objective
            loss = tf.reduce_mean(loss)
            return loss
