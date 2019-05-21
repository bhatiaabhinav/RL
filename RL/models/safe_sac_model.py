from .sac_model import SACModel
import tensorflow as tf


class SafeSACModel(SACModel):
    def tf_actor_loss(self, actor_loss_coeffs, actor_loss_alpha, actor_critics, actor_logpis, name):
        with tf.variable_scope(name):
            violation = actor_critics[-1] - self.context.safety_threshold
            log_feasibility = -tf.exp(violation / self.context.beta)
            objective = sum([actor_loss_coeffs[i] * actor_critics[i] for i in range(self.num_critics - 1)]) - actor_loss_alpha * actor_logpis + actor_loss_alpha * actor_loss_coeffs[-1] * log_feasibility
            loss = -objective
            loss = tf.reduce_mean(loss)
            return loss
