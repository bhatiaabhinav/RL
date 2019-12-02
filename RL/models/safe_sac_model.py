from .sac_model import SACModel
import tensorflow as tf
from RL.common.utils import tf_inputs


class SafeSACModel(SACModel):
    def tf_actor_loss(self, actor_loss_coeffs, actor_loss_alpha, actor_critics, actor_logpis, name):
        with tf.variable_scope(name):
            self.tau, tau = tf_inputs(None, tf.float32, 'tau')
            violation = actor_critics[-1] - tau * self.context.cost_scaling
            c = self.context.safe_sac_penalty_max_grad  # max gradient
            # b = self.context.beta
            self.b, b = tf_inputs(None, tf.float32, 'beta')
            # f = 30
            x = -violation
            if c:
                min_x = -b * tf.log(b * c)  # at this point, grad of exp_penalty is c.
                _x = tf.maximum(x, min_x)  # the part of x for which exp penaly should apply
                # exp_penalty = -tf.exp(-_x / b) * tf.cos(-f * _x)
                exp_penalty = -tf.exp(-_x / b)
                linear_penalty = c * x + b * c * (tf.log(b * c) - 1)  # expression cx + const to make this function cont from where exp_penalty left off.
                step_fn = (1 + tf.sign(x - min_x)) / 2
                penalty = (1 - step_fn) * linear_penalty + step_fn * exp_penalty
            else:
                # penalty = -tf.exp(-x / b) * tf.cost(f * x)
                penalty = -tf.exp(-x / b)
            log_feasibility = penalty
            objective = sum([actor_loss_coeffs[i] * actor_critics[i] for i in range(self.num_critics - 1)]) - actor_loss_alpha * actor_logpis + actor_loss_alpha * actor_loss_coeffs[-1] * log_feasibility
            loss = -objective
            loss = tf.reduce_mean(loss)
            return loss

    def train_actor(self, states, noise, critic_ids, loss_coeffs, alpha, beta, tau):
        '''train the actor to optimize the critics specified by critic_ids weighted by loss_coeffs and optimize entropy weighted by alpha'''
        loss_coeffs_all_ids = [0] * self.num_critics
        actor_critics = []
        for i, coeff in zip(critic_ids, loss_coeffs):
            loss_coeffs_all_ids[i] = coeff
            actor_critics.append(self._actor_critics[i])
        _, loss, actor_critics, logstds, logpis = self.context.session.run([self._actor_train_step, self._actor_loss, actor_critics, self._actor_logstds, self._actor_logpis], {
            self._states_placeholder: states,
            self._actions_noise_placholder: noise,
            self._actor_loss_coeffs_placeholder: loss_coeffs_all_ids,
            self._actor_loss_alpha_placholder: alpha,
            self.b: beta,
            self.tau: tau
        })
        return loss, actor_critics, logstds, logpis
