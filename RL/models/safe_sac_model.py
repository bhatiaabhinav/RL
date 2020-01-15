from .sac_model import SACModel
import tensorflow as tf
from RL.common.utils import tf_inputs


class SafeSACModel(SACModel):
    # def tf_actor_loss(self, actor_loss_coeffs, actor_loss_alpha, actor_critics, actor_logpis, name):
    #     with tf.variable_scope(name):
    #         self._tau_placholder, tau = tf_inputs(None, tf.float32, 'tau')
    #         self._lambda_placeholder, _lambda = tf_inputs(None, tf.float32, 'lambda')
    #         self._cost_placeholder, cost = tf_inputs(None, tf.float32, 'cost')

    #         c = self.context.safe_sac_penalty_max_grad  # max gradient
    #         switch = np.exp(_lambda * (cost - tau))
    #         # f = 30
    #         x = violation
    #         x = tf.reduce_mean(x)
    #         if c and False:
    #             max_x = b * tf.log(b * c)  # at this point, grad of exp_penalty is c.
    #             _x = tf.minimum(x, max_x)  # the part of x for which exp penaly should apply
    #             # _x = tf.reduce_mean(_x)
    #             # exp_penalty = -tf.exp(-_x / b) * tf.cos(-f * _x)
    #             exp_penalty = tf.exp(_x / b)
    #             linear_penalty = c * x + b * c * (tf.log(b * c) - 1)  # expression cx + const to make this function cont from where exp_penalty left off.
    #             step_fn = (1 + tf.sign(x - max_x)) / 2
    #             penalty = step_fn * linear_penalty + (1 - step_fn) * exp_penalty
    #             # penalty = exp_penalty
    #         else:
    #             # penalty = -tf.exp(-x / b) * tf.cost(f * x)
    #             penalty = tf.exp(x / b)
    #         # log_feasibility = penalty
    #         primary_objective = sum([actor_loss_coeffs[i] * actor_critics[i] for i in range(self.num_critics - 1)]) - actor_loss_alpha * actor_logpis
    #         primary_objective = tf.reduce_mean(primary_objective)

    #         objective = primary_objective - actor_loss_alpha * actor_loss_coeffs[-1] * penalty
    #         loss = -objective
    #         return 0

    def tf_actor_grads(self, actor_loss_coeffs, alpha, actor_critics, actor_logpis, actor_trainable_vars, name):
        with tf.variable_scope(name):
            self._actor_loss = tf.constant(0)
            self._tau_placeholder, tau = tf_inputs(None, tf.float32, 'tau')
            self._lambda_placeholder, _lambda = tf_inputs(None, tf.float32, 'lambda')
            self._cost_placeholder, cost = tf_inputs(None, tf.float32, 'cost')

            J_off = tf.reduce_mean(actor_critics[-1])

            c = self.context.safe_sac_penalty_max_grad  # max gradient
            exp_arg = _lambda * (cost - tau)
            exp_arg = tf.minimum(tf.log(float(c)), exp_arg)
            switch = tf.exp(exp_arg)

            term2 = tf.gradients(-_lambda * switch * J_off, actor_trainable_vars)

            sac_obj = sum([actor_loss_coeffs[i] * actor_critics[i] for i in range(self.num_critics - 1)]) - alpha * actor_logpis
            sac_obj = tf.reduce_mean(sac_obj)
            term1 = tf.gradients(sac_obj, actor_trainable_vars)

            ans = []
            for t1, t2 in zip(term1, term2):
                ans.append(-t1 - t2)
            return ans

    def train_actor(self, states, noise, critic_ids, loss_coeffs, on_policy_cost, alpha, cost_scaling, tau):
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
            self._lambda_placeholder: cost_scaling,
            self._tau_placeholder: tau,
            self._cost_placeholder: on_policy_cost
        })
        return loss, actor_critics, logstds, logpis
