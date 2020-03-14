from .sac_model import SACModel
import tensorflow as tf
from RL.common.utils import tf_inputs, need_conv_net, conv_net, dense_net
import RL


class SafeSACModel(SACModel):
    # def __init__(self, context: RL.Context, name: str, num_actors=1, num_critics=1, num_valuefns=1, learn_cost_fn=False, reuse=tf.AUTO_REUSE):
    #     super().__init__(context, name, num_actors, num_critics, num_valuefns, reuse)
    #     self.learn_cost_fn = learn_cost_fn
    #     with tf.variable_scope(name=name, reuse=reuse):
    #         if self.learn_cost_fn:
    #             self._cost_fn = self.tf_reward_fn(self._states_input_normalized, self._actions_input_normalized, "cost_fn")

    def tf_reward_fn(self, states, actions, name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            if need_conv_net(self.state_space):
                states = conv_net(states, self.context.convs, self.context.activation_fn, 'conv', reuse=reuse)
            states_actions = tf.concat(values=[states, actions], axis=-1)
            return dense_net(states_actions, self.context.hidden_layers, self.context.activation_fn, 1, lambda x: x, "dense", layer_norm=self.context.layer_norm, output_kernel_initializer=self.context.output_kernel_initializer, reuse=reuse)[:, 0]

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
