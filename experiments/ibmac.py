import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer

import itertools


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r
        r = r * (1. - done)
        discounted.append(r)
    return discounted[::-1]

def clip_message(message, clip_threshold, is_norm_training, is_inference):

    gamma = tf.Variable(clip_threshold * tf.ones(message.shape[-1]), name='clip_gamma')
    beta = tf.Variable(tf.zeros(message.shape[-1]), name='clip_beta')

    pop_mean = tf.Variable(tf.zeros(message.shape[-1]), trainable=False, name='pop_mean')
    pop_variance = tf.Variable(tf.ones(message.shape[-1]), trainable=False, name='pop_variance')

    epsilon = 1e-8

    def batch_norm_training():
        batch_mean, batch_variance = tf.nn.moments(message, [0])

        decay = 0.999
        train_mean = tf.assign(pop_mean, pop_mean*decay + batch_mean*(1 - decay), name='train_mean')
        train_variance = tf.assign(pop_variance, pop_variance*decay + batch_variance*(1 - decay), name='train_variance')

        with tf.control_dependencies([train_mean, train_variance]):
            return tf.nn.batch_normalization(message, batch_mean, batch_variance, batch_mean, tf.math.sqrt(batch_variance), epsilon, name='train_clip_message')

    def batch_norm_inference():
        return tf.nn.batch_normalization(message, pop_mean, pop_variance, beta, gamma, epsilon, name='inference_clip_message')

    def batch_direct_act():
        return message

    batch_normalized_output = tf.case({is_norm_training: batch_norm_training, is_inference: batch_norm_inference},
                default=batch_direct_act, exclusive=True)

    return batch_normalized_output



def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0 - polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


def p_train(make_obs_ph_n, act_space_n, before_com_func, channel, after_com_func, q_func, optimizer,
            grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None, beta=0.01,
            ibmac_com=True):
    with tf.variable_scope(scope, reuse=reuse):
        clip_threshold = 1 # 1, 5, 10
        is_norm_training = tf.placeholder(tf.bool)
        is_inference = tf.placeholder(tf.bool)


        ibmac_nocom = not ibmac_com
        num_agents = len(make_obs_ph_n)

        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(num_agents)]

        hiddens_n = [before_com_func(obs_ph_n[i], num_units, scope="before_com_{}".format(i), num_units=num_units) for i
                     in range(num_agents)]
        before_com_vars_n = [U.scope_vars(U.absolute_scope_name("before_com_{}".format(i))) for i in range(num_agents)]

        hiddens_n_for_message = tf.concat(
            [before_com_func(obs_ph_n[i], num_units, scope="before_com_{}".format(i), reuse=True, num_units=num_units)
             for i in range(num_agents)], axis=1)
        hiddens_n_for_message = tf.stop_gradient(hiddens_n_for_message)
        channel_output = channel(hiddens_n_for_message, num_units * num_agents, scope="channel",
                                 num_units=num_units * num_agents)
        message_n, mu_message_n, logvar_message_n = [tf.split(item, num_or_size_splits=num_agents, axis=1) for item in
                                                     channel_output]
        logvar_message_n = [tf.clip_by_value(log, -10, 10) for log in logvar_message_n] # constrain kl_loss not to be too large


        message_n = [clip_message(message, clip_threshold, is_norm_training, is_inference) for message in message_n]

        channel_vars_n = [U.scope_vars(U.absolute_scope_name("channel"))]

        if ibmac_nocom:
            print('no_com')
            p_n = [after_com_func(hiddens_n[i], int(act_pdtype_n[i].param_shape()[0]), scope="p_func_{}".format(i),
                                  num_units=num_units) for i in range(num_agents)]
        else:
            check_n = [hiddens_n[i] + message_n[i] for i in range(num_agents)]
            p_n = [after_com_func(hiddens_n[i] + message_n[i], int(act_pdtype_n[i].param_shape()[0]),
                                  scope="p_func_{}".format(i), num_units=num_units) for i in range(num_agents)]
        p_func_vars = [U.scope_vars(U.absolute_scope_name("p_func_{}".format(i))) for i in range(num_agents)]

        # wrap parameters in distribution
        act_pd_n = [act_pdtype_n[i].pdfromflat(p_n[i]) for i in range(num_agents)]

        act_sample_n = [act_pd.sample() for act_pd in act_pd_n]
        p_reg_n = [tf.reduce_mean(tf.square(act_pd.flatparam())) for act_pd in act_pd_n]

        act_input_n_n = [act_ph_n + [] for _ in range(num_agents)]
        for i in range(num_agents):
            act_input_n_n[i][i] = act_pd_n[i].sample()
        q_input_n = [tf.concat(obs_ph_n + act_input_n, 1) for act_input_n in act_input_n_n]

        q_n = [q_func(q_input_n[i], 1, scope="q_func_{}".format(i), reuse=True, num_units=num_units)[:, 0] for i in
               range(num_agents)]
        pg_loss_n = [-tf.reduce_mean(q) for q in q_n]

        # # 0.25
        # kl_loss_message_n = [2 * (tf.pow(mu, 2) + tf.pow(tf.exp(log), 2)) - log + np.log(0.5) - 0.5 for mu, log in
        #                      zip(mu_message_n, logvar_message_n)]

        # #1
        # kl_loss_message_n = [0.5 * (tf.pow(mu, 2) + tf.pow(tf.exp(log), 2)) - log - 0.5 for mu, log in
        #                      zip(mu_message_n, logvar_message_n)]
        # #5
        # kl_loss_message_n = [1.0/50 * (tf.pow(mu, 2) + tf.pow(tf.exp(log), 2)) - log + np.log(5) - 0.5 for mu, log in
        #                      zip(mu_message_n, logvar_message_n)]
        #10
        kl_loss_message_n = [1.0/200 * (tf.pow(mu, 2) + tf.pow(tf.exp(log), 2)) - log + np.log(10) - 0.5 for mu, log in
                             zip(mu_message_n, logvar_message_n)]

        entropy = [tf.exp(log) + 1.4189 for log in logvar_message_n]

        pg_loss = tf.reduce_sum(pg_loss_n)
        p_reg = tf.reduce_sum(p_reg_n)
        kl_loss_message = tf.reduce_mean(kl_loss_message_n)

        if ibmac_nocom:
            loss = pg_loss + p_reg * 1e-3
        else:
            loss = pg_loss + p_reg * 1e-3 + beta * kl_loss_message

        kl_loss = U.function(inputs=obs_ph_n + act_ph_n+[is_norm_training, is_inference], outputs=kl_loss_message)

        var_list = []
        var_list.extend(before_com_vars_n)
        if not ibmac_nocom:
            var_list.extend(channel_vars_n)
        var_list.extend(p_func_vars)
        var_list = list(itertools.chain(*var_list))
        optimize_expr = U.minimize_and_clip(optimizer, loss, var_list, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n+[is_norm_training, is_inference], outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=obs_ph_n+[is_norm_training, is_inference], outputs=act_sample_n)
        p_values = U.function(inputs=obs_ph_n+[is_norm_training, is_inference], outputs=p_n)
        if not ibmac_nocom:
            check_values = U.function(inputs=obs_ph_n+[is_norm_training, is_inference], outputs=check_n)
            channel_com = U.function(inputs=obs_ph_n+[is_norm_training, is_inference], outputs=channel_output)
            check_mu = U.function(inputs=obs_ph_n+[is_norm_training, is_inference], outputs=mu_message_n)
            check_log = U.function(inputs=obs_ph_n+[is_norm_training, is_inference], outputs=logvar_message_n)
        else:
            check_values = lambda x: 0
            channel_com = lambda x: 0
            check_mu = lambda x: 0
            check_log = lambda x: 0

        # target network
        target_hiddens_n = [
            before_com_func(obs_ph_n[i], num_units, scope="target_before_com_{}".format(i), num_units=num_units) for i
            in range(num_agents)]
        target_before_com_vars = [U.scope_vars(U.absolute_scope_name("target_before_com_{}".format(i))) for i in
                                  range(num_agents)]

        target_hiddens_n_for_message = tf.concat([before_com_func(obs_ph_n[i], num_units,
                                                                  scope="target_before_com_{}".format(i), reuse=True,
                                                                  num_units=num_units) for i in range(num_agents)],
                                                 axis=1)
        target_hiddens_n_for_message = tf.stop_gradient(target_hiddens_n_for_message)
        target_channel_output = channel(target_hiddens_n_for_message, num_units * num_agents, scope="target_channel",
                                        num_units=num_units * num_agents)
        target_message_n, target_mu_message_n, target_logvar_message_n = [
            tf.split(item, num_or_size_splits=num_agents, axis=1) for item in target_channel_output]
        target_channel_vars = [U.scope_vars(U.absolute_scope_name("target_channel"))]
        if ibmac_nocom:
            target_p_n = [after_com_func(target_hiddens_n[i], int(act_pdtype_n[i].param_shape()[0]),
                                         scope="target_p_func_{}".format(i), num_units=num_units) for i in
                          range(num_agents)]
        else:
            target_p_n = [
                after_com_func(target_hiddens_n[i] + target_message_n[i], int(act_pdtype_n[i].param_shape()[0]),
                               scope="target_p_func_{}".format(i), num_units=num_units) for i in range(num_agents)]
            # target_p_n = [after_com_func(tf.concat([target_hiddens_n[i],target_message_n[i]], axis=1), int(act_pdtype_n[i].param_shape()[0]), scope="target_p_func_{}".format(i), num_units=num_units) for i in range(num_agents)]
        target_p_func_vars = [U.scope_vars(U.absolute_scope_name("target_p_func_{}".format(i))) for i in
                              range(num_agents)]

        target_var_list = []
        target_var_list.extend(target_before_com_vars)
        if not ibmac_nocom:
            target_var_list.extend(target_channel_vars)
        target_var_list.extend(target_p_func_vars)
        target_var_list = list(itertools.chain(*target_var_list))
        update_target_p = make_update_exp(var_list, target_var_list)

        target_act_sample_n = [act_pdtype_n[i].pdfromflat(target_p_n[i]).sample() for i in range(num_agents)]
        target_act = U.function(inputs=obs_ph_n+[is_norm_training, is_inference], outputs=target_act_sample_n)


        check_message_n = U.function(inputs=obs_ph_n+[is_norm_training, is_inference], outputs=message_n)
        check_hiddens_n = U.function(inputs=obs_ph_n+[is_norm_training, is_inference], outputs=hiddens_n)
        check_entropy = U.function(inputs=obs_ph_n+[is_norm_training, is_inference], outputs=entropy)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act, 'kl_loss': kl_loss,
                                             'check_values': check_values, 'channel_com': channel_com,
                                             'check_mu': check_mu, 'check_log': check_log,
                                             'check_message_n':check_message_n, 'check_hiddens_n': check_hiddens_n,
                                             'check_entropy': check_entropy}


def q_train(make_obs_ph_n, act_space_n, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer",
            reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        num_agents = len(make_obs_ph_n)

        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action_{}".format(i)) for i in
                    range(len(act_space_n))]
        target_ph_n = [tf.placeholder(tf.float32, [None], name="target_{}".format(i)) for i in range(num_agents)]
        is_norm_training = tf.placeholder(tf.bool)
        is_inference = tf.placeholder(tf.bool)

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        q_n = [q_func(q_input, 1, scope="q_func_{}".format(i), num_units=num_units)[:, 0] for i in range(num_agents)]
        q_func_vars = [U.scope_vars(U.absolute_scope_name("q_func_{}".format(i))) for i in range(num_agents)]

        q_loss_n = [tf.reduce_mean(tf.square(q - target_ph)) for q, target_ph in zip(q_n, target_ph_n)]

        # viscosity solution to Bellman differential equation in place of an initial condition
        # q_reg = tf.reduce_mean(tf.square(q))
        q_loss = tf.reduce_sum(q_loss_n)
        loss = q_loss  # + 1e-3 * q_reg

        var_list = list(itertools.chain(*q_func_vars))
        optimize_expr = U.minimize_and_clip(optimizer, loss, var_list, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + target_ph_n+[is_norm_training, is_inference], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n+[is_norm_training, is_inference], q_n)

        # target network
        target_q_n = [q_func(q_input, 1, scope="target_q_func_{}".format(i), num_units=num_units)[:, 0] for i in
                      range(num_agents)]
        target_q_func_vars = [U.scope_vars(U.absolute_scope_name("target_q_func_{}".format(i))) for i in
                              range(num_agents)]

        traget_var_list = list(itertools.chain(*target_q_func_vars))
        update_target_q = make_update_exp(var_list, traget_var_list)

        target_q_values = U.function(obs_ph_n + act_ph_n+[is_norm_training, is_inference], target_q_n)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


class IBMACAgentTrainer(AgentTrainer):
    def __init__(self, name, before_com_model, channel, after_com_model, critic_mlp_model, obs_shape_n, act_space_n,
                 args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation_" + str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_func=critic_mlp_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            before_com_func=before_com_model,
            channel=channel,
            after_com_func=after_com_model,
            q_func=critic_mlp_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
            beta=args.beta,
            ibmac_com=args.ibmac_com,
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        # self.max_replay_buffer_len = 50 * args.max_episode_len
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

        self.message_1_for_record = []

    def action(self, obs_n, is_norm_training=False, is_inference=False):
        obs = [obs[None] for obs in obs_n]
        message_n = self.p_debug['check_message_n'](*(list(obs)+[is_norm_training, is_inference]))
        self.message_1_for_record.append(message_n[0])
        if len(self.message_1_for_record)%2500 == 0:
            # print(np.var(self.message_1_for_record, axis=0))
            # print(0.5 * np.log(2 * np.pi * np.mean(np.var(self.message_1_for_record, axis=0))) + 0.5)
            self.message_1_for_record = []
        return self.act(*(list(obs)+[is_norm_training, is_inference]))

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, [float(d) for d in done])

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len:  # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return
        is_norm_training = True
        is_inference = False
        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        samples = self.replay_buffer.sample_index(index)
        obs_n, act_n, rew_n, obs_next_n, done_n = [np.swapaxes(item, 0, 1) for item in samples]
        # for i in range(self.n):
        #     obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
        #     obs_n.append(obs)
        #     obs_next_n.append(obs_next)
        #     act_n.append(act)
        # obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        # print(len(obs_next_n))
        for i in range(num_sample):
            target_act_next_n = self.p_debug['target_act'](*(list(obs_next_n)+[is_norm_training, is_inference]))
            target_q_next_n = self.q_debug['target_q_values'](*(list(obs_next_n) + list(target_act_next_n)+[is_norm_training, is_inference]))
            target_q_n = [rew + self.args.gamma * (1.0 - done) * target_q_next for rew, done, target_q_next in
                          zip(rew_n, done_n, target_q_next_n)]
        target_q_n = [target_q / num_sample for target_q in target_q_n]
        q_loss = self.q_train(*(list(obs_n) + list(act_n) + target_q_n + [is_norm_training, is_inference]))

        # train p network
        p_loss = self.p_train(*(list(obs_n) + list(act_n)+[is_norm_training, is_inference]))

        self.p_update()
        self.q_update()

        # p_values = self.p_debug['p_values'](*(list(obs_n)))
        kl_loss = self.p_debug['kl_loss'](*(list(obs_n) + list(act_n)+[is_norm_training, is_inference]))
        # print('kl_loss', self.p_debug['kl_loss'](*(list(obs_n) + list(act_n))))
        # if t % 5000 == 0:
            #     print('p_values', p_values[0][0])
            #     print('check_value', self.p_debug['p_values'](*(list(obs_n)))[0][0])
            #     print('check_mu', self.p_debug['check_mu'](*(list(obs_n)))[0][0])
            #     print('check_log', self.p_debug['check_log'](*(list(obs_n)))[0][0])

            # print('kl_loss', kl_loss)
            # message_n = self.p_debug['check_message_n'](*(list(obs_n)+[is_norm_training, is_inference]))
            # hiddens_n = self.p_debug['check_hiddens_n'](*list(obs_n))
            # print("message_n", message_n[0][0])
            # for message in message_n:
            #     print("mean, var", np.mean(message, axis=0), np.var(message,axis=0))
            # print("hiddens_n", hiddens_n[0][0])
            # entropy = self.p_debug['check_entropy'](*list(obs_n))
            # print("entropy",np.mean(entropy, (1,2)))

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew_n), np.mean(target_q_next_n), np.std(target_q), kl_loss]
