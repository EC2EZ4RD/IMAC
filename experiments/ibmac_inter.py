import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer_with_messages import ReplayBuffer

import itertools


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r
        r = r * (1. - done)
        discounted.append(r)
    return discounted[::-1]


def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0 - polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


def p_train(make_obs_ph_n, make_meesages_ph_n, act_space_n, p_func, q_func, optimizer, grad_norm_clipping=None,
            local_q_func=False, num_units=64, scope="trainer", reuse=None, beta=0.01):
    with tf.variable_scope(scope, reuse=reuse):
        num_agents = len(make_obs_ph_n)

        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(num_agents)]

        messages_ph_n = make_meesages_ph_n

        # multi_head = pre_message(messages_ph_n)

        items = [p_func([obs_ph_n[i], tf.concat(messages_ph_n, 1)], int(act_pdtype_n[i].param_shape()[0]),
                        scope="p_func_{}".format(i), num_units=num_units) for i in range(num_agents)]
        p_n, message_n, mu_message_n, logvar_message_n = list(zip(*items))

        logvar_message_n = [tf.clip_by_value(log, -10, 10) for log in
                            logvar_message_n]  # constrain kl_loss not to be too large

        p_func_vars = [U.scope_vars(U.absolute_scope_name("p_func_{}".format(i))) for i in range(num_agents)]

        # wrap parameters in distribution
        act_pd_n = [act_pdtype_n[i].pdfromflat(p_n[i]) for i in range(num_agents)]

        act_sample_n = [act_pd.sample() for act_pd in act_pd_n]
        p_reg_n = [tf.reduce_mean(tf.square(act_pd.flatparam())) for act_pd in act_pd_n]

        act_input_n_n = [act_ph_n + [] for _ in range(num_agents)]
        for i in range(num_agents):
            act_input_n_n[i][i] = act_pd_n[i].sample()
        q_input_n = [tf.concat(obs_ph_n + messages_ph_n + act_input_n, 1) for act_input_n in act_input_n_n]

        q_n = [q_func(q_input_n[i], 1, scope="q_func_{}".format(i), reuse=True, num_units=num_units)[:, 0] for i in
               range(num_agents)]
        pg_loss_n = [-tf.reduce_mean(q) for q in q_n]

        kl_loss_message_n = [0.5 * (tf.pow(mu, 2) + tf.pow(tf.exp(log), 2)) - log - 0.5 for mu, log in
                             zip(mu_message_n, logvar_message_n)]
        kl_loss_message = tf.reduce_mean(kl_loss_message_n)

        pg_loss = tf.reduce_sum(pg_loss_n)
        p_reg = tf.reduce_sum(p_reg_n)
        loss = pg_loss + p_reg * 1e-3 + beta * kl_loss_message

        var_list = []
        var_list.extend(p_func_vars)
        var_list = list(itertools.chain(*var_list))
        optimize_expr = U.minimize_and_clip(optimizer, loss, var_list, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + messages_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=obs_ph_n + messages_ph_n, outputs=[act_sample_n, message_n])
        p_values = U.function(inputs=obs_ph_n + messages_ph_n, outputs=p_n)

        # target network
        target_items = [p_func([obs_ph_n[i], tf.concat(messages_ph_n, 1)], int(act_pdtype_n[i].param_shape()[0]),
                               scope="target_p_func_{}".format(i), num_units=num_units) for i in range(num_agents)]

        target_p_n, target_message_n, target_mu_message_n, target_logvar_message_n = list(zip(*target_items))
        target_logvar_message_n = [tf.clip_by_value(log, -10, 10) for log in
                                   target_logvar_message_n]  # constrain kl_loss not to be too large

        target_p_func_vars = [U.scope_vars(U.absolute_scope_name("target_p_func_{}".format(i))) for i in
                              range(num_agents)]

        target_var_list = []
        target_var_list.extend(target_p_func_vars)
        target_var_list = list(itertools.chain(*target_var_list))
        update_target_p = make_update_exp(var_list, target_var_list)

        target_act_sample_n = [act_pdtype_n[i].pdfromflat(target_p_n[i]).sample() for i in range(num_agents)]
        target_act = U.function(inputs=obs_ph_n + messages_ph_n, outputs=[target_act_sample_n, target_message_n])

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}


def q_train(make_obs_ph_n, make_meesages_ph_n, act_space_n, q_func, optimizer, grad_norm_clipping=None,
            local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        num_agents = len(make_obs_ph_n)

        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        messages_ph_n = make_meesages_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action_{}".format(i)) for i in
                    range(len(act_space_n))]
        target_ph_n = [tf.placeholder(tf.float32, [None], name="target_{}".format(i)) for i in range(num_agents)]

        q_input = tf.concat(obs_ph_n + messages_ph_n + act_ph_n, 1)
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
        train = U.function(inputs=obs_ph_n + messages_ph_n + act_ph_n + target_ph_n, outputs=loss,
                           updates=[optimize_expr])
        q_values = U.function(obs_ph_n + messages_ph_n + act_ph_n, q_n)

        # target network
        target_q_n = [q_func(q_input, 1, scope="target_q_func_{}".format(i), num_units=num_units)[:, 0] for i in
                      range(num_agents)]
        target_q_func_vars = [U.scope_vars(U.absolute_scope_name("target_q_func_{}".format(i))) for i in
                              range(num_agents)]

        traget_var_list = list(itertools.chain(*target_q_func_vars))
        update_target_q = make_update_exp(var_list, traget_var_list)

        target_q_values = U.function(obs_ph_n + messages_ph_n + act_ph_n, target_q_n)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


class IBMACInterAgentTrainer(AgentTrainer):
    def __init__(self, name, actor_model, critic_mlp_model, obs_shape_n, act_space_n, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.args = args
        obs_ph_n = []
        messages_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation_" + str(i)).get())
            messages_ph_n.append(U.BatchInput((args.dim_message,), name="message_" + str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            make_meesages_ph_n=messages_ph_n,
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
            make_meesages_ph_n=messages_ph_n,
            act_space_n=act_space_n,
            p_func=actor_model,
            q_func=critic_mlp_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
            beta=args.beta,
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        # self.max_replay_buffer_len = 50 * args.max_episode_len
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs_n, message_n):
        obs = [obs[None] for obs in obs_n]
        message = [message[None] for message in message_n]
        return self.act(*(list(obs) + list(message)))

    def experience(self, obs, message, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, message, act, rew, new_obs, [float(d) for d in done])

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len:  # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        samples = self.replay_buffer.sample_index(index)
        obs_n, message_n, act_n, rew_n, obs_next_n, done_n = [np.swapaxes(item, 0, 1) for item in samples]
        # for i in range(self.n):
        #     obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
        #     obs_n.append(obs)
        #     obs_next_n.append(obs_next)
        #     act_n.append(act)
        # obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n, target_next_message_n = self.p_debug['target_act'](*(list(obs_next_n) + list(message_n)))
            target_q_next_n = self.q_debug['target_q_values'](
                *(list(obs_next_n) + list(target_next_message_n) + list(target_act_next_n)))
            target_q_n = [rew + self.args.gamma * (1.0 - done) * target_q_next for rew, done, target_q_next in
                          zip(rew_n, done_n, target_q_next_n)]
        target_q_n = [target_q / num_sample for target_q in target_q_n]
        q_loss = self.q_train(*(list(obs_n) + list(message_n) + list(act_n) + target_q_n))

        # train p network
        p_loss = self.p_train(*(list(obs_n) + list(message_n) + list(act_n)))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew_n), np.mean(target_q_next_n), np.std(target_q)]
