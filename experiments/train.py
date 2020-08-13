import sys

sys.path.append("..")

import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import os
import random

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from ibmac import IBMACAgentTrainer
from ibmac_inter import IBMACInterAgentTrainer


# from ..maddpg.trainer.atoc_comma import ATOC_COMA_AgentTrainer
import tensorflow.contrib.layers as layers

from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=100000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--trainer",
                        choices=['maddpg', 'ibmac', 'ibmac_inter'], default='ibmac',
                        help="name of the scenario script")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--beta", type=float, default=0.05, help="discount factor")
    parser.add_argument("--ibmac_com", action="store_true", default=False)
    parser.add_argument("--random-seed", type=int, default=42, help="random seed")
    parser.add_argument("--dim-message", type=int, default=4, help="dimension of messages")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./results/maddpg",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=25000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def critic_mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def before_com_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=tf.nn.relu)
        return out


def channel(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        z_mu = layers.fully_connected(out, num_outputs=num_units, activation_fn=None)
        z_log_sigma_sq = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        eps = tf.random_normal(
            shape=tf.shape(z_log_sigma_sq),
            mean=0, stddev=1, dtype=tf.float32)
        z = z_mu + tf.exp(0.5 * z_log_sigma_sq) * eps
        return z, z_mu, z_log_sigma_sq


def after_com_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def inter_step(input, num_outputs, scope, reuse=False, num_units=64, dim_message=4, rnn_cell=None):
    with tf.variable_scope(scope, reuse=reuse):
        obs, message = input
        h = layers.fully_connected(obs, num_outputs=num_units, activation_fn=tf.nn.relu)
        m = layers.fully_connected(message, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(h + m, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=None)
        action = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        z_mu = layers.fully_connected(out, num_outputs=dim_message, activation_fn=None)
        z_log_sigma_sq = layers.fully_connected(out, num_outputs=dim_message, activation_fn=None)
        eps = tf.random_normal(
            shape=tf.shape(z_log_sigma_sq),
            mean=0, stddev=1, dtype=tf.float32)
        z = z_mu + tf.exp(0.5 * z_log_sigma_sq) * eps
        return action, z, z_mu, z_log_sigma_sq


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    discrete_action = True  # if scenario_name == 'simple_spread' else False
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data,
                            discrete_action=discrete_action)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            discrete_action=discrete_action)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    if arglist.trainer == 'maddpg':
        trainer = MADDPGAgentTrainer
        for i in range(num_adversaries):
            trainers.append(trainer(
                "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
                local_q_func=(arglist.adv_policy == 'ddpg')))
        for i in range(num_adversaries, env.n):
            trainers.append(trainer(
                "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
                local_q_func=(arglist.good_policy == 'ddpg')))
    elif arglist.trainer == 'ibmac':
        trainer = IBMACAgentTrainer
        if num_adversaries != 0:
            trainers.append(trainer(
                "adversary_team", before_com_model, channel, after_com_model, critic_mlp_model,
                obs_shape_n[:num_adversaries], env.action_space[:num_adversaries], arglist,
                local_q_func=(arglist.adv_policy == 'ddpg')))
        trainers.append(trainer(
            "good_team", before_com_model, channel, after_com_model, critic_mlp_model, obs_shape_n[num_adversaries:],
            env.action_space[num_adversaries:], arglist,
            local_q_func=(arglist.good_policy == 'ddpg')))

    elif arglist.trainer == 'battle':

        trainer_1 = IBMACAgentTrainer
        trainer_2 = IBMACAgentTrainer
        if num_adversaries != 0:
            trainers.append(trainer_1(
                "adversary_team", before_com_model, channel, after_com_model, critic_mlp_model,
                obs_shape_n[:num_adversaries], env.action_space[:num_adversaries], arglist,
                local_q_func=(arglist.adv_policy == 'ddpg')))
            # trainers.append(trainer_2(
            #     "adversary_team", critic_mlp_model, obs_shape_n[:num_adversaries], env.action_space[:num_adversaries],
            #     arglist,
            #     local_q_func=(arglist.good_policy == 'ddpg')))
            # trainers.append(
            #     trainer_3("adversary_team", before_com_model, attention_channel, after_com_model, critic_mlp_model,
            #               obs_shape_n[:num_adversaries], env.action_space[:num_adversaries], arglist,
            #               local_q_func=(arglist.adv_policy == 'ddpg')))
        trainers.append(trainer_1(
            "good_team", before_com_model, channel, after_com_model, critic_mlp_model,
            obs_shape_n[num_adversaries:], env.action_space[num_adversaries:], arglist,
            local_q_func=(arglist.adv_policy == 'ddpg')))
        # trainers.append(trainer_2(
        #     "good_team", critic_mlp_model, obs_shape_n[num_adversaries:], env.action_space[num_adversaries:],
        #         arglist, local_q_func=(arglist.good_policy == 'ddpg')))
        # trainers.append(
        #     trainer_3("good_team", before_com_model, attention_channel, after_com_model, critic_mlp_model,
        #               obs_shape_n[num_adversaries:], env.action_space[num_adversaries:], arglist,
        #               local_q_func=(arglist.adv_policy == 'ddpg')))

    elif arglist.trainer == 'ibmac_inter':
        trainer = IBMACInterAgentTrainer
        if num_adversaries != 0:
            trainers.append(trainer(
                "adversary_team", inter_step, critic_mlp_model, obs_shape_n[:num_adversaries],
                env.action_space[:num_adversaries],
                arglist,
                local_q_func=(arglist.adv_policy == 'ddpg')))
        trainers.append(trainer(
            "good_team", inter_step, critic_mlp_model, obs_shape_n[num_adversaries:],
            env.action_space[num_adversaries:], arglist,
            local_q_func=(arglist.good_policy == 'ddpg')))


    else:
        raise NotImplementedError

    return trainers

def train(arglist):
    # random.seed(arglist.random_seed)
    # np.random.seed(arglist.random_seed)
    # tf.set_random_seed(arglist.random_seed)

    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        savers = [tf.train.Saver(U.scope_vars(trainer.name)) for trainer in trainers]

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            # U.load_state(arglist.load_dir)
            [U.load_state(os.path.join(arglist.load_dir, 'team_{}'.format(i)), saver=saver) for i, saver in
             enumerate(savers)]

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        if arglist.trainer == 'tarmac' or arglist.trainer == 'reuse_tarmac' or arglist.trainer == 'ibmac_inter':
            message_n = np.zeros([len(obs_n), 4])
        is_training = True

        t_start = time.time()

        writer = tf.summary.FileWriter("graph", U.get_session().graph)
        writer.close()

        writer = SummaryWriter(arglist.save_dir)

        print('Starting iterations...')
        while True:
            # get action
            if arglist.trainer == 'ibmac' or arglist.trainer == 'reuse_ibmac':
                is_inference = False
                if arglist.display or arglist.restore or arglist.benchmark:
                    is_inference = False
                if len(trainers) == 2:
                    action_n1 = trainers[0].action(obs_n[:num_adversaries], is_inference=is_inference)
                    action_n2 = trainers[1].action(obs_n[num_adversaries:], is_inference=is_inference)
                    action_n = [action[0] for action in action_n1] + [action[0] for action in action_n2]
                else:
                    action_n = trainers[0].action(obs_n, is_inference=is_inference)
                    action_n = [action[0] for action in action_n]
            elif arglist.trainer == 'ibmac_inter':
                if len(trainers) == 2:
                    action_n1, message_action_n1 = trainers[0].action(obs_n[:num_adversaries],
                                                                      message_n[:num_adversaries])
                    action_n2, message_action_n2 = trainers[1].action(obs_n[num_adversaries:],
                                                                      message_n[num_adversaries:])
                    action_n = [action[0] for action in action_n1] + [action[0] for action in action_n2]
                else:
                    action_n, message_action_n = trainers[0].action(obs_n, message_n)
                    action_n = [action[0] for action in action_n]
                    message_n = [message_action[0] for message_action in message_action_n]
            else:
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            if arglist.trainer == 'ibmac':
                if len(trainers) == 2:
                    trainers[0].experience(obs_n[:num_adversaries], action_n[:num_adversaries], rew_n[:num_adversaries],
                                           new_obs_n[:num_adversaries], done_n[:num_adversaries], terminal)
                    trainers[1].experience(obs_n[num_adversaries:], action_n[num_adversaries:], rew_n[num_adversaries:],
                                           new_obs_n[num_adversaries:], done_n[num_adversaries:], terminal)
                else:
                    trainers[0].experience(obs_n, action_n, rew_n, new_obs_n, done_n, terminal)
            elif arglist.trainer == 'ibmac_inter':
                if len(trainers) == 2:
                    trainers[0].experience(obs_n[:num_adversaries], message_n[:num_adversaries],
                                           action_n[:num_adversaries], rew_n[:num_adversaries],
                                           new_obs_n[:num_adversaries], done_n[:num_adversaries], terminal)
                    trainers[1].experience(obs_n[num_adversaries:], message_n[:num_adversaries],
                                           action_n[num_adversaries:], rew_n[num_adversaries:],
                                           new_obs_n[num_adversaries:], done_n[num_adversaries:], terminal)
                else:
                    trainers[0].experience(obs_n, message_n, action_n, rew_n, new_obs_n, done_n, terminal)
            else:
                for i, agent in enumerate(trainers):
                    agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for i, agent in enumerate(trainers):
                loss = agent.update(trainers, train_step)
                if loss:
                    if isinstance(agent, IBMACAgentTrainer) or isinstance(agent, ReuseIBMACAgentTrainer) :
                        q_loss, p_loss, _, _, _, _, kl_loss = loss
                        writer.add_scalar('agent_{}/loss_kl'.format(i), kl_loss, train_step)
                    else:
                        q_loss, p_loss, _, _, _, _ = loss
                    writer.add_scalar('agent_{}/loss_policy'.format(i), p_loss, train_step)
                    writer.add_scalar('agent_{}/loss_critic'.format(i), q_loss, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                [U.save_state(os.path.join(arglist.save_dir, 'team_{}'.format(i)), saver=saver) for i, saver in
                 enumerate(savers)]
                # print statement depends on whether or not there are adversaries

                for i in range(len(agent_rewards)):
                    writer.add_scalar('agent_{}/mean_episode_reward'.format(i),
                                      np.mean(agent_rewards[i][-arglist.save_rate:]), len(episode_rewards))

                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        round(time.time() - t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time() - t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
