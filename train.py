import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tqdm import tqdm
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import ddpg_agent

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step
from tf_agents.utils import common
import logging

import config
from  environments import CardGameEnv
from utils import *

tf.compat.v1.enable_v2_behavior()
os.makedirs(config.LOGDIR,exist_ok=True)
os.makedirs(config.MODEL_SAVE,exist_ok=True)
logging.basicConfig(filename=os.path.join(config.LOGDIR,'log.log'), 
level=logging.INFO, 
format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
if __name__=='__main__':
    train_py_env = CardGameEnv()
    eval_py_env = CardGameEnv()
    
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    actor_fc_layers = config.actor_fc_layers
    critic_obs_fc_layers = config.critic_obs_fc_layers
    critic_action_fc_layers = config.critic_action_fc_layers
    critic_joint_fc_layers = config.critic_joint_fc_layers
    ou_stddev = config.ou_stddev
    ou_damping = config.ou_damping
    target_update_tau = config.target_update_tau
    target_update_period = config.target_update_period
    dqda_clipping = config.dqda_clipping
    td_errors_loss_fn = config.td_errors_loss_fn
    gamma = config.gamma
    reward_scale_factor = config.reward_scale_factor
    gradient_clipping = config.gradient_clipping

    actor_learning_rate = config.actor_learning_rate
    critic_learning_rate = config.critic_learning_rate
    debug_summaries = config.debug_summaries
    summarize_grads_and_vars = config.summarize_grads_and_vars
    
    global_step = tf.compat.v1.train.get_or_create_global_step()

    actor_net = actor_network.ActorNetwork(
            train_env.time_step_spec().observation,
            train_env.action_spec(),
            fc_layer_params=actor_fc_layers,
        )

    critic_net_input_specs = (train_env.time_step_spec().observation,
                            train_env.action_spec())

    critic_net = critic_network.CriticNetwork(
        critic_net_input_specs,
        observation_fc_layer_params=critic_obs_fc_layers,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers,
    )

    tf_agent = ddpg_agent.DdpgAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        ou_stddev=ou_stddev,
        ou_damping=ou_damping,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        dqda_clipping=dqda_clipping,
        td_errors_loss_fn=td_errors_loss_fn,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)
    tf_agent.initialize()
    
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=config.REPLAY_BUFFER_MAX_LENGTH)

    collect_data(train_env, random_policy, replay_buffer, steps=100)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=config.BATCH_SIZE, 
        num_steps=2).prefetch(3)
    
    my_policy = tf_agent.collect_policy
    saver = PolicySaver(my_policy, batch_size=None)

    iterator = iter(dataset)
    tf_agent.train = common.function(tf_agent.train)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, tf_agent.policy, \
                                    config.NUM_EVAL_EPISODES)
    returns = [avg_return]
    iterations=[0]
    for _ in tqdm(range(config.NUM_ITERATIONS),total=config.NUM_ITERATIONS):
            # Collect a few steps using collect_policy and save to the replay buffer.
            for _ in range(config.COLLECT_STEPS_PER_ITERATION):
                collect_step(train_env, tf_agent.collect_policy, replay_buffer)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = tf_agent.train(experience).loss

            step = tf_agent.train_step_counter.numpy()

            if step % config.LOG_INTERVAL == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

            if step % config.EVAL_INTERVAL == 0:
                avg_return = compute_avg_return(eval_env, tf_agent.policy, \
                                                config.NUM_EVAL_EPISODES)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                logging.info('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)
                iterations.append(step)
            if step % config.MODEL_SAVE_FREQ == 0:
                saver.save(os.path.join(config.MODEL_SAVE,f'policy_step_{step}_gamma.mdl'))
                
        # except:
        #     print("error_skipping")

    # iterations = range(0, num_iteratioens + 1, eval_interval)
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.ylim(top=50)
    plt.show()
    plt.savefig("output_img_gamma.png")
    pd.DataFrame({"interations":iterations,"Return":returns}).to_csv(os.path.join(config.LOGDIR,"output_ar_gamma.csv"),index=None)