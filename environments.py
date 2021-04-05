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


tf.compat.v1.enable_v2_behavior()

class CardGameEnv(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            (len(config.COINS)+1,), np.float64, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(len(config.OBS_COLS),), dtype=np.float64, minimum=config.OBS_COLS_MIN,\
                maximum=config.OBS_COLS_MAX,\
                     name='observation')
        self.reset()
        self._episode_ended = False
        


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.memory_return = pd.DataFrame(columns=[t+"_close" for t in config.COINS])
        self._episode_ended = False
        self.index = 0
        self.time_delta = pd.Timedelta(5,unit='m')
        self.init_cash = 1000
        self.current_cash = self.init_cash
        self.current_value = self.init_cash
        self.previous_price = {}
        self.old_dict_coin_price_1 = {}
        self.old_dict_coin_price_2 = {}

        self.money_split_ratio = np.zeros((len(config.COINS)+1))
        self.money_split_ratio[0] = 1

        self.df = pd.read_csv(config.FILE)
        self.scaler = preprocessing.StandardScaler()
        
        self.df["date"] = self.df["date"].apply(lambda x: pd.Timestamp(x, unit='s', tz='US/Pacific'))
        self.df = self.df[self.df["coin"].isin(config.COINS)].sort_values("date")
        self.scaler.fit(self.df[config.SCOLS].values)
        self.df = self.df.reset_index(drop=True)

        self.max_index = self.df.shape[0]
        start_point = (np.random.choice(np.arange(3,self.max_index - config.EPISODE_LENGTH))//3) *3
        end_point = start_point + config.EPISODE_LENGTH//3 *3
        self.df = self.df.loc[start_point:end_point+2].reset_index(drop=True)
        
        
        self.df = self.df.reset_index(drop=True)


        self.init_time = self.df.loc[0,"date"]
        self.current_time = self.init_time
        self.dfslice = self.df[(self.df["coin"].isin(config.COINS))&(self.df["date"]>=self.current_time)&(self.df["date"]<self.current_time+pd.Timedelta(5,unit='m'))].copy().drop_duplicates("coin")

        self.current_stock_num_distribution = self.calculate_actual_shares_from_money_split()
        self.previous_value = self.current_value
        self.current_stock_money_distribution,self.current_value  = self.calculate_money_from_num_stocks()
        self.money_split_ratio = self.normalize_money_dist()
        
        self.step_reward = 0
        
        info_ =  {"state":"state",\
                "money_split":self.money_split_ratio,"share_num":self.current_stock_num_distribution,\
                "value":self.current_value,"time":self.current_time,\
                "reward":self.step_reward,\
                # "raw_output":self.get_observations_unscaled(),
                "scaled_output":self.get_observations()}
        self._state = info_["scaled_output"][config.OBS_COLS].values.flatten()
        reward = info_["reward"]
        self._episode_ended = True if self.index==config.EPISODE_LENGTH//3 else False
        

        return ts.restart(self._state)

    def _step(self, action):
 
        if self._episode_ended:
 
            return self.reset()
        if sum(action)<=1e-3:
            self.money_split_ratio = [1/len(action) for t in action]
        else:
            self.money_split_ratio = action/sum(action)

        self.current_stock_num_distribution = self.calculate_actual_shares_from_money_split()
        self.step_time()
        self.index +=1

        info_ =  {"state":"state",\
                    "money_split":self.money_split_ratio,"share_num":self.current_stock_num_distribution,\
                    "value":self.current_value,"time":self.current_time,\
                    "reward":self.step_reward,\
                    "scaled_output":self.get_observations()}
 
        self._state = info_["scaled_output"][config.OBS_COLS].values.flatten()
        reward = info_["reward"]
        self._episode_ended = True if self.index==config.EPISODE_LENGTH//3 else False
        if self._episode_ended:
            reward = 0
            return ts.termination(self._state , reward)
        else:
            try:
                return ts.transition(
                    self._state, reward=reward, discount=1)
            except Exception as e:
                print("ERRORRRRRR!!!!!!!!!!!!!!!!")
                print(self._state)
                print(reward)
                print(self.step_reward, self.current_value, self.previous_value)
                print(self.current_stock_money_distribution)
                print(self.current_stock_num_distribution)
                print(action)
                print(self.index)
                print(self.dfslice)
                print(self.current_time)
                print(self.money_split_ratio )
                print(e)
                self.df.to_csv(os.path.join(LOGDIR,"error_df.csv"))
                
                raise ValueError

    def step_time(self):
        self.current_time += self.time_delta
        self.dfslice = self.df[(self.df["coin"].isin(config.COINS))&(self.df["date"]>=self.current_time)&(self.df["date"]<self.current_time+pd.Timedelta(5,unit='m'))].copy().drop_duplicates("coin")
        self.previous_value = self.current_value
        self.current_stock_money_distribution,self.current_value  = self.calculate_money_from_num_stocks()
        self.money_split_ratio = self.normalize_money_dist()
        self.step_reward = self.current_value - self.previous_value
        # self.step_reward = np.min([self.step_reward,0.25])


    def get_observations(self):
        dfslice = self.dfslice
        dfs = pd.DataFrame()
        for i,grp in dfslice.groupby("coin"):
            tempdf = pd.DataFrame(self.scaler.transform(grp[config.SCOLS].values))
            tempdf.columns = [i+"_"+c for c in config.SCOLS]
            if dfs.empty:
                dfs = tempdf
            else:
                dfs = dfs.merge(tempdf,right_index=True,left_index=True,how='inner')

        return dfs
    def get_observations_unscaled(self):
        dfslice = self.dfslice
        dfs = pd.DataFrame()
        for i,grp in dfslice.groupby("coin"):
            tempdf = pd.DataFrame(grp[config.COLS].values)
            tempdf.columns = [i+"_"+c for c in config.COLS]
            if dfs.empty:
                dfs = tempdf
            else:
                dfs = dfs.merge(tempdf,right_index=True,left_index=True,how='inner')
        
        self.memory_return = pd.concat([self.memory_return,dfs[[t+"_close" for t in config.COINS]]],ignore_index=True)
        
        return dfs
    def calculate_actual_shares_from_money_split(self):
        dict_coin_price = self.dfslice[["coin","open"]]\
                        .set_index("coin").to_dict()["open"]
        
        num_shares = []
        for i,c in enumerate(config.COINS):
            if c in dict_coin_price:
                num_shares.append( self.money_split_ratio[i+1]*self.current_value//dict_coin_price[c] )
            else:
                num_shares.append( self.money_split_ratio[i+1]*self.current_value//self.old_dict_coin_price_1[c] )
            
        self.current_cash = self.money_split_ratio[0]*self.current_value
        for c in dict_coin_price:
            self.old_dict_coin_price_1[c] = dict_coin_price[c]
        
        return num_shares
    def calculate_money_from_num_stocks(self):
        money_dist = []
        money_dist.append(self.current_cash)
        dict_coin_price = self.dfslice[["coin","open"]]\
                        .set_index("coin").to_dict()["open"]
        for i,c in enumerate(config.COINS):
            if c in dict_coin_price:
                money_dist.append(self.current_stock_num_distribution[i]*dict_coin_price[c])
            else:
                money_dist.append(self.current_stock_num_distribution[i]*self.old_dict_coin_price_2[c])
        
        for c in dict_coin_price:
            self.old_dict_coin_price_2[c] = dict_coin_price[c]
        return money_dist,sum(money_dist)
    def normalize_money_dist(self):
        normal = []
        
        for i,c in enumerate(self.current_stock_money_distribution):
            normal.append(c/self.current_value)
        return normal