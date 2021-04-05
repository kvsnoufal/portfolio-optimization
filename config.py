import tensorflow as tf
FILE = "cleaned_preprocessed.csv"
COINS = ["DASH","LTC","STR"]
COLS = ['high', 'low', 'open', 'close', 'volume', 'quoteVolume','weightedAverage']
SCOLS = ['vh', 'vl', 'vc', 'open_s', 'volume_s', 'quoteVolume_s', 'weightedAverage_s', 'vh_roll_7', \
    'vh_roll_14', 'vh_roll_30', 'vl_roll_7', 'vl_roll_14', 'vl_roll_30', 'vc_roll_7', 'vc_roll_14', 'vc_roll_30', \
        'open_s_roll_7', 'open_s_roll_14', 'open_s_roll_30', 'volume_s_roll_7', 'volume_s_roll_14', 'volume_s_roll_30', \
            'quoteVolume_s_roll_7', 'quoteVolume_s_roll_14', 'quoteVolume_s_roll_30', 'weightedAverage_s_roll_7', \
                'weightedAverage_s_roll_14', 'weightedAverage_s_roll_30']
OBS_COLS = ['DASH_vh', 'LTC_vh', 'STR_vh', 'DASH_vl', 'LTC_vl', 'STR_vl', 'DASH_vc', 'LTC_vc', 'STR_vc', \
    'DASH_open_s', 'LTC_open_s', 'STR_open_s', 'DASH_volume_s', 'LTC_volume_s', 'STR_volume_s', 'DASH_quoteVolume_s', \
        'LTC_quoteVolume_s', 'STR_quoteVolume_s', 'DASH_weightedAverage_s', 'LTC_weightedAverage_s', 'STR_weightedAverage_s', \
            'DASH_vh_roll_7', 'LTC_vh_roll_7', 'STR_vh_roll_7', 'DASH_vh_roll_14', 'LTC_vh_roll_14', 'STR_vh_roll_14', \
                'DASH_vh_roll_30', 'LTC_vh_roll_30', 'STR_vh_roll_30', 'DASH_vl_roll_7', 'LTC_vl_roll_7', 'STR_vl_roll_7', \
                    'DASH_vl_roll_14', 'LTC_vl_roll_14', 'STR_vl_roll_14', 'DASH_vl_roll_30', 'LTC_vl_roll_30', 'STR_vl_roll_30', \
                        'DASH_vc_roll_7', 'LTC_vc_roll_7', 'STR_vc_roll_7', 'DASH_vc_roll_14', 'LTC_vc_roll_14', 'STR_vc_roll_14', \
                            'DASH_vc_roll_30', 'LTC_vc_roll_30', 'STR_vc_roll_30', 'DASH_open_s_roll_7', 'LTC_open_s_roll_7', \
                                'STR_open_s_roll_7', 'DASH_open_s_roll_14', 'LTC_open_s_roll_14', 'STR_open_s_roll_14', 'DASH_open_s_roll_30', \
                                    'LTC_open_s_roll_30', 'STR_open_s_roll_30', 'DASH_volume_s_roll_7', 'LTC_volume_s_roll_7', 'STR_volume_s_roll_7', \
                                        'DASH_volume_s_roll_14', 'LTC_volume_s_roll_14', 'STR_volume_s_roll_14', 'DASH_volume_s_roll_30',\
                                             'LTC_volume_s_roll_30', 'STR_volume_s_roll_30', 'DASH_quoteVolume_s_roll_7', 'LTC_quoteVolume_s_roll_7', \
                                                 'STR_quoteVolume_s_roll_7', 'DASH_quoteVolume_s_roll_14', 'LTC_quoteVolume_s_roll_14', \
                                                     'STR_quoteVolume_s_roll_14', 'DASH_quoteVolume_s_roll_30', 'LTC_quoteVolume_s_roll_30', \
                                                         'STR_quoteVolume_s_roll_30', 'DASH_weightedAverage_s_roll_7', 'LTC_weightedAverage_s_roll_7', \
                                                             'STR_weightedAverage_s_roll_7', 'DASH_weightedAverage_s_roll_14', 'LTC_weightedAverage_s_roll_14',\
                                                                  'STR_weightedAverage_s_roll_14', 'DASH_weightedAverage_s_roll_30', 'LTC_weightedAverage_s_roll_30', 'STR_weightedAverage_s_roll_30']

OBS_COLS = ['DASH_vh','LTC_vh','STR_vh','DASH_vl','LTC_vl','STR_vl','DASH_vc','LTC_vc','STR_vc','DASH_open_s','LTC_open_s','STR_open_s','DASH_volume_s','LTC_volume_s','STR_volume_s','DASH_quoteVolume_s','LTC_quoteVolume_s','STR_quoteVolume_s','DASH_weightedAverage_s','LTC_weightedAverage_s','STR_weightedAverage_s','DASH_vh_roll_30','LTC_vh_roll_30','STR_vh_roll_30','DASH_vl_roll_30','LTC_vl_roll_30','STR_vl_roll_30','DASH_vc_roll_30','LTC_vc_roll_30','STR_vc_roll_30','DASH_open_s_roll_30','LTC_open_s_roll_30','STR_open_s_roll_30','DASH_volume_s_roll_30','LTC_volume_s_roll_30','STR_volume_s_roll_30','DASH_quoteVolume_s_roll_30','LTC_quoteVolume_s_roll_30','STR_quoteVolume_s_roll_30','DASH_weightedAverage_s_roll_30','LTC_weightedAverage_s_roll_30','STR_weightedAverage_s_roll_30']                                                                  

OBS_COLS_MIN = [-0.39,-0.39,-0.39,-52.92,-144.06,-159.81,-31.78,-19.57,-61.76,-87.7,-65.71,-130.35,-51.36,-51.09,-77.06,-0.03,-0.16,-74.55,-73.04,-73.72,-145.94,-0.77,-0.77,-0.77,-12.17,-14.24,-25.47,-15.64,-12.37,-12.73,-44.75,-25.01,-32.13,-39.2,-83.44,-74.04,-0.03,-0.2,-75.79,-44,-25,-32.25]
OBS_COLS_MAX = [53.16,69.71,90.97,0.4,0.4,0.4,39.31,24.4,85.49,114.25,66.32,130.35,55.72,60.25,66.09,0.04,0.18,96.32,67.32,73.63,145.94,9.96,17.79,53.89,0.81,0.81,0.81,7.58,9.9,14.93,41.44,16.35,32.11,34.64,82.48,59.62,0.03,0.2,78.16,38.94,16.29,32.24]
EPISODE_LENGTH = 1500
LOGDIR="LOGDIR"
MODEL_SAVE = "model_save"


NUM_ITERATIONS = 1000
COLLECT_STEPS_PER_ITERATION = 100
LOG_INTERVAL = 10
EVAL_INTERVAL = 4
MODEL_SAVE_FREQ = 12


REPLAY_BUFFER_MAX_LENGTH = 10000 #100000
BATCH_SIZE = 100
NUM_EVAL_EPISODES = 4

actor_fc_layers=(400, 300)
critic_obs_fc_layers=(400,)
critic_action_fc_layers=None
critic_joint_fc_layers=(300,)
ou_stddev=0.2
ou_damping=0.15
target_update_tau=0.05
target_update_period=5
dqda_clipping=None
td_errors_loss_fn=tf.compat.v1.losses.huber_loss
gamma=0.05
reward_scale_factor=1.0
gradient_clipping=None

actor_learning_rate=1e-4
critic_learning_rate=1e-3
debug_summaries=False
summarize_grads_and_vars=False