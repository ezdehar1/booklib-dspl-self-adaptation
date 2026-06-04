import csv
import json
import GA2
import logging
import gym
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pfrl
from pfrl import agents, replay_buffers, explorers, action_value
import logging
import joblib
import ast  # for safely parsing the “list” strings
import sklearn
import time
import random
from xgboost import XGBRegressor
import os
from time import perf_counter

def set_global_seed(seed: int = 0):
    """Set all relevant random seeds for reproducibility."""
    # Python stdlib
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    #env.action_space.seed(SEED)
    # PyTorch (CPU)
    torch.manual_seed(seed)
    # If you ever use CUDA in the future
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # PFRL (uses numpy, random etc. internally)
    pfrl.utils.set_random_seed(seed)



print(sklearn.__version__)

OUT_DIR = "plots_Dync"
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    filename='logging1/logger.log',
    filemode='w',            # append to existing log
    level=logging.INFO,
    format='%(asctime)s %(message)s',
)
logger = logging.getLogger(__name__)

# ==============================
# Running statistics for normalization
# ==============================
class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.array(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean, self.var, self.count = new_mean, new_var, tot_count


##################################### Normalizer different ways
def min_max_scale(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin )



####################################################################################################################################

class CustomD4RLNormalizedEnv(gym.Env):
    metadata = {'render.modes': []}



    def __init__(self, configs,config_binary,startup_df=None, max_steps=1, gamma=0, eps=1e-8):
        super().__init__()
        self.stats = {
            'RT': {'min': float('inf'), 'min_cfg': None, 'max': float('-inf'), 'max_cfg': None},
            'norm_rt': {'min': float('inf'), 'min_cfg': None, 'max': float('-inf'), 'max_cfg': None},
            #Utility
            'Utility': {'min': float('inf'), 'min_cfg': None, 'max': float('-inf'), 'max_cfg': None},
            'CPU': {'min': float('inf'), 'min_cfg': None, 'max': float('-inf'), 'max_cfg': None},
            'Mem': {'min': float('inf'), 'min_cfg': None, 'max': float('-inf'), 'max_cfg': None},
            'Cost': {'min': float('inf'), 'min_cfg': None, 'max': float('-inf'), 'max_cfg': None},
            'F_reward': {'min': float('inf'), 'min_cfg': None, 'max': float('-inf'), 'max_cfg': None},
        }

        self.configs = configs
        self.df = self.configs
        self.All_config= config_binary

        # build a list of unique config IDs
        self.action_ids = sorted(self.configs['config'].unique().tolist())
        self.rpm_vals = np.sort(self.configs['actual_rpm'].unique())
        self.num_users_vals = np.sort(self.configs['num_users'].unique())

        # Load File conatining single Utiltiy value for each config

        with open('Data/utility_result_4.json', 'r') as f:
            # load JSON and convert keys to int for easy lookup
            raw = json.load(f)
            self.config_utils = {int(k): v['utility'] for k, v in raw.items()}
            self.config_utils_B= {int(k): v['Utility_Bus_norm'] for k, v in raw.items()}
            util_values = np.array(list(self.config_utils.values()))   # convert to array in order to take mean and std

        with open('Data/UX.json', 'r') as f:
            # load JSON and convert keys to int for easy lookup
            raw = json.load(f)
            self.config_utils = {int(k): v['UX'] for k, v in raw.items()}

            util_values = np.array(list(self.config_utils.values()))   # convert to array in order to take mean and std
            #print(util_values)


        ###### Collect stat to do normalization, we tried several waies
        # build min/max rpm lookup for each config
        #Way1 min_max
        _group = self.df.groupby("config")["actual_rpm"]
        self._min_rpm = _group.min().to_dict()
        self._max_rpm = _group.max().to_dict()

        #Way2  z_score  initilze stats for normalization
        self.rpm_mean, self.rpm_std = self.configs['actual_rpm'].mean(), self.configs['actual_rpm'].std()
        self.users_mean, self.users_std = self.configs['num_users'].mean(), self.configs['num_users'].std()
        self.rt_mean, self.rt_std = self.configs['avg_response_time_ms'].mean(), self.configs['avg_response_time_ms'].std()
                ####### Normalize Other Things for reward cal
        self.cpu_mean, self.cpu_std = self.configs["total_cpu"].mean(), self.configs["total_cpu"].std()
        self.mem_mean, self.mem_std = self.configs["total_mem"].mean(), self.configs["total_mem"].std()

        self.utility_mean, self.utility_std =  util_values.mean(),  util_values.std()

        #print( self.utility_mean,"  UU  ", self.utility_std)

        stats = {
            "rpm_mean": float(self.rpm_mean), "rpm_std": float(self.rpm_std),
            "rt_mean": float(self.rt_mean), "rt_std": float(self.rt_std),
            "cpu_mean": float(self.cpu_mean), "cpu_std": float(self.cpu_std),
            "mem_mean": float(self.mem_mean), "mem_std": float(self.mem_std),
            "utility_mean": float(self.utility_mean), "utility_std": float(self.utility_std),
        }

        dfStat = pd.DataFrame([stats])  # one-row table
        dfStat.to_csv("BookLibStat", index=False)




        # other way
        self.rpm_min, self.rpm_max = self.rpm_vals.min(), self.rpm_vals.max()
        self.nu_min, self.nu_max = self.num_users_vals.min(), self.num_users_vals.max()

        # Normalizers
        self.obs_rms = RunningMeanStd(shape=(1,))  # only track RT
        self.ret_rms = RunningMeanStd(shape=())  # normalizer for  return

        self.cpu_normalizer = RunningMeanStd(shape=())
        self.mem_normalizer = RunningMeanStd(shape=())
        self.util_normalizer = RunningMeanStd(shape=())
        self.ep=1e-4  # for normalizaion
        ####################################################################################################
        ################### DQN parametrs
        self.max_steps = max_steps
        self.gamma = gamma
        self.epsilon = eps
        self.action_space = gym.spaces.Discrete(len(self.action_ids))
        self.observation_space = gym.spaces.Box(
            low=np.array([self.rpm_min,  -np.inf], dtype=np.float32),
            high=np.array([self.rpm_max,  np.inf], dtype=np.float32),
            dtype=np.float32,
        )



        flag_cols = ['Agg', 'GB1', 'GB2', 'Inven1', 'Inven2', 'Rev1', 'Rev2', 'Recom1', 'Recom2', 'Adv']
        # In future refactor this code , it seems that you only need flh_cols, in the prediction inside the step mwhere
        # the slef.config_flags code is unnecessary
        self.config_flags = (
            self.All_config[['Config_ID'] + flag_cols]
            .drop_duplicates(subset='Config_ID')
            .set_index('Config_ID')[flag_cols]
            .to_dict(orient='index')
        )


        # To Do check PM (Response Time) monolotic !!!!!! or not

        #Load preditive model
        # self.preproc = joblib.load("Model_Mix/preproc.pkl")
        # self.RT_model = joblib.load("Model_Mix/RT_model_mix_36.pkl")
        # self.CPU_model= joblib.load("Model_Mix/CPU_model_mix_36.pkl")
        # self.Mem_model= joblib.load("Model_Mix/Mem_model_mix_36.pkl")

        # self.preproc = joblib.load("Model_Last/Models/BookLib_RT_preproc_binary.pkl")
        # self.RT_model = joblib.load("Model_Last/Models/BookLib_RT_XGB_binary.pkl")
        # self.CPU_model = joblib.load("Model_Last/Models/BookLib_CPU_XGB_binary.pkl")
        # self.Mem_model = joblib.load("Model_Last/Models/BookLib_MEM_XGB_binary.pkl")

        self.preproc = joblib.load("Model_Last/Models/BookLib_RT_preproc_binary_3wise24.pkl")
        self.RT_model = joblib.load("Model_Last/Models/BookLib_RT_XGB_binary_3wise24.pkl")
        self.CPU_model = joblib.load("Model_Last/Models/BookLib_CPU_XGB_binary_3wise24.pkl")
        self.Mem_model = joblib.load("Model_Last/Models/BookLib_MEM_XGB_binary_3wise24.pkl")

        # placeholders
        self.current_workload = None
        self.response_time = None
        self.cpu = None
        self.mem = None
        self.utility = None
        self.utility_B = None
        self.returns = None
        self.step_count = None
        self.w_rt = 0.4  # weight for response‐time penalty
        self. w_util = 0.3  # weight for QoE
        self. w_cost = 0.3  # weight for cost penalty
        self.tau_ms=None


        #### New code

        self.startup_df = startup_df
        if self.tau_ms is None:
            self.tau_ms = 4600#float(self.configs["avg_response_time_ms"].quantile(0.70))
        else:
            self.tau_ms = 4600#float(self.tau_ms)

        print( "Tau.... ",self.tau_ms)


        # or absolute path

        # keep only what you need (optional)
        self.startup_df = self.startup_df[["actual_rpm",  "config", "avg_response_time_ms"]].dropna()

        # if you want to remember τ for reporting/logs:
        #self.tau_ms = 6233.65  # global p95 from the summary file (optional)

        self.rpm_low_thr = int(round(self.df["actual_rpm"].quantile(0.10)))  # 191 instead of 200

        self.rpm_spike_thr = int(round(self.df["actual_rpm"].quantile(0.90)))  # 818

        print("rpm" ,     self.rpm_low_thr,   "   " , self.rpm_spike_thr)
        ############



    def reset(self):
        self.step_count = 0
        self.returns = 0.0
        self.utility = 0.0
        self.cost_norm = 0

        # --- NEW: realistic trigger-based initialization ---
        if hasattr(self, "startup_df") and len(self.startup_df) > 0:
            r = self.startup_df.sample(1).iloc[0]
            #print(r)

            self.current_rpm = int(r["actual_rpm"])
            #self.current_num_users = int(r["num_users"])
            self.response_time = float(r["avg_response_time_ms"])

            self.init_rt_ms = float(r["avg_response_time_ms"])  # store trigger RT for improvement-based reward
            # optional: store which "prev config" caused this violation (for logging)
            self.prev_config = int(r["config"])
        else:
            # fallback (should rarely happen)
            self.current_rpm = int(np.random.choice(self.rpm_vals))
            #self.current_num_users = int(np.random.choice(self.num_users_vals))
            self.response_time = 5000.0
            self.init_rt_ms = float(self.response_time)
            self.prev_config = -1

        # update running stats for RT normalization
        self.obs_rms.update(np.array([[self.response_time]]))

        # Way1 normalize rpm and num_users
        norm_rpm = min_max_scale(self.current_rpm, self.rpm_min, self.rpm_max)
        #norm_nu = min_max_scale(self.current_num_users, self.nu_min, self.nu_max)

        # RT normalized using running mean/std (your current way)
        #norm_rt = (self.response_time - self.obs_rms.mean[0]) / np.sqrt(self.obs_rms.var[0] + self.ep)
        norm_rt2 = (self.response_time - self.rt_mean) / (self.rt_std + self.ep)

        obs1 = np.array([norm_rpm,  norm_rt2], dtype=np.float32)
        return obs1

    def step(self, action):

        self.step_count += 1

        # map the integer action back to your real Config_ID
        config_id = self.action_ids[action]
        #print(config_id,"  ",action)fag
        # lookup in the dataframefagent
        self.df = self.configs

        flags = self.config_flags[config_id]
        # build the feature dict
        feat = {
            'config': config_id,


            'actual_rpm': self.current_rpm,
            **flags,  # merges in Agg, GB1, …, Adv
        }

        df_pred = pd.DataFrame([feat])

        # 1) transform exactly as during training
        X_t = self.preproc.transform(df_pred)

        # 2) predict RT ................
        log_rt = self.RT_model.predict(X_t)[0]
        predicted_rt = float(np.expm1(log_rt))
        #print("Predicted RT ",predicted_rt)
        self.response_time = predicted_rt

        # 3) predict CPU_Cost ................
        # self.cpu = self.CPU_model.predict(X_t)
        # self.cpu = float(np.expm1(self.cpu).item())
        # #print("self.cpu ", self.cpu)
        # self.cpu_normalizer.update(np.array([[self.cpu]]))
        self.cpu=0
        #norm_cpu = (self.cpu - self.cpu_normalizer.mean[0]) / np.sqrt(self.cpu_normalizer.var[0] + self.ep)
        #norm_cpu2 = (self.cpu - self.cpu_mean) / (self.cpu_std + self.ep)
        #print("cpu norm ", norm_cpu)

        # 4) predict Mem_Cost ................
        # self.mem = self.Mem_model.predict(X_t)
        # self.mem= float(np.expm1(self.mem).item())
        # #print("self.mem ", self.mem)
        self.mem=0
        #self.mem_normalizer.update(np.array([[self.mem]]))


        #norm_mem = (self.mem - self.mem_normalizer.mean[0]) / np.sqrt(self.mem_normalizer.var[0] + self.ep)
        #norm_mem2 = (self.mem - self.mem_mean) / (self.mem_std + self.ep)

        #print("mem norm ", norm_mem)
        # new cost penalty term (negative since higher cost is worse)
        #cost_penalty =  ((norm_cpu2 + norm_mem2)/2)
        #cost_penalty=float(cost_penalty .item())
        #print("cost ", cost_penalty )

        ############################################################


        # Get utility value for the chosen config

        self.utility = self.config_utils[config_id]  # I used the already normalized
        self.utility_B= self.config_utils_B[config_id]

        self.util_normalizer.update(np.array([[self.utility]]))
        norm_util = (self.utility - self.util_normalizer.mean[0]) / np.sqrt(self.util_normalizer.var[0] + self.ep)
        norm_util2= (self.utility - self.utility_mean) /  (self.utility_std + self.ep)
        norm_util3 = self.utility # the ready 0-1 in reslut3
        norm_utilb= self.utility_B
        # print(self.utility)
        #####################################################################
        # normalization of RT
        self.obs_rms.update(np.array([[self.response_time]]))
        #norm_rt = (self.response_time - self.obs_rms.mean[0]) / np.sqrt(self.obs_rms.var[0] + self.ep)

        #print("norm_rt ", norm_rt)
        norm_rpm = min_max_scale(self.current_rpm, self.rpm_min, self.rpm_max)
        #norm_nu = min_max_scale(self.current_num_users, self.nu_min, self.nu_max)
        #print(self.rpm_min," RPM  ", self.rpm_max)

        # Way2 use z_score for all ==> Not used

        norm_rt2=( self.response_time - self.rt_mean) / (self.rt_std + self.ep)
        norm_rt2 = (np.tanh(norm_rt2) + 1) / 2
        ########## End of way2



        obs = np.array([
            norm_rpm,
            #norm_nu,
            norm_rt2           #give true rt without negtive
        ], dtype=np.float32)



        done =   (self.step_count >= self.max_steps )
        info = {'step_Count': self.step_count}

        cost_penalty=0
        #norm_rt5 = np.clip(norm_rt, 0.0, 1.0)
        cost_penalty2= np.clip(cost_penalty, 0, 1)
        norm_util4=np.clip(norm_util3, 0.0, 1.0)
        norm_utilb = np.clip(norm_utilb, 0.0, 1.0)

        # Improvement relative to the trigger RT (init_rt_ms): positive if we reduce RT
        base_rt = getattr(self, 'init_rt_ms', None)
        # if base_rt is None:
        #     base_rt = self.response_time
        # imp = 0.0
        # try:
        #     base_rt = float(base_rt)
        #     if base_rt > 0:
        #         imp = (base_rt - float(self.response_time)) / (base_rt + self.ep)
        #         imp = float(np.clip(imp, -1.0, 1.0))
        # except Exception:
        #     imp = 0.0
        # final_reward4 = self.w_rt * - norm_rt2 \
        #                 + self.w_cost * - cost_penalty2 \
        #                 + self.w_util * (norm_util4) + self.w_rt * imp

        alpha2 =  np.clip((self.current_rpm - self.rpm_low_thr) / (self.rpm_spike_thr - self.rpm_low_thr), 0.0, 1.0)

        final_reward4= alpha2 *  - norm_rt2 \
                        + 0.0 *  - cost_penalty2 \
                        + (1-alpha2) * (norm_util4)




        if (self.response_time >self.tau_ms):
           final_reward4 -= self.w_rt * 3



        log_msg = (
            f"Config={config_id}, "
            #f"users={int(self.current_num_users)}, "
            f"rpm={self.current_rpm:.3f}, "
            f"RT={self.response_time:.1f}, "
            
            
            f"norm_rt={( norm_rt2):.2f}, "
            f"U={norm_util4:.4f}, "
            f"CPU={self.cpu:.1f}, "
            f"Mem={self.mem:.1f}, "
            f"Cost={( cost_penalty2):.4f}, "
            f"F_reward={final_reward4.item():.4f}"
        )
        logger.info(log_msg)
        # (you can remove or keep the old print if you still want console output.txt)
        val_map = {
            'RT': self.response_time,
            'norm_rt': ( norm_rt2),
            'Utility':norm_util4,
            'CPU': self.cpu,
            'Mem': self.mem,

            'Cost': (cost_penalty2),
            'F_reward': final_reward4.item(),
        }
        for key, val in val_map.items():
            rec = self.stats[key]
            if val < rec['min']:
                rec['min'], rec['min_cfg'] = val, config_id
            if val > rec['max']:
                rec['max'], rec['max_cfg'] = val, config_id

        return obs, float(final_reward4.item()), done, False, info

# I have two obs for trying reasons
####################################################################


# ==============================
# Dueling Q-Network
# ==============================
class DuelingQFunction(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_adv = nn.Linear(hidden_size, n_actions)
        self.fc_val = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [norm_rt, workload]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        adv = self.fc_adv(h)
        val = self.fc_val(h)
        adv_mean = adv.mean(dim=1, keepdim=True)
        q = val + (adv - adv_mean)
        return action_value.DiscreteActionValue(q)
#==========================================================================================



# 4) Define policy functions
def ddqn_policy(obs):
    obs_norm = np.array([obs[0]], dtype=np.float32)
    qv = q_func(obs_norm)  # use normalized observation
    return int(qv.greedy_actions.numpy()[0])






# ==============================
# Main Training Loop
# ==============================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s')

    #SEED = 42 # DoubleDQN= 13% 92%
    #SEED = 40 # give 100%
    #SEED =41
    #SEED =41
    #SEED=46 #90
    SEED=45 #94%
    #SEED=44 #98%
    #SEED=43 #89%
    # 94%
    #chnage seed from 40 to 41 reduce regret match deom 80 % to 52% on the parameters yiu see in this file  Date 1/1/2026
    # using DDQN is not good
    #SEED = 39 #
    print("Run for SEED ",SEED)
    set_global_seed(SEED)


    # Load
    #configs = pd.read_csv('NUFinal_DS3.csv') # last DS
    #NUFinal_DS3_updated
    #df=configs = pd.read_csv('Data/Final_DS2.csv')   #old DS (just befor cleaning below)
    #df=configs = pd.read_csv('Data/Final_DS2_V2_C.csv')   #New cleaded  DS
    #BookLib_Final_DS_k6 (1).csv
    df = configs = pd.read_csv('Data/BookLib_Final_DS_k6.csv')
    #startup_df = pd.read_csv("Data/booklib_startup_points_tau_global_p95.csv")
    #booklib_startup2
    startup_df = pd.read_csv("Data/booklib_startup_J.csv")
    config_binary = pd.read_csv("Data/ALL_Config_binary_BookLib.csv")
    eval_df_in = pd.read_csv("Data/booklib_eval_J.csv")
    #booklib_eval_ood
    #eval_df_in = pd.read_csv("Data/booklib_eval_ood_diverse.csv")
    config_ids = configs['config'].unique()

    # Env

    env = CustomD4RLNormalizedEnv(configs,config_binary, startup_df=startup_df)
    env.action_space.seed(SEED)


    rt_col = "avg_response_time_ms"
    cpu_col = "total_cpu"
    mem_col = "total_mem"

    # 3) Compute means and standard deviations
    rt_mean, rt_std = df[rt_col].mean(), df[rt_col].std()
    cpu_mean, cpu_std = df[cpu_col].mean(), df[cpu_col].std()
    mem_mean, mem_std = df[mem_col].mean(), df[mem_col].std()

    # 4) Create z-score columns
    df['norm_rt_z'] = (df[rt_col] - rt_mean) / (rt_std + env.ep)
    df['total_cpu_z'] = (df[cpu_col] - cpu_mean) / (cpu_std + env.ep)
    df['total_mem_z'] = (df[mem_col] - mem_mean) / (mem_std + env.ep)

    # 5) Save the enriched dataset
    df.to_csv('Data/Final_DS2_with_zscores.csv', index=False)
    print("Saved: Data/Final_DS2_with_zscores.csv")






    # Agent
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    q_func = DuelingQFunction(obs_size, n_actions)
    optimizer = torch.optim.Adam(q_func.parameters(), lr=1e-4)
    explorer = explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0, end_epsilon=0.05,
        decay_steps=50000,
        random_action_func=lambda: env.action_space.sample()
    )
    '''
    buffer = replay_buffers.PrioritizedReplayBuffer(
        capacity=100000,
        alpha=0.6,
        beta0=0.4,
        betasteps=120000,  # or total_steps // update_interval
        eps=1e-6,
        normalize_by_max="memory",
    )
    '''
    # Orginal way for buffer
    buffer = replay_buffers.ReplayBuffer(capacity=100000)


    agent = agents.DoubleDQN(
        q_function=q_func,
        optimizer=optimizer,
        replay_buffer=buffer,
        gamma=0.90,
        explorer=explorer,
        replay_start_size=1000,
        gpu=-1,
        target_update_interval=2000,

    )

    # TrainE
    total_steps = 120000# was 120000

    # ---- timing start ----
    start_time = time.time()
    obs = env.reset()
    ep_reward = 0.0
    episode_rewards = []
    for t in range(total_steps):
        action = agent.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.observe(obs, reward, done, reset=done)
        ep_reward += reward
        if done:

            #print("Retun ", ep_reward)
            episode_rewards.append(ep_reward)
            ep_reward = 0.0
            obs = env.reset()
            logging.info('===========+++++++++++++++++========================= Epoisde  Done===================================')

    end_time = time.time()
    total_time_sec = end_time - start_time
    steps_per_sec = total_steps / total_time_sec

    print(total_steps)
    print(f"Total training time: {total_time_sec:.2f} seconds")
    print(f"Steps per second: {steps_per_sec:.1f}")

    logging.info('Training completed')
    print('Training finished')
    print(f'Train mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}')


    #agent.save("Dyn_Online_DDQNUtil_3.pth")
    #agent.replay_buffer.save("Dyn_full_buffer_3.pkl")

    ####Save Final stat ===>Important
    # rt_mean_final, rt_var_final = env.obs_rms.mean[0], env.obs_rms.var[0]
    # cpu_mean_final, cpu_var_final =env.cpu_normalizer.mean[0], env.cpu_normalizer.var[0]
    # mem_mean_final, mem_var_final= env.mem_normalizer.mean[0], env.mem_normalizer.var[0]
    #
    # rows = [
    #     ["metric", "mean", "var"],
    #     ["rt", rt_mean_final, rt_var_final],
    #     ["cpu", cpu_mean_final, cpu_var_final],
    #     ["mem", mem_mean_final, mem_var_final],
    # ]
    #
    # with open("stats8.csv", "w", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(rows)
    #     print(rows)


    s = env.stats
    # after your training loop in main.py
    s = env.stats

    logger.info(
        "TRAINING COMPLETE ➔\n"
        "RT:       min %.1f (cfg %s), max %.1f (cfg %s)\n"
        "norm_rt:  min %.8f (cfg %s), max %.2f (cfg %s)\n"
        "Utility:  min %.4f (cfg %s), max %.4f (cfg %s)\n"
        
         "Mem:  min %.4f (cfg %s), max %.4f (cfg %s)\n"
         "CPU:  min %.4f (cfg %s), max %.4f (cfg %s)\n"
        
        "Cost:     min %.4f (cfg %s), max %.4f (cfg %s)\n"
        "F_reward: min %.4f (cfg %s), max %.4f (cfg %s)",
        s['RT']['min'], s['RT']['min_cfg'],
        s['RT']['max'], s['RT']['max_cfg'],
        s['norm_rt']['min'], s['norm_rt']['min_cfg'],
        s['norm_rt']['max'], s['norm_rt']['max_cfg'],
        s['Utility']['min'], s['Utility']['min_cfg'],  # ← Utility line
        s['Utility']['max'], s['Utility']['max_cfg'],  # ← Utility line

        s['CPU']['min'], s['CPU']['min_cfg'],
        s['CPU']['max'], s['CPU']['max_cfg'],

        s['Mem']['min'], s['Mem']['min_cfg'],
        s['Mem']['max'], s['Mem']['max_cfg'],


        s['Cost']['min'], s['Cost']['min_cfg'],
        s['Cost']['max'], s['Cost']['max_cfg'],
        s['F_reward']['min'], s['F_reward']['min_cfg'],
        s['F_reward']['max'], s['F_reward']['max_cfg'],
    )


    # Plot training rewards per episode
    #plt.figure(figsize=(10, 5))

    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Episode Reward')
    plt.title('BookLIB this is Training Episode Rewards')
    plt.grid(True)
    plt.tight_layout()

    #plt.show()

    window_size = 100

    # Turn your list of episode rewards into a pandas Series
    rewards_series = pd.Series(episode_rewards)

    # Compute the moving average
    smoothed_rewards = rewards_series.rolling(window=window_size, min_periods=1).mean()

    # ---- Convergence detection (episodes until stability) ----
    moving_avg = smoothed_rewards.values

    # Use the last 500 episodes as the plateau (change 500 if you want)
    tail_size = min(500, len(moving_avg))  # in case you have fewer episodes
    plateau_tail = moving_avg[-tail_size:]
    plateau_mean = plateau_tail.mean()

    threshold = 0.9 * plateau_mean  # 90% of plateau

    conv_index = None
    K = 200  # how long we require it to "stay" near plateau

    for i, v in enumerate(moving_avg):
        if v >= threshold:
            tail = moving_avg[i : i + K]
            if len(tail) < K:
                break
            # less than 10% of the next K points are below threshold
            if (tail < threshold).mean() < 0.1:
                conv_index = i
                break

    if conv_index is not None:
        conv_episode = conv_index + window_size - 1  # map to episode index
        print(f"Approx convergence episode: {conv_episode}")
    else:
        print("Convergence episode not found with current settings.")


    # Plot the raw and smoothed curves
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_series, alpha=0.3, label='Raw Episode Reward')
    plt.plot(smoothed_rewards, linewidth=2, label=f'{window_size}-Episode Moving Avg')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Episode Reward')
    plt.title('BookLIB Training Rewards training (Raw vs. Moving Average)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(OUT_DIR, "training_reward.png")
    plt.savefig(filename, dpi=150)
    plt.show()



    #============================    # Evaluate   ============================================
    #agent.load("Dyn_Online_DDQNUtil_3.pth")
    #agent.eval_mode()
    #pfrl.utils.set_random_seed(0)


    agent.explorer.epsilon = 0.0
    n_eval = 50
    rewards = []
    for ep in range(n_eval):
        obs = env.reset()
        total = 0.0
        done = False
        env.returns = 0.0  # reset returns for reward normalization during eval
        while not done:
            #print(obs)
            #qv = q_func(obs)
            #action = int(qv.greedy_actions.numpy()[0])
            action = agent.act(obs)

            obs, raw_reward, terminated, truncated, _ = env.step(action)
            # normalize reward manually for evaluation phase
            env.returns = env.returns * env.gamma + raw_reward
            norm_reward = raw_reward / np.sqrt(env.ret_rms.var + env.epsilon)
            #print("Eval Reward ", norm_reward )
            done = terminated or truncated
            total += norm_reward
        rewards.append(total)
        agent.stop_episode()
    print(f'Eval mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}')
    true_ret_var = env.ret_rms.var
    # Plot with episodes labeled
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_eval + 1), rewards, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Evaluation Rewards per Episode')
    plt.grid(True)
    plt.tight_layout()

    # Turn your list of episode rewards into a pandas Series
    rewards_series = pd.Series(rewards)
    # Compute the moving average
    smoothed_rewards = rewards_series.rolling(window=20, min_periods=1).mean()

    # Plot the raw and smoothed curves
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_series, alpha=0.3, label='Raw Episode Reward')
    plt.plot(smoothed_rewards, linewidth=2, label=f'{window_size}-Episode Moving Avg')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Episode Reward')
    plt.title('Training Rewards (Raw vs. Moving Average)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



    ###============================  Evaluation with Oracle  ===================================================================================


    # 1) Load the oracle mapping we just computed
    eval_df =  pd.read_csv("Data/eval_DS_3.csv")
    #eval_df_in= pd.read_csv("Data/eval_in_DS.csv")
    #eval_in_DS_triggers_only_tau_global_p95
    #eval_df_in = pd.read_csv("Data/Sample_in_100.csv")
    #booklib_eval_2
    #eval_df_in = pd.read_csv("Data/booklib_eval_2.csv")

    eval_df['agent_config'] = np.nan
    eval_df['agent_rew'] = np.nan
    eval_df['oracle_config'] = np.nan
    eval_df['oracle_rew'] = np.nan
    eval_df[ 'Match']= None
    eval_df['Top_Five_Agent'] = None
    eval_df['Top_Five_Oracle']= None
    eval_df['Common_Count']=None
    eval_df['ddqn_latency'] = None


    eval_df_in['agent_config'] = np.nan
    eval_df_in['agent_rew'] = np.nan
    eval_df_in['oracle_config'] = np.nan
    eval_df_in['oracle_rew'] = np.nan
    eval_df_in['Match'] = None
    eval_df_in['Top_Five_Agent'] = None
    eval_df_in['Top_Five_Oracle'] = None
    eval_df_in['Common_Count']=None
    eval_df_in['ddqn_latency'] = None



    def normalize(rpm,  rt):

        # normalize  rpm and num_users  Way1
        norm_rpm = min_max_scale(rpm, env.rpm_min, env.rpm_max)
        #norm_nu = min_max_scale(num_users, env.nu_min, env.nu_max)
        #rt_mean_final, rt_var_final
        #norm_rt = (rt - rt_mean_final) / np.sqrt(rt_var_final + env.ep)
        norm_rt = (rt - env.rt_mean) / (env.rt_std + env.ep)

        return np.array([
            norm_rpm,
            #norm_nu,
            norm_rt
        ], dtype=np.float32)


    def compute_reward_oracle(utility, cost_penalty, norm_rt, RT, rpm):
        # Map z-scored RT to [0,1] (higher is worse)
        norm_rt5 = (np.tanh(norm_rt) + 1) / 2
        cost_penalty2 = np.clip(cost_penalty, 0, 1)
        norm_util4 = np.clip(utility, 0.0, 1.0)

        alpha2 =  np.clip((rpm - env.rpm_low_thr) / (env.rpm_spike_thr - env.rpm_low_thr), 0.0, 1.0)
        #print("rpm ", rpm, "   ", alpha2)
        Oracle_reward = (
                alpha2 * -norm_rt5
            + 0.0 * -cost_penalty2
            +(1-alpha2) * (norm_util4)
        )

        # SLA / trigger penalty
        if RT > env.tau_ms:
            Oracle_reward -= env.w_rt * 3

        # Improvement-based shaping: reward configs that reduce RT relative to the trigger RT (state)
        # if init_rt_ms is not None:
        #     try:
        #         base = float(init_rt_ms)
        #         if base > 0:
        #             imp = (base - float(RT)) / (base + env.ep)  # positive if improved vs trigger
        #             imp = float(np.clip(imp, -1.0, 1.0))
        #             Oracle_reward += env.w_rt * imp
        #     except Exception:
        #         pass

        return Oracle_reward
    ############## Evale#####################################################################################################



    def get_reward_components(cfg,rpm):
       P=False
       flags = env.config_flags[cfg]
       #print(cfg,"    ", flags)
       feat = {
           'config': cfg,
           #'num_users': num_users ,
           'actual_rpm': rpm ,
           **flags,  # merges in Agg, GB1, …, Adv
       }

       df_pred = pd.DataFrame([feat])

       # 1) transform exactly as during training
       X_t = env.preproc.transform(df_pred)

       # 2) predict RT ................
       log_rt = env.RT_model.predict(X_t)[0]
       response_time = float(np.expm1(log_rt))
       # print("Predicted RT ",predicted_rt)
       norm_rt=(response_time - env.rt_mean) / (env.rt_std + env.ep)

       # 3) predict CPU_Cost ................
       cpu = env.CPU_model.predict(X_t)
       cpu = float(np.expm1(cpu).item())
       # print("self.cpu ", self.cpu)
       norm_cpu = (cpu - env.cpu_mean) / (env.cpu_std + env.ep)
       # print("cpu norm ", norm_cpu)

       # 4) predict Mem_Cost ................
       mem = env.Mem_model.predict(X_t)
       mem = float(np.expm1(mem).item())
       norm_mem = (mem - env.mem_mean) / (env.mem_std + env.ep)

       # print("mem norm ", norm_mem)
       cost_penalty = ((norm_cpu + norm_mem) / 2)



       utility = env.config_utils[cfg]  # I used the already normalized

       # if(rpm >  env.rpm_spike_thr): #800):
       #     utility = env.config_utils_B[cfg]

      # if (rpm <=  env.rpm_low_thr) and utility < 0.3: #instead of 200
            #P=True
       return norm_rt,cost_penalty,utility,response_time,mem,cpu



    # #####################################Agent Eval loop
    agent.eval_mode()
    DFs = [eval_df_in]# , eval_df]
    for i, dff in  enumerate(DFs, start=1):
     for idx, row in dff.iterrows():

           norm_state = normalize(row['actual_rpm'],  row['init_rt_ms'])

           t0 = perf_counter()
           with torch.no_grad():
               q_output = q_func(torch.tensor(norm_state, dtype=torch.float32).unsqueeze(0))  # Ensure batch dimension
           # normalize(rpm, num_users, rt):
               action = int(q_output.greedy_actions.numpy()[0])
               ddqn_latency_ms = (perf_counter() - t0) * 1000.0
           #print("action by agent ", action)

               norm_rt, cost_penalty, utility,RT,_,_ = get_reward_components(action+1, row['actual_rpm'])
               #print("Agent ", RT,"   ",cost_penalty,"   ",utility)
               init_rt = row['init_rt_ms'] if 'init_rt_ms' in row else row.get('avg_response_time_ms', None)
               R1 = compute_reward_oracle(utility, cost_penalty, norm_rt,RT,row['actual_rpm'])

               dff.at[idx, 'agent_config'] = action+1
               dff.at[idx, 'agent_rew'] = R1
               dff.at[idx, "ddqn_latency"] = ddqn_latency_ms



           #print("config by agent ",row['actual_rpm'],"  ", row['num_users'],"  ", action+1)



               q_values = q_output.q_values.numpy().flatten()  # Ensure 1D array
               best_action = int(q_output.greedy_actions.numpy()[0])

               # Sort Q-values in descending order and get top 5
               sorted_indices = q_values.argsort()[::-1][:5]
               sorted_q_values = q_values[sorted_indices]

               #print(f"  Greedy Action: {best_action + 1}")  # Adjusted to start from 1
               #print("  Top 5 Q-values (action: value):")
              # for action, value in zip(sorted_indices, sorted_q_values):
                   #print(f"    Action {action + 1}: {value:.4f}")  # Adjusted to start from 1

               top5 = [(action + 1) for action in sorted_indices[:5]]

               dff.at[idx, 'Top_Five_Agent'] = top5






    #######################################################################################################
    #Oracle Loop
    All = []


    for i,dff in  enumerate(DFs, start=1):
     for idx, row in dff.iterrows():  #head(5)

        results = []
        best_R = -float("inf")
        best_cfg = None
        best_row_idx = None
        RTT=None
        c=None
        u=None

        #state = {'actual_rpm': row['actual_rpm'], 'num_users': row['num_users'],'avg_response_time_ms': 5000}
        norm_state = normalize(row['actual_rpm'],   row['init_rt_ms'])

        for cfg in config_ids:
            #print(f"Processing config ID: {cfg}")


            norm_rt,cost_penalty,utility,RT,_,_ = get_reward_components(cfg, row['actual_rpm'])


            init_rt = row['init_rt_ms'] if 'init_rt_ms' in row else row.get('avg_response_time_ms', None)


            R= compute_reward_oracle(utility, cost_penalty, norm_rt,RT, row['actual_rpm'])


            results.append({
                "row_idx": idx,
                "config": cfg,
                "oracle_reward": R
            })

            All.append({
                "row_idx": idx,
                "config": cfg,
                "oracle_reward": R
            })

            # update global best
            if R > best_R:
                best_R = R
                best_cfg = cfg
                best_row_idx = idx
                RTT=RT
                c=cost_penalty
                u=utility


        #print("Oracle ",RTT,"   ",c,"   ",u)



        # after loops, turn into a DataFrame if you like:
        oracle_df = pd.DataFrame(results)

        #oracle_df.to_csv("Data/results_oracle.csv")


        # fill into your original eval_df: best per‐row
        # 1) Compute the best oracle reward *per* eval-row
        best_per_row = oracle_df.groupby("row_idx")["oracle_reward"].max()

        # 2) Create (or overwrite) the oracle_reward column in eval_df by mapping via its index
        #eval_df["oracle_reward"] = eval_df.index.map(best_per_row)

        #print(
            #f"Highest oracle reward = {best_R:.4f}, "
           # f"achieved by config {best_cfg} "
          #  f"on eval_df row index {best_row_idx}"
        #)

        results.sort(key=lambda entry: entry["oracle_reward"], reverse=True)
        configs_only = [entry['config'] for entry in results]
        dff.at[idx, 'Top_Five_Oracle'] = configs_only[:5]
        #print("Top 5 configs by R:", results[:5])
        #oracle_rew,agent_rew
        dff.at[idx, 'oracle_config'] = best_cfg
        dff.at[idx, 'oracle_rew'] = best_R
        #dff.at[idx, 'Match'] = (row['oracle_config'] ==  row['agent_config'])
        dff.at[idx, 'Match'] = (best_cfg == int(dff.at[idx, 'agent_config']))


        #Count match in Top 5

        def parse_top5(x):
            if isinstance(x, list):
                return x
            elif isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    return []
            else:
                return []


        # 2) Apply it to both columns
        dff['Top_Five_Oracle'] = dff['Top_Five_Oracle'].apply(parse_top5)
        dff['Top_Five_Agent'] = dff['Top_Five_Agent'].apply(parse_top5)

        # 3) Compute the count of common configs in one shot
        dff['Common_Count'] = dff.apply(
            lambda row: len(set(row['Top_Five_Oracle']) & set(row['Top_Five_Agent'])),
            axis=1
        )



     dff.to_csv(f"Results2/Oracle_result{i}.csv", index=False)

     ###########################################################################


     ILP_df = pd.DataFrame(All)
     ILP_df.to_csv("Data/LIP_DS.csv")
     print(" DS for ILP is saved to LIP_DS.csv ")


     dff['regret'] = dff['oracle_rew'] - dff['agent_rew']

     # 2) Cumulative regret (optional)
     dff['cum_regret'] = dff['regret'].cumsum()

     # 3) Summary statistics
     mean_regret = dff['regret'].mean()
     median_regret = dff['regret'].median()
     std_regret = dff['regret'].std()
     rmse_regret = np.sqrt((dff['regret'] ** 2).mean())
     zero_regret_pct = (dff['regret'] == 0).mean() * 100
     total_regret = dff['regret'].sum()
     OMA = dff['Match'].mean()
     # Convert both columns to int
     agent_ids = dff['agent_config'].astype(int)
     oracle_ids = dff['oracle_config'].astype(int)

     # Now compare
     matches = agent_ids == oracle_ids

     # Count
     num_matches = matches.sum()
     total_states = len(dff)
     pct_matches = num_matches / total_states * 100

     print(f"Matches: {num_matches} / {total_states} ({pct_matches:.2f}%)")


     print(f"Mean regret:   {mean_regret:.4f}")
     print(f"Median regret: {median_regret:.4f}")
     print(f"Total regret:  {total_regret:.4f}")
     print(f"Std deviation:       {std_regret:.4f}")
     print(f"RMSE of regret:      {rmse_regret:.4f}")
     print(f"States w/ zero regret: {zero_regret_pct:.2f}%")
     #print(f"Match Accuracy:  {OMA:.4f}")

     threshold = 0.05
     pct_below = (dff['regret'] <= threshold).mean() * 100
     print(f"{pct_below:.2f}% of states have regret ≤ {threshold}")

     # Multiple thresholds at once
     thresholds = [0.01, 0.05, 0.1]
     for t in thresholds:
         pct = (dff['regret'] <= t).mean() * 100
         print(f"{pct:.2f}% of states have regret ≤ {t}")

     # 1) Histogram
     plt.figure()
     plt.hist(dff['regret'], bins=50)
     plt.xlabel("Regret per state")
     plt.ylabel("Count")
     plt.title("Histogram of Regret")
     filename = os.path.join(OUT_DIR, f"Histogram{i}.png")
     plt.savefig(filename, dpi=150)
     plt.show()

     # How many “mistake” states to examine
     K = 10

     # Get the K states with largest regret
     top_mistakes = dff.nlargest(K, 'regret')

     # Show the key columns—state identifier plus metrics
     pd.set_option('display.max_columns', None)  # show any number of columns
     pd.set_option('display.width', 0)  # auto-detect width
     print(top_mistakes[['actual_rpm','agent_config','agent_rew','oracle_config','oracle_rew','regret']])

     plt.figure(figsize=(6, 6))
     plt.scatter(dff['oracle_rew'], dff['agent_rew'], alpha=0.2, label='All states')
     plt.scatter(top_mistakes['oracle_rew'], top_mistakes['agent_rew'],
                 color='red', marker='x', s=80, label='Top-K Regret')
     plt.plot([0, 1], [0, 1], 'k--')
     plt.xlabel("Oracle reward")
     plt.ylabel("Agent reward")
     plt.title("Highlighting High-Regret States")
     plt.legend()
     plt.grid(True)
     filename = os.path.join(OUT_DIR, f"Scatter{i}.png")
     plt.savefig(filename, dpi=150)

     plt.show()

     # 2) CDF

    vals = np.sort(dff['regret'])
    cdf = np.arange(1, len(vals) + 1) / len(vals)

    plt.figure()
    plt.plot(vals, cdf, label='CDF of Regret')
    for t in thresholds:
        plt.axvline(t, linestyle='--', label=f"Threshold = {t}")
    plt.xlabel("Regret")
    plt.ylabel("Fraction of States ≤ Regret")
    plt.title("CDF of Regret with Threshold Annotations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(OUT_DIR, f"CDF{i}.png")
    plt.savefig(filename, dpi=150)
    plt.show()


    '''
    # 3) Scatter Oracle vs Agent
    plt.figure()
    plt.scatter(eval_df['oracle_rew'], eval_df['agent_rew'], alpha=0.5)
    plt.plot([0, 1], [0, 1], 'k--')  # identity line
    plt.xlabel("Oracle reward")
    plt.ylabel("Agent reward")
    plt.title("Agent vs Oracle Reward")
    plt.show()

    # 4) Cumulative Regret
    plt.figure()
    plt.plot(eval_df['cum_regret'])
    plt.xlabel("State index (or sorted by regret)")
    plt.ylabel("Cumulative regret")
    plt.title("Cumulative Regret over Evaluation Set")
    plt.show()
    '''
    #Emulator Code
    import pandas as pd

    ga = GA2.GA( config_binary, configs)


    # Inputs
    trace = pd.read_csv("Data/booklib_trace_emulation_40min.csv")  # minute,rpm,phase...
    tau_high = 4600
    tau_low = int(0.8 * tau_high)
    rpm_low = 388

    N = 2  # violation persistence
    M = 2  # relaxed persistence
    cooldown_w = 3  # windows

    # Replace with your baseline config id
    current_cfg = 39

    def predict_rt(rpm, cfg):

        norm_rt, cost_penalty, utility, RT,mem,cpu = get_reward_components(cfg,rpm)
        R = compute_reward_oracle(utility, cost_penalty, norm_rt, RT,rpm)

        return RT,cost_penalty,mem,cpu,R


    def select_config(norm_state):

        with torch.no_grad():
            q_output = q_func(torch.tensor(norm_state, dtype=torch.float32).unsqueeze(0))  # Ensure batch dimension
            # normalize(rpm, num_users, rt):
            action = int(q_output.greedy_actions.numpy()[0])
        return action+1


    viol_count = 0
    relax_count = 0
    cooldown = 0
    best_cfg_ga= 39
    logs = []

    for t, row in trace.iterrows():
        rpm = float(row["rpm"])

        rt,cost,mem,cpu,R = predict_rt(rpm, current_cfg)

        rt_ga ,cost_ga,mem_ga,cpu_ga,R_ga= predict_rt(rpm, best_cfg_ga)

        #ux_now = ux.get(current_cfg, 0.5)
        ux=env.config_utils[current_cfg]
        #print( " Ux now ",ux)
        trigger = "none"

        if cooldown > 0:
            cooldown -= 1
        else:
            # Trigger 1: protect SLO
            if rt > tau_high:
                viol_count += 1
            else:
                viol_count = 0

            if viol_count >= N:

                # Call DDQN
                norm_state = normalize(row['rpm'], rt)
                new_cfg = select_config(norm_state)

                #Call GA
                state2 = {"rpm": rpm, "avg_response_time_ms": rt}
                best_idx, best_R_ga, latency = ga.ga_choose_config(
                    state2, prev_idx=None, pop=20, gens=10,
                    p_cx=0.8, p_mut=0.05, lambda_switch=0.05, seed=45
                )
                nbest_cfg_ga = int(ga.configs.iloc[best_idx]["Config_ID"])
                print(nbest_cfg_ga)

                print("selectd confg ",  new_cfg)
                if new_cfg != current_cfg:
                    current_cfg = new_cfg
                    cooldown = cooldown_w
                    trigger = "protect"

                if best_cfg_ga != nbest_cfg_ga:
                    best_cfg_ga = nbest_cfg_ga
                    #cooldown = cooldown_w

                viol_count = 0
                relax_count = 0
            else:
                # Trigger 2: upgrade UX in relaxed regime
                if (rpm <= rpm_low) and (rt < tau_low):
                    relax_count += 1
                else:
                    relax_count = 0

                if relax_count >= M:
                    norm_state = normalize(row['rpm'], rt)

                    # Call DDQN
                    new_cfg = select_config(norm_state)

                    #Call GA
                    # Call GA
                    state2 = {"rpm": rpm, "avg_response_time_ms": rt}
                    best_idx, best_R_ga, latency = ga.ga_choose_config(
                        state2, prev_idx=None, pop=20, gens=10,
                        p_cx=0.8, p_mut=0.05, lambda_switch=0.05, seed=45
                    )
                    nbest_cfg_ga = int(ga.configs.iloc[best_idx]["Config_ID"])
                    print(nbest_cfg_ga)

                    if best_cfg_ga != nbest_cfg_ga:
                        best_cfg_ga = nbest_cfg_ga

                    if new_cfg != current_cfg:
                        current_cfg = new_cfg
                        cooldown = cooldown_w
                        trigger = "upgrade_ux"
                    relax_count = 0

        # log after possible switch (recompute rt/ux for chosen cfg)
        rt ,cost,mem,cpu,R= predict_rt(rpm, current_cfg)

        rt_ga ,cost_ga,mem_ga,cpu_ga,R_ga= predict_rt(rpm, best_cfg_ga)

        ux_ga= env.config_utils[best_cfg_ga]



        ux = env.config_utils[current_cfg]

        logs.append({
            "minute": int(row.get("minute", t + 1)),
            "rpm": rpm,
            "phase": row.get("phase", ""),
            "config": current_cfg,
            "rt_pred_ms": rt,
            "GA_rt" : rt_ga ,
            "ux": ux,
            "ux_GA": ux_ga,


            "Agent Reward ":R,

            "GA config":best_cfg_ga,
            "GA reward": R_ga,


            "trigger": trigger
        })

    log_df = pd.DataFrame(logs)
    log_df.to_csv("trace_emulation_results.csv", index=False)
    print("Saved: trace_emulation_results.csv")








