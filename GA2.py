import csv
import json

import joblib
import numpy as np
import pandas as pd   # needed for DataFrame


from time import perf_counter
import random
from typing import List, Tuple, Any, Dict
import json
import os

from time import perf_counter
import random
from typing import List, Tuple, Any, Dict
import json
import os

class GA:
    def __init__(self,  all_configs,metrics_config):

        stats = {}
        self.configs = all_configs
        self.configs_m = metrics_config
        self.tau_ms = 4600 #float(self.configs_m["avg_response_time_ms"].quantile(0.95))
        self.rpm_spike_thr = int(round(self.configs_m["actual_rpm"].quantile(0.90)))
        self.rpm_low_thr = int(round(self.configs_m["actual_rpm"].quantile(0.10)))

        self.tau_ms = round(float(self.tau_ms), 2)

        flag_cols = ['Agg', 'GB1', 'GB2', 'Inven1', 'Inven2', 'Rev1', 'Rev2', 'Recom1', 'Recom2', 'Adv']
        # In future refactor this code , it seems that you only need flh_cols, in the prediction inside the step mwhere
        # the slef.config_flags code is unnecessary
        self.config_flags = (
            self.configs[['Config_ID'] + flag_cols]
            .drop_duplicates(subset='Config_ID')
            .set_index('Config_ID')[flag_cols]
            .to_dict(orient='index')
        )

        stats_df = pd.read_csv("BookLibStat")  # or "JNNStat.csv" if extension exists
        self.rpm_mean = stats_df["rpm_mean"].iloc[0]
        self.rpm_std = stats_df["rpm_std"].iloc[0]

        self.rt_mean = stats_df["rt_mean"].iloc[0]
        self.rt_std = stats_df["rt_std"].iloc[0]

        self.cpu_mean = stats_df["cpu_mean"].iloc[0]
        self.cpu_std = stats_df["cpu_std"].iloc[0]

        self.mem_mean = stats_df["mem_mean"].iloc[0]
        self.mem_std = stats_df["mem_std"].iloc[0]

        self.utility_mean = stats_df["utility_mean"].iloc[0]
        self.utility_std = stats_df["utility_std"].iloc[0]

        # self.preproc = joblib.load("Modelss/preproc_final5.pkl")
        # self.RT_model = joblib.load("Modelss/RT_model.pkl")
        # self.CPU_model = joblib.load("Modelss/CPU_model.pkl")
        # self.Mem_model = joblib.load("Modelss/Mem_model.pkl")

        self.preproc = joblib.load("Model_Last/Models/BookLib_RT_preproc_binary_3wise24.pkl")
        self.RT_model = joblib.load("Model_Last/Models/BookLib_RT_XGB_binary_3wise24.pkl")
        self.CPU_model = joblib.load("Model_Last/Models/BookLib_CPU_XGB_binary_3wise24.pkl")
        self.Mem_model = joblib.load("Model_Last/Models/BookLib_MEM_XGB_binary_3wise24.pkl")

        # 5) Get utility value for the chosen config
        # with open('Data/utility_result_4.json', 'r') as f:
        #     # load JSON and convert keys to int for easy lookup
        #     raw = json.load(f)
        #     self.config_utils = {int(k): v['utility'] for k, v in raw.items()}
        #     self.config_utils_B = {int(k): v['Utility_Bus_norm'] for k, v in raw.items()}
        #     util_values = np.array(list(self.config_utils.values()))  # convert to array in order to take mean and std

        with open('Data/UX.json', 'r') as f:
            # load JSON and convert keys to int for easy lookup
            raw = json.load(f)
            self.config_utils = {int(k): v['UX'] for k, v in raw.items()}

            util_values = np.array(list(self.config_utils.values()))

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
        self.w_util = 0.3  # weight for QoE
        self.w_cost = 0.3  # weight for cost penalty
        self.ep = 1e-4
        self.action_ids = sorted(self.configs['Config_ID'].unique().tolist())
        self.rpm_vals = np.sort(self.configs_m['actual_rpm'].unique())
        #self.num_users_vals = np.sort(self.configs['num_users'].unique())
        # other way
        self.rpm_min, self.rpm_max = self.rpm_vals.min(), self.rpm_vals.max()
        #self.nu_min, self.nu_max = self.num_users_vals.min(), self.num_users_vals.max()

    def min_max_scale(x, xmin, xmax):
        return (x - xmin) / (xmax - xmin)

    def compute_reward_oracle(self, utility, cost_penalty, norm_rt, RT, rpm):

        # norm_rt = np.clip(norm_rt, 0.0, 1.0)
        norm_rt = (np.tanh(norm_rt) + 1) / 2

        cost_penalty = np.clip(cost_penalty, 0, 1)
        #norm_util = np.clip(utility, 0.0, 1.0)
        alpha2 = np.clip((rpm - self.rpm_low_thr) / (self.rpm_spike_thr - self.rpm_low_thr), 0.0, 1.0)

        Oracle_reward = alpha2 * - norm_rt \
                        + 0 * - cost_penalty \
                        + (1 - alpha2) * (utility)



        if (RT > self.tau_ms):
            Oracle_reward = Oracle_reward - self.w_rt * 3

        return Oracle_reward


    def get_reward_components(self, cfg, rpm):
        # P=False
        flags = self.config_flags[cfg]
        # print(cfg,"    ", flags)
        feat = {
            'Config_ID': cfg,

            'actual_rpm': rpm,
            **flags,  # merges in Agg, GB1, …, Adv
        }

        df_pred = pd.DataFrame([feat])

        # 1) transform exactly as during training
        X_t = self.preproc.transform(df_pred)

        # 2) predict RT ................
        log_rt = self.RT_model.predict(X_t)[0]
        response_time = float(np.expm1(log_rt))
        # print("Predicted RT ",predicted_rt)
        norm_rt = (response_time - self.rt_mean) / (self.rt_std + self.ep)

        # 3) predict CPU_Cost ................
        cpu = self.CPU_model.predict(X_t)
        cpu = float(np.expm1(cpu).item())
        # print("self.cpu ", self.cpu)
        norm_cpu = (cpu - self.cpu_mean) / (self.cpu_std +self.ep)
        # print("cpu norm ", norm_cpu)

        # 4) predict Mem_Cost ................
        mem = self.Mem_model.predict(X_t)
        mem = float(np.expm1(mem).item())
        norm_mem = (mem - self.mem_mean) / (self.mem_std+ self.ep)

        # print("mem norm ", norm_mem)
        cost_penalty = ((norm_cpu + norm_mem) / 2)


        utility = self.config_utils[cfg]  # I used the already normalized



        #print(self.rpm_spike_thr," ")
        #if (rpm > self.rpm_spike_thr):
           # utility = self.config_utils_B[cfg]



        return norm_rt, cost_penalty, utility, response_time


    def normalize(self,rpm,  rt):

        # normalize  rpm and num_users  Way1
        norm_rpm = self.min_max_scale(rpm, self.rpm_min, self.rpm_max)
        #norm_nu = self.min_max_scale(num_users, self.nu_min, self.nu_max)
        # rt_mean_final, rt_var_final
        #norm_rt = (rt - self.rt_mean_final) / np.sqrt(self.rt_var_final + self.ep)

        norm_rt = (rt - self.rt_mean) / (self.rt_std + self.ep)
        norm_rt = (np.tanh(norm_rt) + 1) / 2

        return np.array([
            norm_rpm,

            norm_rt
        ], dtype=np.float32)


    # ---------- GA (Option A over 72 configs) ----------
    def ga_choose_config(
            self,
            state: Dict[str, Any],
            prev_idx: int = None,
            pop: int = 24,
            gens: int = 15,
            p_cx: float = 0.8,
            p_mut: float = 0.05,
            tournament_k: int = 3,
            lambda_switch: float = 0.0,
            seed: int = None,
    ) -> Tuple[int, float, float]:
        """
        GA baseline that searches over self.configs (72 discrete configurations).
        Returns: (best_index, best_reward, decision_latency_ms).
        """
        if seed is not None:
            random.seed(seed)

        N = len(self.configs)
        assert N > 0, "self.configs is empty"

        # initialize population with valid indices
        pop_idx = [random.randrange(N) for _ in range(pop)]
        _cache: Dict[int, float] = {}

        def fitness(idx: int) -> float:
            if idx in _cache:
                return _cache[idx]

            cfg = self.configs.iloc[idx]  # get row from DataFrame
            cfg_id = int(cfg["Config_ID"])  # actual config ID

            norm_rt, cost_penalty, utility, RT = self.get_reward_components(
                cfg_id, state["rpm"]
            )

            R = self.compute_reward_oracle(utility, cost_penalty, norm_rt, RT, state["rpm"])

            if prev_idx is not None and idx != prev_idx:
                R -= lambda_switch

            _cache[idx] = R
            return R

        def tournament() -> int:
            cand = random.sample(pop_idx, k=min(tournament_k, len(pop_idx)))
            return max(cand, key=fitness)

        start = perf_counter()
        best = max(pop_idx, key=fitness)

        for _ in range(gens):
            p1, p2 = tournament(), tournament()
            child = p1 if (random.random() > 0.5) else p2

            if random.random() < p_cx:
                mid = int(round((p1 + p2) / 2))
                lo = max(0, mid - 2)
                hi = min(N - 1, mid + 2)
                child = random.randint(lo, hi)

            if random.random() < p_mut:
                child = random.randrange(N)

            worst = min(pop_idx, key=fitness)
            if fitness(child) > fitness(worst):
                pop_idx[pop_idx.index(worst)] = child

            if fitness(child) > fitness(best):
                best = child

        latency_ms = (perf_counter() - start) * 1000.0
        return best, fitness(best), latency_ms


    ###################################################################################################################





import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # === Load eval sets (raw states) ===
    eval_df_in  = pd.read_csv("Data/booklib_eval_FV.csv")
    #eval_df_ood = pd.read_csv("Data/eval_DS_3.csv")    # OOD (500 states)

    # === Load agent results (already computed separately) ===
    agent1 = pd.read_csv("Results2/Oracle_result1.csv")  # matches eval_in_DS


    # Keep only agent info from these
    agent1 = agent1[["init_rt_ms","actual_rpm","agent_config","agent_rew","oracle_config","oracle_rew","ddqn_latency"]]
    #agent2 = agent2[["num_users","actual_rpm","agent_config","agent_rew"]]

    # === Prepare GA ===
    metrics_df = pd.read_csv("Data/BookLib_Final_DS_k6.csv")
    all_cfg_df =  pd.read_csv("Data/ALL_Config_binary_BookLib.csv")


    ga = GA(all_cfg_df, metrics_df)

    config_ids = ga.action_ids

    os.makedirs("Results2", exist_ok=True)

    # === Evaluate Oracle + GA for both datasets ===
    for i, (dff, agent_df) in enumerate([(eval_df_in, agent1)], start=1):
        print(f"\n=== Evaluating Dataset {i} ===")

        # Add missing cols
        for col in ["init_rt_ms","actual_rpm","agent_config","agent_rew","oracle_config","oracle_rew",'ga_config','ga_rew','ga_latency']:
            if col not in dff.columns:
                dff[col] = np.nan

        for idx, row in agent_df.iterrows():
            #state = {"rpm": row['actual_rpm'], "num_users": row['num_users']}
            state = {"rpm": row['actual_rpm'], "avg_response_time_ms": row['init_rt_ms']}
            # # ----- Oracle exhaustive search -----
            # best_R, best_cfg = -float("inf"), None
            # for cfg in config_ids:
            #     norm_rt, cost_penalty, utility, RT, P = ga.get_reward_components(
            #         cfg, row['actual_rpm'], row['num_users']
            #     )
            #     R = ga.compute_reward_oracle(utility, cost_penalty, norm_rt, RT, P)
            #     if R > best_R:
            #         best_R, best_cfg = R, cfg

            # ----- GA search -----
            best_idx, best_R_ga, latency = ga.ga_choose_config(
                state, prev_idx=None, pop=20, gens=10,
                p_cx=0.8, p_mut=0.05, lambda_switch=0.05, seed=45
            )
            best_cfg_ga = int(ga.configs.iloc[best_idx]["Config_ID"])

            # Store results
            dff.at[idx, 'oracle_config'] = row['oracle_config']
            dff.at[idx, 'oracle_rew'] = row['oracle_rew']
            dff.at[idx, 'agent_config'] = row['agent_config']
            dff.at[idx, 'agent_rew'] = row['agent_rew']
            dff.at[idx, 'agent_latency'] = row['ddqn_latency']
            dff.at[idx, 'ga_config']     = best_cfg_ga
            dff.at[idx, 'ga_rew']        = best_R_ga
            dff.at[idx, 'ga_latency']    = float(latency)

        # === Merge with agent results (row-wise, keep 500 rows only) ===
        state1 = {"rpm": 739,"avg_response_time_ms":  34454.7}
        state2 = {"rpm":365, "avg_response_time_ms": 6.3}
        best_idx, best_R_ga, latency = ga.ga_choose_config(
            state1, prev_idx=None, pop=20, gens=10,
            p_cx=0.8, p_mut=0.05, lambda_switch=0.05, seed=45
        )
        best_cfg_ga = int(ga.configs.iloc[best_idx]["Config_ID"])
        print(  best_cfg_ga  )
        print(best_idx, best_R_ga, latency)

        best_idx, best_R_ga, latency = ga.ga_choose_config(
            state2, prev_idx=None, pop=20, gens=10,
            p_cx=0.8, p_mut=0.05, lambda_switch=0.05, seed=45
        )
        best_cfg_ga = int(ga.configs.iloc[best_idx]["Config_ID"])
        print(best_cfg_ga)
        print(best_idx, best_R_ga, latency)

        dff["agent_config"] = agent_df["agent_config"].values
        dff["agent_rew"]    = agent_df["agent_rew"].values

        # === Stats ===
        eps = 1e-9
        # Match rates
        agent_match_rate = 100.0 * (dff["agent_config"] == dff["oracle_config"]).mean()
        ga_match_rate    = 100.0 * (dff["ga_config"]    == dff["oracle_config"]).mean()

        # Regrets
        agent_regret = (dff["oracle_rew"] - dff["agent_rew"]).abs()
        ga_regret    = (dff["oracle_rew"] - dff["ga_rew"]).abs()

        # Function to print stats
        def print_stats(name, regret, match_rate):
            mean_regret   = float(regret.mean())
            median_regret = float(regret.median())
            pct_eq0       = 100.0 * (regret <= eps).mean()
            pct_le_005    = 100.0 * (regret <= 0.05).mean()
            pct_le_010    = 100.0 * (regret <= 0.10).mean()
            print(f"\n=== {name} vs Oracle (Dataset {i}) ===")
            print(f"Mean regret      : {mean_regret:.4f}")
            print(f"Median regret    : {median_regret:.4f}")
            print(f"Regret = 0       : {pct_eq0:.2f}%")
            print(f"Regret ≤ 0.05    : {pct_le_005:.2f}%")
            print(f"Regret ≤ 0.10    : {pct_le_010:.2f}%")
            print(f"Top-1 match rate : {match_rate:.2f}%")

        print_stats("Agent", agent_regret, agent_match_rate)
        print_stats("GA", ga_regret, ga_match_rate)

        # === Plot histogram: side-by-side Agent vs GA regrets ===
        plt.figure(figsize=(6, 4))

        # Use the same bin edges for both
        bins = np.linspace(0, max(ga_regret.max(), agent_regret.max()), 30)
        agent_counts, _ = np.histogram(agent_regret, bins=bins)
        ga_counts, _ = np.histogram(ga_regret, bins=bins)

        # Compute bin centers
        bin_width = bins[1] - bins[0]
        centers = (bins[:-1] + bins[1:]) / 2

        # Plot side-by-side bars
        plt.bar(centers - bin_width / 4, agent_counts, width=bin_width / 2,
                alpha=0.8, label="Agent", edgecolor='black')
        plt.bar(centers + bin_width / 4, ga_counts, width=bin_width / 2,
                alpha=0.8, label="GA", edgecolor='black')

        plt.xlabel("Regret per state")
        plt.ylabel("Count")
        plt.title(f"Regret Histogram (Dataset {i})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Results2/regret_hist_{i}.png", dpi=200)
        plt.close()

        # Save merged dataset
        out_file = f"Results2/Agent_GA_Oracle_result{i}.csv"
        dff.to_csv(out_file, index=False)
        print(f"Saved merged results to {out_file}")




if __name__ == "__main__":
    main()
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # === INPUT ===
    CSV_PATH = f"Results2/Agent_GA_Oracle_result1.csv"

    # === LOAD ===
    df = pd.read_csv(CSV_PATH)


    # Columns in your file:
    # agent_config, agent_rew, ga_config, ga_rew, oracle_config, oracle_rew, actual_rpm, ga_latency

    # === HELPERS ===
    def match_rate(config_col: str) -> float:
        return (df[config_col].astype(int) == df["oracle_config"].astype(int)).mean()


    def regret(rew_col: str) -> np.ndarray:
        # Regret = OracleReward - MethodReward (>=0 ideally, since oracle is best)
        r = (df["oracle_rew"] - df[rew_col]).to_numpy()
        return np.clip(r, 0.0, None)


    def plot_cdf(values: np.ndarray, label: str):
        x = np.sort(values)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, y, label=label)


    # === METRICS ===
    mr_ddqn = match_rate("agent_config")
    mr_ga = match_rate("ga_config")

    reg_ddqn = regret("agent_rew")
    reg_ga = regret("ga_rew")

    summary = pd.DataFrame([
        {
            "method": "DDQN",
            "match_rate": mr_ddqn,
            "regret_mean": float(np.mean(reg_ddqn)),
            "regret_median": float(np.median(reg_ddqn)),
            "regret_p95": float(np.percentile(reg_ddqn, 95)),
        },
        {
            "method": "GA",
            "match_rate": mr_ga,
            "regret_mean": float(np.mean(reg_ga)),
            "regret_median": float(np.median(reg_ga)),
            "regret_p95": float(np.percentile(reg_ga, 95)),
        }
    ])

    print(summary)
    summary.to_csv("rq1_summary.csv", index=False)
    print("Saved: rq1_summary.csv")

    # === PLOT 1: Match-rate bar chart ===
    plt.figure()
    plt.bar(["DDQN", "GA"], [mr_ddqn, mr_ga])
    plt.ylim(0, 1)
    plt.ylabel("Match-rate vs Oracle")
    plt.title("RQ1: Oracle Match-rate (Top-1)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig("rq1_match_rate.png", dpi=200, bbox_inches="tight")
    plt.show()

    # === PLOT 2: Regret CDF ===
    plt.figure()
    plot_cdf(reg_ddqn, "DDQN")
    plot_cdf(reg_ga, "GA")
    plt.xlabel("Regret = OracleReward - MethodReward")
    plt.ylabel("CDF")
    plt.title("RQ1: Regret Distribution vs Oracle (lower is better)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("rq1_regret_cdf.png", dpi=200, bbox_inches="tight")
    plt.show()

    # === PLOT 3: Regret boxplot ===
    plt.figure()
    plt.boxplot([reg_ddqn, reg_ga], labels=["DDQN", "GA"], showfliers=False)
    plt.ylabel("Regret")
    plt.title("RQ1: Regret Summary (Boxplot)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig("rq1_regret_box.png", dpi=200, bbox_inches="tight")
    plt.show()

    # === PLOT 4: Match-rate by RPM bins (workload-aware view) ===
    # Use actual_rpm; create equal-count bins
    bins = 5
    rpm_bins = pd.qcut(df["actual_rpm"], q=bins, duplicates="drop")

    m_ddqn = (df["agent_config"].astype(int) == df["oracle_config"].astype(int)).groupby(rpm_bins).mean()
    m_ga = (df["ga_config"].astype(int) == df["oracle_config"].astype(int)).groupby(rpm_bins).mean()

    plt.figure()
    x = np.arange(len(m_ddqn))
    plt.plot(x, m_ddqn.values, marker="o", label="DDQN")
    plt.plot(x, m_ga.values, marker="o", label="GA")
    plt.ylim(0, 1)
    plt.xticks(x, [str(b) for b in m_ddqn.index], rotation=25, ha="right")
    plt.ylabel("Match-rate vs Oracle")
    plt.title("RQ1: Match-rate across RPM bins")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("rq1_match_by_rpm_bins.png", dpi=200, bbox_inches="tight")
    plt.show()

    # === OPTIONAL: GA latency distribution (since you have ga_latency) ===
    if "ga_latency" in df.columns:
        ga_lat = df["ga_latency"].dropna().to_numpy()

        plt.figure()
        plt.boxplot([ga_lat], labels=["GA latency"], showfliers=False)
        plt.ylabel("Latency (ms)")
        plt.title("RQ1: GA Decision Latency (per decision point)")
        plt.grid(True, axis="y", alpha=0.3)
        plt.savefig("rq1_ga_latency_box.png", dpi=200, bbox_inches="tight")
        plt.show()

        print("GA latency stats (ms):",
              "mean=", float(np.mean(ga_lat)),
              "median=", float(np.median(ga_lat)),
              "p95=", float(np.percentile(ga_lat, 95)))
