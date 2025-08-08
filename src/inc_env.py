"""
DRIFT-RL end-to-end research pipeline
------------------------------------
▸ Dynamic Gymnasium env with action-masking
▸ Curriculum + Maskable-PPO training loop
▸ Three non-RL baselines (EEHE, LEACH-C-HE, Q-Routing)
▸ Evaluation harness producing CSV for plots/paper

Dependencies  (≈PyPI names)

• gymnasium>=0.29
• numpy
• networkx
• stable-baselines3>=2.2
• sb3/-contrib>=2.2  # for MaskablePPO
• pandas
• tqdm
• (optional) torch>=2.0  # installed automatically by SB3

Usage

python drift_rl_pipeline.py train   --epochs 800 --out drift_final.zip
python drift_rl_pipeline.py eval    --model drift_final.zip --out results.csv
python drift_rl_pipeline.py baselines --out baseline_results.csv
"""
from __future__ import annotations
import argparse, json, os, random, time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
N_MAX = 120

def set_global_seed(seed: int) -> None:
    """Synchronise *random* and *numpy* RNGs."""
    random.seed(seed)
    np.random.seed(seed)
# ---------------------------------------------------------------------------
# 1.  Helper: synthetic node‑feature generator
# ---------------------------------------------------------------------------

def get_node_ids_and_features(N: int, rng: np.random.Generator):
    """Return ([node_ids], [dict features]). 8 features / node.
    Features are *already normalised to [0,1]* where sensible.
    """
    # --- hyper-params for normalisation (tweak as your env evolves) ----------
    MAX_NEIGHBOURS   = 8          # highest degree you expect in the topology
    MAX_PACKET_AGE_s = 0.500      # treat 0.5 s as “stale”


    nodes = [f"N{i}" for i in range(N)]
    feats = []
    for _ in nodes:
        feats.append(
            {
                "Current Latency (ms)": rng.uniform(5, 300) / 1000.0,  # 0‑0.3
                "Battery Level (%)": rng.uniform(40, 100) / 100.0,     # 0.4‑1.0
                "Trust Score (0-1)": rng.uniform(0.4, 1.0),            # 0.4‑1.0
                "Aggregation Potential": rng.uniform(0, 1.0),         # 0‑1
                # extra placeholders so Box shape==8
                "Conn Degree":rng.integers(0, MAX_NEIGHBOURS + 1) / MAX_NEIGHBOURS,
                "Encrypt Power": rng.uniform(0.5, 2.0) / 2.0,
                "Buf Load": rng.uniform(0, 1.0),
                "Packet age":rng.uniform(0, MAX_PACKET_AGE_s) / MAX_PACKET_AGE_s,
            }
        )
    return nodes, feats

# ---------------------------------------------------------------------------
# 2.  Gymnasium environment with dynamic topology + action masking
# ---------------------------------------------------------------------------
class INCForwardEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, N=10, graph_seed: int | None = None, p: float = 0.25, max_deg: int = 6):
        super().__init__()
        self.N, self.p, self.max_deg = N, p, max_deg
        self.rng = np.random.default_rng(graph_seed)
        self.nodes, self.node_features = get_node_ids_and_features(N, self.rng)
        self.graph, self.positions, cloud_node = self.generate_trimmed_rgg_graph(N, p, max_deg, graph_seed)
        self.neighbors = dict(self.graph.adjacency())
        self.steps = 0
        self._build_spaces()

    # ------------------------------------------------------------------
    def generate_trimmed_rgg_graph(self, N: int, radius: float = 0.25, max_deg: int = 8, seed: int = None):
        rng = random.Random(seed)
        
        while True:
            G = nx.random_geometric_graph(N, radius=radius, seed=seed)
            if nx.is_connected(G):
                break
            seed = seed + 1 if seed is not None else None
            rng.seed(seed)

        # Degree-capped edge construction with weights
        mapping = {n: f"N{n}" for n in G.nodes}
        G = nx.relabel_nodes(G, mapping)

        neighbors = {n: {} for n in G.nodes}
        for u, v in G.edges():
            if len(neighbors[u]) < max_deg and len(neighbors[v]) < max_deg:
                w = rng.uniform(0.5, 1.0)
                neighbors[u][v] = w
                neighbors[v][u] = w

        # Set weights and trim G to match neighbors dict
        G_trimmed = nx.Graph()
        G_trimmed.add_nodes_from(G.nodes(data=True))
        for u in neighbors:
            for v, w in neighbors[u].items():
                if not G_trimmed.has_edge(u, v):
                    G_trimmed.add_edge(u, v, weight=w)

        # Scale positions to 100x100 grid
        pos = {
            n: (
                round(100 * G.nodes[n]['pos'][0], 2),
                round(100 * G.nodes[n]['pos'][1], 2)
            ) for n in G.nodes
        }
        nx.set_node_attributes(G_trimmed, pos, "pos")

        # Assign roles
        cloud_node = min(
            pos,
            key=lambda n: ((100 - pos[n][0])**2 + (100 - pos[n][1])**2)
        )
        roles = {n: None for n in G_trimmed.nodes}
        # ✅ Correctly assign "router" to the first node (e.g., "N0")
        first_node = list(G_trimmed.nodes)[0]
        roles[first_node] = "router"
        roles[cloud_node] = "cloud"
        nx.set_node_attributes(G_trimmed, roles, "role")

        return G_trimmed, pos, cloud_node


    # ------------------------------------------------------------------

    def _build_spaces(self):
        self.max_actions = self.max_deg  # pad/truncate neighbour list to this
        self.observation_space = spaces.Dict(
            {
                "current_node_idx": spaces.Discrete(N_MAX),
                "node_features": spaces.Box(0.0, 1.0, shape=(self.max_actions, 8), dtype=np.float32),
                "action_mask": spaces.Box(0, 1, (self.max_actions,), dtype=bool),
            }
        )
        self.action_space = spaces.Discrete(self.max_actions)

    # ------------------------------------------------------------------
    def _get_obs(self):
        cur_idx = self.nodes.index(self.current_node)
        nbr_ids = list(self.graph.neighbors(self.current_node))
        feats = []
        for nid in nbr_ids[: self.max_actions]:
            d = self.node_features[self.nodes.index(nid)]
            feats.append(list(d.values()))
        while len(feats) < self.max_actions:
            feats.append([0.0] * 8)
        mask = np.zeros(self.max_actions, bool)
        for i, nid in enumerate(nbr_ids[: self.max_actions]):
            mask[i] = nid not in self.visited
        if not mask.any():                # nothing valid – unblock all neighbours
            mask[: len(nbr_ids)] = True
        return {
            "current_node_idx": cur_idx,
            "node_features": np.asarray(feats, dtype=np.float32),
            "action_mask": mask,
        }

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_node = self.nodes[0]     # put this first
        self.steps = 0
        self.sink = self.nodes[-1]
        self.aggregation_buffer = defaultdict(list)

        self.visited = {self.current_node}    # now all helpers see the right node
        self._path   = [self.current_node]
        self._agg_score = 0.0

        obs = self._get_obs()
        self._last_obs = obs
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, action):
        # ------------------------------------------------------------
        # 0. bookkeeping
        # ------------------------------------------------------------
        self.steps += 1
        max_steps = self.N                      # allow up-to N hops
        nbrs      = self.neighbors[self.current_node]
        nbr_ids   = list(nbrs.keys())

        # ------------------------------------------------------------
        # 1. invalid-action guard  (should never trigger w/ proper mask)
        # ------------------------------------------------------------
        if action >= len(nbr_ids):
            info = {"path": self._path, "agg_score": self._agg_score}
            return self._last_obs, -10.0, True, False, info

        # ------------------------------------------------------------
        # 2. select next node and compute base reward
        # ------------------------------------------------------------
        next_node = nbr_ids[action]

        # (extra safety – should not happen)
        if next_node not in self.nodes:
            info = {"path": self._path, "agg_score": self._agg_score}
            return self._last_obs, -10.0, True, False, info

        # ---- feature lookup ----------------------------------------------------
        feat    = self.node_features[self.nodes.index(next_node)]
        latency = feat["Current Latency (ms)"]     # already 0-0.3
        batt    = feat["Battery Level (%)"]        # 0-1
        trust   = feat["Trust Score (0-1)"]        # 0-1
        aggr    = feat["Aggregation Potential"]    # 0-1
        if aggr > 0.5:
            self.aggregation_buffer[next_node].append(aggr)
        loop = next_node in self.visited
        arrived = next_node == self.sink

        # ------------------------------------------------------------
        # Encryption model (optional)
        # ------------------------------------------------------------
        encrypt_power = feat["Encrypt Power"]
        encryption_energy = 0.03 / encrypt_power
        encryption_delay = 0.02 * (1.0 - encrypt_power)

        # ------------------------------------------------------------
        # Euclidean energy model
        # ------------------------------------------------------------
        pos1 = self.positions[self.current_node]
        pos2 = self.positions[next_node]
        dist = np.linalg.norm(np.array(pos1) - np.array(pos2)) / 100.0  # normalize to [0,1] scale

        TX_ELEC = 0.05   # base electronics energy
        TX_AMP  = 0.001  # amplifier coefficient
        PATH_LOSS_EXP = 2  # or use 4 for harsher wireless env

        energy_tx = TX_ELEC + TX_AMP * (dist ** PATH_LOSS_EXP)
        energy_rx = 0.01
        energy = encryption_energy + energy_tx + energy_rx

        # Update latency to include encryption delay
        latency += encryption_delay

        # Packet penalty based on trust
        pkt_pen = 0.0 if trust > 0.7 else 1.0

        

        # ------------------------------------------------------------
        # 3. loop-penalty & arrival bonus
        # ------------------------------------------------------------
        loop     = next_node in self.visited
        arrived  = next_node == self.sink
        delivered = 0
        #packet_dropped = 0
        if arrived:
            drop_prob = 1.0 - trust  # Lower trust → higher drop chance
            if self.rng.uniform(0, 1) < drop_prob:
                arrived = False  # Packet dropped despite reaching the sink
                #packet_dropped = 1
            else:
                delivered = 1

        
        # base reward
        reward = (
            1.5 * delivered        # strong reward for delivery
            + 1.5 * aggr           # reward aggregation if it helps
            - 1.0 * latency   # penalize high delay
            - 0.5 * (1 - batt)     # encourage energy preservation
            + 1.0 * trust          # encourage trust
            #- 2.0 * packet_dropped # discourage loss
        )

        if loop:
            reward -= 5.0
        if arrived:
            reward += 10.0

        # ------------------------------------------------------------
        # 4. update environment state
        # ------------------------------------------------------------
        self.visited.add(next_node)
        self.current_node = next_node
        self._path.append(next_node)
        self._agg_score += aggr

        # episode termination logic
        done = (
            arrived
            or self.steps >= max_steps
            or len(self.neighbors[self.current_node]) == 0
        )

        # ------------------------------------------------------------
        # 5. observation for the next step
        # ------------------------------------------------------------
        obs = self._get_obs()
        self._last_obs = obs

        # always return info so evaluator can log paths
        info = {
            "path":      self._path,
            "agg_score": self._agg_score,
            "latency":   latency,
            "energy": energy,
            "delivered": int(arrived),
        }
        return obs, reward, done, False, info

# ---------------------------------------------------------------------------
# 3.  Mask wrapper factory (SB3‑co ntrib requirement)
# ---------------------------------------------------------------------------

def mask_fn(env):
    return env._last_obs["action_mask"]

def make_env(**kwargs):
    def _thunk():
        env = INCForwardEnv(**kwargs)
        env = ActionMasker(env, mask_fn)
        return env
    return _thunk

# ---------------------------------------------------------------------------
# 4.  Curriculum schedule generator
# ---------------------------------------------------------------------------

def curriculum(epoch: int, rng: random.Random):
    if epoch < 100:
        return dict(N=rng.randint(10, 15), p=0.25)
    if epoch < 300:
        return dict(N=rng.randint(15, 30), p=0.25)
    if epoch < 600:
        return dict(N=rng.randint(30, 60), p=0.22)
    return dict(N=rng.randint(60, 120), p=0.20)


# ---------------------------------------------------------------------------
# 5.  Non‑RL baseline policies
# ---------------------------------------------------------------------------
@dataclass
class EpisodeStats:
    latency: list[float]
    delivered: int = 0
    sent: int = 0
    energy: float = 0.0
    delay: float = 0.0

class BasePolicy:
    def act(self, obs):
        raise NotImplementedError

class RandomPolicy(BasePolicy):
    def act(self, obs):
        mask = obs["action_mask"]
        valid = np.nonzero(mask)[0]
        return int(np.random.choice(valid))

# TODO: implement EEHEPolicy, LeachPolicy, QLearningPolicy as needed.

from BaselineModels import (
    EEHEPolicy,
    LeachPolicy,
    QLearningPolicy,
    run_policy,
    aggregate_stats,
    default_encoder,
)
# ---------------------------------------------------------------------------
# 7b.  Baseline-only evaluation routine
# ---------------------------------------------------------------------------
def evaluate_baselines(out_csv: str = "baseline_results.csv"):
    rows = []

    # pick the same topologies you test DRIFT-RL on
    for N, seed in TEST_SEEDS:
        env_kwargs = dict(N=N, graph_seed=seed)
        env = make_env(**env_kwargs)()          # unwrap thunk

        # ---- run each baseline for 20 episodes ------------------------------
        baselines = {
            "EEHE":  EEHEPolicy(),
            "LEACH": LeachPolicy(round_len=20),
            "Q-Routing": QLearningPolicy(
                n_actions=env.action_space.n,
                state_encoder=default_encoder,
            ),
        }

        for name, policy in baselines.items():
            stats = aggregate_stats(run_policy(env, policy, episodes=20))
            rows.append({
                "baseline":  name,
                "N":         N,
                "seed":      seed,
                **stats,
            })

        env.close()

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved baseline summary CSV to {out_csv}")

# ---------------------------------------------------------------------------
# 6.  Training loop (Maskable‑PPO)
# ---------------------------------------------------------------------------

def train_drift(output_path: str, epochs: int = 800, steps_per_epoch: int = 10_000, seed: int = 0):
    rng = random.Random(seed)
    vec_envs = []
    model = None
    for epoch in tqdm(range(epochs), desc="training"):
        env_params = curriculum(epoch, rng)
        env_params |= {"graph_seed": epoch}
        vec_env = SubprocVecEnv([make_env(**env_params) for _ in range(8)])
        vec_envs.append(vec_env)
        if model is None:
            model = MaskablePPO(
                        "MultiInputPolicy",
                        vec_env,
                        seed=seed,
                        verbose=0,
                        learning_rate=2.5e-4,
                        batch_size=1024,
                        n_epochs=10,             # PPO default is 10, can increase to 15–20
                        gae_lambda=0.95,
                        gamma=0.99,
                        clip_range=0.2,
                        ent_coef=0.01,
                        vf_coef=0.5,
                        tensorboard_log="runs"
                    )
        model.set_env(vec_env)
        model.learn(total_timesteps=steps_per_epoch, reset_num_timesteps=False, progress_bar=False)
        vec_env.close()
        if (epoch + 1) % 100 == 0:
            ckpt = f"{output_path.rstrip('.zip')}_ep{epoch+1}.zip"
            model.save(ckpt)
    model.save(output_path)
    return output_path

# ---------------------------------------------------------------------------
# 7.  Evaluation routine for a given policy
# ---------------------------------------------------------------------------
TEST_SEEDS = [(nodes, 9000 + i) for nodes in range(20, 120, 10) for i in range(10)]
# ---------------------------------------------------------------------------
# 7.  DRIFT-RL evaluation (summary-only CSV)
# ---------------------------------------------------------------------------
from typing import List, Dict, Any

def evaluate(
        model_path: str,
        *,
        out_csv: str = "rl_results.csv",
        out_json: str = "episode_logs.json",
        episodes_per_seed: int = 20,
    ):
        """Run the trained agent on TEST_SEEDS and output *aggregate* metrics."""
        model = MaskablePPO.load(model_path, device="cpu")

        rows: List[Dict[str, Any]] = []     # → CSV (one per N,seed)
        ep_logs: List[Dict[str, Any]] = []  # → JSON (all episodes)

        for N, seed in TEST_SEEDS:
            env = make_env(N=N, graph_seed=seed)()
            set_global_seed(seed)

            ep_stats: List[EpisodeStats] = []

            for ep in range(episodes_per_seed):
                obs, _ = env.reset()
                done = False
                ep_reward = 0.0
                stats = EpisodeStats(latency=[])

                while not done:
                    action, _ = model.predict(obs, deterministic=True, action_masks=obs["action_mask"])
                    obs, reward, done, truncated, info = env.step(action)
                    ep_reward += reward

                    stats.latency.append(info["latency"])
                    stats.delay += info["latency"]

                    stats.delay += info["latency"]
                    stats.energy += info["energy"]
                    stats.sent += 1
                    stats.delivered += info["delivered"]

                ep_stats.append(stats)
                ep_logs.append(
                    {
                        "N": N,
                        "seed": seed,
                        "episode": ep,
                        "path": info["path"],
                        "agg_score": info["agg_score"],
                        "reward_sum": ep_reward,
                    }
                )

            # --- aggregate once for this (N, seed) ------------------------
            summary = aggregate_stats(ep_stats)
            rows.append(
                {"baseline": "DRIFT-RL", "N": N, "seed": seed, **summary}
            )
            env.close()

        # save summary-only CSV
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved DRIFT-RL summary CSV → {out_csv}")

        # save detailed episode log JSON (if you still want it)
        with open(out_json, "w") as f:
            json.dump(ep_logs, f, indent=2)
        print(f"Detailed episode logs → {out_json}")

''' def evaluate(model_path: str,
             out_csv: str = "results.csv",
             out_json: str = "episode_logs.json",
             episodes_per_seed: int = 20):

    model = MaskablePPO.load(model_path, device="cpu")

    rows = []          # → CSV summary
    ep_logs = []       # → JSON detailed paths

    for N, seed in TEST_SEEDS:
        env = make_env(N=N, graph_seed=seed)()   # unwrap thunk

        for ep in range(episodes_per_seed):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True,action_masks=obs["action_mask"])
                obs, reward, done, truncated, info = env.step(action)
                ep_reward += reward

            # done=True → info contains final path & agg_score
            rows.append({
                "N": N,
                "seed": seed,
                "episode": ep,
                "reward_sum": ep_reward,
            })
            ep_logs.append({
                "N": N,
                "seed": seed,
                "episode": ep,
                "path": info["path"],
                "agg_score": info["agg_score"],
                "reward_sum": ep_reward,
            })

        env.close()

    # save summaries
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved summary CSV to {out_csv}")

    with open(out_json, "w") as f:
        json.dump(ep_logs, f, indent=2)
    print(f"Saved detailed episode logs to {out_json}")'''

# ---------------------------------------------------------------------------
# 8.  CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- train ---------------------------------------------------------
    p_train = sub.add_parser("train", help="Train DRIFT‑RL")
    p_train.add_argument("--epochs", type=int, default=800)
    p_train.add_argument("--out", type=str, default="drift_final.zip")

    # ---- eval ----------------------------------------------------------
    p_eval = sub.add_parser("eval", help="Evaluate saved DRIFT‑RL model")
    p_eval.add_argument("--model", type=str, required=True)
    p_eval.add_argument("--out", type=str, default="rl_results.csv")

    # ---- baselines -----------------------------------------------------
    p_base = sub.add_parser("baselines", help="Run non‑RL baselines")
    p_base.add_argument("--out", type=str, default="baseline_results.csv")

    # ---- all -----------------------------------------------------------
    p_all = sub.add_parser(
        "all",
        help="Run baselines **and** DRIFT‑RL, merge into one CSV",
    )
    p_all.add_argument("--model", type=str, required=True, help="Path to DRIFT‑RL .zip")
    p_all.add_argument("--out", type=str, default="all_results.csv", help="Merged CSV filename")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Dispatch ---------------------------------------------------------
    # ------------------------------------------------------------------
    if args.cmd == "train":
        train_drift(output_path=args.out, epochs=args.epochs)

    elif args.cmd == "eval":
        evaluate(args.model, out_csv=args.out)

    elif args.cmd == "baselines":
        evaluate_baselines(args.out)

    elif args.cmd == "all":
        baseline_csv = "baseline_results.csv"
        rl_csv = "rl_results.csv"

        # 1) Run baselines
        evaluate_baselines(baseline_csv)

        # 2) Run DRIFT‑RL evaluation
        evaluate(args.model, out_csv=rl_csv)

        # 3) Merge CSVs
        base_df = pd.read_csv(baseline_csv)
        rl_df = pd.read_csv(rl_csv)
        rl_df["baseline"] = "DRIFT‑RL"
        merged = pd.concat([base_df, rl_df], ignore_index=True)
        merged.to_csv(args.out, index=False)
        print(f"✅ All results saved to {args.out}")
#python inc_env.py all --model drift_final.zip --out all_results.csv

