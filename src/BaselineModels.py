from __future__ import annotations
"""
Baseline routing/forwarding policies and utilities for benchmarking against DRIFT‑RL.

The **INCForwardEnv** observations look like:
```python
{
    "current_node_idx": int,
    "node_features"   : np.ndarray(shape=(max_actions, 8)),
    "action_mask"     : np.ndarray(shape=(max_actions,), dtype=bool),
}
```
`node_features` columns (index → feature):
0 ‒ Current latency (scaled 0‑0.3)
1 ‒ Battery level (0‑1)
2 ‒ Trust score
3 ‒ Aggregation potential
4 ‒ Connectivity degree
5 ‒ Encrypt power
6 ‒ Buffer load
7 ‒ Packet age

No standalone `"battery"` / `"latency"` keys are present, so the baseline
policies must reference those columns directly.
"""

from dataclasses import dataclass
from collections import defaultdict
from typing import Callable, List, Any, Dict
import numpy as np
import random

# ---------------------------------------------------------------------------
#  Metrics helper
# ---------------------------------------------------------------------------
@dataclass
class EpisodeStats:
    latency: List[float]
    delivered: int = 0
    sent: int = 0
    energy: float = 0.0
    delay: float = 0.0

# ---------------------------------------------------------------------------
#  Base & random policies
# ---------------------------------------------------------------------------
class BasePolicy:
    def act(self, obs: Dict[str, Any]) -> int:  # noqa: D401 – simple verb OK
        raise NotImplementedError

    # optional hooks ----------------------------------------------------
    def reset(self):
        pass

    def update(self, *args, **kwargs):
        pass


class RandomPolicy(BasePolicy):
    """Uniform random among currently valid actions."""

    def act(self, obs: Dict[str, Any]) -> int:
        valid = np.nonzero(obs["action_mask"])[0]
        return int(np.random.choice(valid))


# ---------------------------------------------------------------------------
#  EEHE – Energy‑Efficient Heuristic Policy
# ---------------------------------------------------------------------------
class EEHEPolicy(BasePolicy):
    """Battery‑ and latency‑aware greedy heuristic."""

    def __init__(self, w_battery: float = 0.7, w_latency: float = 0.3):
        self.wb = w_battery
        self.wl = w_latency

    def act(self, obs: Dict[str, Any]) -> int:
        mask   = obs["action_mask"]
        valid  = np.nonzero(mask)[0]
        feats  = obs["node_features"][valid]   # (|valid|, 8)

        batt   = feats[:, 1]                    # column 1
        lat    = feats[:, 0]                    # column 0

        batt_norm = (batt - batt.min()) / (np.ptp(batt) + 1e-9)
        lat_norm  = (lat  -  lat.min()) / (np.ptp(lat)  + 1e-9)

        score = self.wb * batt_norm - self.wl * lat_norm
        return int(valid[np.argmax(score)])


# ---------------------------------------------------------------------------
#  LEACH – very compact cluster‑head policy
# ---------------------------------------------------------------------------
class LeachPolicy(BasePolicy):
    """Pick a cluster head every *round_len* steps and forward to it."""

    def __init__(self, round_len: int = 20):
        self.round_len = round_len
        self.cluster_head: int | None = None
        self.t = 0

    def reset(self):
        self.cluster_head = None
        self.t = 0

    def _elect_new_head(self, obs: Dict[str, Any]):
        cand = obs.get("candidate_heads")
        if cand is None or len(cand) == 0:
            cand = np.nonzero(obs["action_mask"])[0]
        self.cluster_head = int(np.random.choice(cand))

    def act(self, obs: Dict[str, Any]) -> int:
        self.t += 1
        if (
            self.cluster_head is None
            or self.t % self.round_len == 0
            or obs["action_mask"][self.cluster_head] == 0
        ):
            self._elect_new_head(obs)
        return self.cluster_head


# ---------------------------------------------------------------------------
#  Q‑learning baseline
# ---------------------------------------------------------------------------
class QLearningPolicy(BasePolicy):
    """Tabular ε‑greedy Q‑learning baseline."""

    def __init__(
        self,
        n_actions: int,
        state_encoder: Callable[[Dict[str, Any]], Any],
        alpha: float = 0.1,
        gamma: float = 0.95,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 1e-4,
    ) -> None:
        self.Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
        self.enc = state_encoder
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.prev_s = None
        self.prev_a = None

    # ------------------------------------------------------------------
    def act(self, obs: Dict[str, Any]) -> int:
        s = self.enc(obs)
        mask = obs["action_mask"]
        if random.random() < self.eps:
            valid = np.nonzero(mask)[0]
            a = int(np.random.choice(valid))
        else:
            q = self.Q[s].copy()
            q[mask == 0] = -np.inf
            a = int(np.argmax(q))
        self.prev_s, self.prev_a = s, a
        return a

    def update(self, next_obs: Dict[str, Any], reward: float, done: bool):
        if self.prev_s is None:
            return
        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[self.enc(next_obs)])
        self.Q[self.prev_s][self.prev_a] += self.alpha * (
            target - self.Q[self.prev_s][self.prev_a]
        )
        if self.eps > self.eps_end:
            self.eps -= self.eps_decay

    def reset(self):
        self.prev_s = None
        self.prev_a = None


# ---------------------------------------------------------------------------
#  Roll‑out & aggregation helpers
# ---------------------------------------------------------------------------

def run_policy(env, policy: BasePolicy, episodes: int = 10) -> List[EpisodeStats]:
    """Roll out *episodes* of *policy* in *env* and collect EpisodeStats."""
    all_stats: List[EpisodeStats] = []
    for _ in range(episodes):
        obs, _info = env.reset()         # Gymnasium reset returns (obs, info)
        policy.reset()
        ep_stats = EpisodeStats(latency=[])
        done = False
        while not done:
            action = policy.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            policy.update(next_obs, reward, done)

            ep_stats.latency.append(info.get("latency", 0))
            ep_stats.delay += info.get("latency", 0)
            ep_stats.sent      += 1
            ep_stats.delivered += int(info.get("delivered", 0))
            ep_stats.energy    += info.get("energy", 0.0)
            obs = next_obs
        all_stats.append(ep_stats)
    return all_stats


def aggregate_stats(stats: List[EpisodeStats]):
    total_sent = sum(s.sent for s in stats)
    total_delivered = sum(s.delivered for s in stats)
    total_energy = sum(s.energy for s in stats)
    flat_latency = [l for s in stats for l in s.latency]
    pdr = (total_delivered / total_sent) if total_sent > 0 else 0.0
    avg_delay = np.mean([s.delay for s in stats]) if stats else 0.0

    return {
        "avg_latency": float(np.mean(flat_latency)) if flat_latency else 0.0,
        "avg_end_to_end_delay": round(avg_delay, 4),
        "pdr": round(pdr, 4),
        "avg_energy": float(np.mean([s.energy for s in stats])) if stats else 0.0,
        "energy_per_packet": total_energy / total_sent if total_sent else 0.0,
    }

# ---------------------------------------------------------------------------
#  Example state encoder (VERY simple – customise!)
# ---------------------------------------------------------------------------

def default_encoder(obs: Dict[str, Any]):
    feats = obs["node_features"]
    batt_bin = int(feats[:, 1].max() * 5)           # bucket battery (0‑1 → 0‑5)
    lat_bin  = int(feats[:, 0].min() / 0.05)        # latency (0‑0.3) → ~6 bins
    return (batt_bin, lat_bin)
