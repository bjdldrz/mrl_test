"""
评估指标 (论文 Table 4)
=======================
- Total Reward
- Observation Success Rate
- Dynamic Mission Completion Rate
- Routine Mission Completion Rate
- Computation Time
- Cumulative Dynamic Reward
- Cumulative Routine Reward
"""

import numpy as np
import time
from typing import Dict, List
from collections import defaultdict


class MetricsTracker:
    """训练/评估指标记录与汇总"""

    def __init__(self):
        self.history = defaultdict(list)
        self._timer_start = None

    def start_timer(self):
        self._timer_start = time.time()

    def stop_timer(self) -> float:
        elapsed = time.time() - self._timer_start if self._timer_start else 0.0
        self._timer_start = None
        return elapsed

    def record(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            self.history[k].append(v)

    def get_latest(self) -> Dict[str, float]:
        return {k: v[-1] for k, v in self.history.items() if v}

    def get_average(self, last_n: int = 100) -> Dict[str, float]:
        avg = {}
        for k, v in self.history.items():
            window = v[-last_n:] if len(v) >= last_n else v
            avg[k] = np.mean(window) if window else 0.0
        return avg

    def summary(self) -> str:
        avg = self.get_average()
        lines = [f"  {k}: {v:.4f}" for k, v in sorted(avg.items())]
        return "\n".join(lines)


def compare_algorithms(
    results: Dict[str, List[Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """
    对比多个算法的指标 (对应论文 Table 6)。

    参数
    ----
    results : {algorithm_name: [episode_metrics, ...]}

    返回
    ----
    {algorithm_name: {metric: mean_value}}
    """
    summary = {}
    for algo_name, episodes in results.items():
        if not episodes:
            continue
        keys = episodes[0].keys()
        summary[algo_name] = {
            k: np.mean([ep[k] for ep in episodes])
            for k in keys
        }
    return summary
