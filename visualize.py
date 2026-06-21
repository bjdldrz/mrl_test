"""
可视化脚本
==========
用法:
    # 可视化一次完整训练运行的所有图表
    python visualize.py --run_dir runs/mrl_dms_xxxxx

    # 仅可视化调度甘特图 (需要 checkpoint)
    python visualize.py --run_dir runs/mrl_dms_xxxxx --plot gantt --checkpoint checkpoints/best.pt

    # 对比多次运行
    python visualize.py --run_dirs runs/run1 runs/run2 --labels MRL-DMS PPO

生成的图表保存在 <run_dir>/plots/ 目录下。
"""

import argparse
import os
import sys
import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger("visualize")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------------------------------------------------------
# 通用样式
# -----------------------------------------------------------------------
COLORS = {
    "MRL-DMS":  "#2196F3",
    "PPO":      "#FF9800",
    "A2C":      "#4CAF50",
    "DQN":      "#9C27B0",
    "routine":  "#42A5F5",
    "dynamic":  "#EF5350",
    "idle":     "#E0E0E0",
    "meta_loss":"#78909C",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "figure.dpi":       120,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})


# -----------------------------------------------------------------------
# 1. 训练曲线：Reward + Meta-Loss + Dynamic Rate
# -----------------------------------------------------------------------
def plot_training_curves(train_log: pd.DataFrame, out_dir: Path, label: str = "MRL-DMS"):
    """从 train_log.csv 画训练过程曲线（3 子图）。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Training Curves — {label}", fontsize=14, y=1.02)

    iters = train_log["iter"]

    # (a) avg_reward
    ax = axes[0]
    ax.plot(iters, train_log["avg_reward"], color=COLORS[label] if label in COLORS else "#2196F3",
            linewidth=1.2, alpha=0.6, label="per-iter")
    window = min(10, max(1, len(train_log) // 5))
    smoothed = train_log["avg_reward"].rolling(window, min_periods=1).mean()
    ax.plot(iters, smoothed, color="black", linewidth=2, label=f"MA-{window}")
    ax.set_title("Average Reward")
    ax.set_xlabel("Meta-Iteration")
    ax.set_ylabel("Reward")
    ax.legend()

    # (b) meta_loss
    ax = axes[1]
    ax.plot(iters, train_log["meta_loss"], color=COLORS["meta_loss"],
            linewidth=1.2, alpha=0.6, label="per-iter")
    smoothed_loss = train_log["meta_loss"].rolling(window, min_periods=1).mean()
    ax.plot(iters, smoothed_loss, color="black", linewidth=2, label=f"MA-{window}")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("Meta Loss")
    ax.set_xlabel("Meta-Iteration")
    ax.set_ylabel("Loss")
    ax.legend()

    # (c) avg_dynamic_rate
    ax = axes[2]
    ax.plot(iters, train_log["avg_dynamic_rate"] * 100,
            color="#EF5350", linewidth=1.2, alpha=0.6, label="per-iter")
    smoothed_dyn = (train_log["avg_dynamic_rate"] * 100).rolling(window, min_periods=1).mean()
    ax.plot(iters, smoothed_dyn, color="black", linewidth=2, label=f"MA-{window}")
    ax.set_title("Dynamic Mission Completion Rate")
    ax.set_xlabel("Meta-Iteration")
    ax.set_ylabel("Completion Rate (%)")
    ax.set_ylim(0, 100)
    ax.legend()

    fig.tight_layout()
    path = out_dir / "training_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"已保存: {path}")


# -----------------------------------------------------------------------
# 2. 评估曲线：eval_log.csv 中的多指标
# -----------------------------------------------------------------------
def plot_eval_curves(eval_log: pd.DataFrame, out_dir: Path, label: str = "MRL-DMS"):
    """从 eval_log.csv 画评估指标随迭代变化的曲线。"""
    if eval_log.empty:
        logger.warning("eval_log.csv 为空，跳过评估曲线绘制")
        return

    metrics = [
        ("total_reward",              "Total Reward",              None),
        ("observation_success_rate",  "Observation Success Rate",  (0, 1)),
        ("dynamic_completion_rate",   "Dynamic Completion Rate",   (0, 1)),
        ("routine_completion_rate",   "Routine Completion Rate",   (0, 1)),
        ("n_scheduled",               "Tasks Scheduled",           None),
    ]
    # 仅保留 eval_log 中存在的列
    metrics = [(col, title, ylim) for col, title, ylim in metrics if col in eval_log.columns]

    ncols = 3
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).flatten()

    color = COLORS.get(label, "#2196F3")
    for ax, (col, title, ylim) in zip(axes, metrics):
        ax.plot(eval_log["iter"], eval_log[col], "o-", color=color,
                linewidth=1.8, markersize=5)
        ax.set_title(title)
        ax.set_xlabel("Meta-Iteration")
        if ylim:
            ax.set_ylim(*ylim)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(f"Evaluation Metrics — {label}", fontsize=14)
    fig.tight_layout()
    path = out_dir / "eval_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"已保存: {path}")


# -----------------------------------------------------------------------
# 3. 调度甘特图（单星）
# -----------------------------------------------------------------------
def plot_gantt_single(schedule_log, sat_name: str, horizon_s: float,
                      out_dir: Path, missions=None):
    """
    绘制单颗卫星的调度甘特图。

    参数
    ----
    schedule_log : List[ScheduleRecord]
        环境的 env.schedule_log
    sat_name : str
    horizon_s : float
        规划周期（秒），用于设置 x 轴
    missions : list, optional
        Mission 对象列表，用于区分 routine/dynamic 任务类型
    """
    if not schedule_log:
        logger.warning(f"{sat_name}: schedule_log 为空，跳过甘特图")
        return

    # 用 mission_id 判断类型（需 missions 列表）
    dynamic_ids = set()
    if missions:
        dynamic_ids = {m.id for m in missions if m is not None and m.is_dynamic}

    fig, ax = plt.subplots(figsize=(16, 3.5))
    y = 0.3  # 单行，高度 0.4

    for rec in schedule_log:
        start_h = rec.obs_start_s / 3600
        dur_h   = (rec.obs_end_s - rec.obs_start_s) / 3600
        color = COLORS["dynamic"] if rec.mission_id in dynamic_ids else COLORS["routine"]
        ax.barh(y, dur_h, left=start_h, height=0.4, color=color, edgecolor="white", linewidth=0.4)

    ax.set_xlim(0, horizon_s / 3600)
    ax.set_ylim(0, 0.8)
    ax.set_yticks([y])
    ax.set_yticklabels([sat_name])
    ax.set_xlabel("Time (hours)")
    ax.set_title(f"Scheduling Plan — {sat_name}  |  "
                 f"{len(schedule_log)} tasks scheduled")

    legend_patches = [
        mpatches.Patch(color=COLORS["routine"], label="Routine"),
        mpatches.Patch(color=COLORS["dynamic"], label="Dynamic"),
    ]
    ax.legend(handles=legend_patches, loc="upper right")

    fig.tight_layout()
    path = out_dir / f"gantt_{sat_name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"已保存: {path}")


def plot_gantt_multi(sat_logs: dict, horizon_s: float, out_dir: Path, dynamic_ids: set = None):
    """
    多星甘特图：每行一颗卫星。

    参数
    ----
    sat_logs : {sat_name: List[ScheduleRecord]}
    """
    if not sat_logs:
        return

    dynamic_ids = dynamic_ids or set()
    n_sats = len(sat_logs)
    fig, ax = plt.subplots(figsize=(16, max(3, n_sats * 1.2)))

    sat_names = sorted(sat_logs.keys())
    yticks, ylabels = [], []

    for i, sat_name in enumerate(sat_names):
        y = i
        yticks.append(y)
        ylabels.append(sat_name)
        for rec in sat_logs[sat_name]:
            start_h = rec.obs_start_s / 3600
            dur_h   = (rec.obs_end_s - rec.obs_start_s) / 3600
            color = COLORS["dynamic"] if rec.mission_id in dynamic_ids else COLORS["routine"]
            ax.barh(y, dur_h, left=start_h, height=0.6, color=color,
                    edgecolor="white", linewidth=0.3)

    ax.set_xlim(0, horizon_s / 3600)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time (hours)")
    ax.set_title("Multi-Satellite Scheduling Plan")

    legend_patches = [
        mpatches.Patch(color=COLORS["routine"], label="Routine"),
        mpatches.Patch(color=COLORS["dynamic"], label="Dynamic"),
    ]
    ax.legend(handles=legend_patches, loc="upper right")
    fig.tight_layout()
    path = out_dir / "gantt_multi.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"已保存: {path}")


# -----------------------------------------------------------------------
# 4. 完成率对比柱状图（多算法）
# -----------------------------------------------------------------------
def plot_completion_comparison(results: dict, out_dir: Path):
    """
    对比多个算法的完成率指标。

    参数
    ----
    results : {algo_name: {metric: value}}
        例如 {"MRL-DMS": {"dynamic_completion_rate": 0.72, ...}, "PPO": {...}}
    """
    metrics_to_plot = [
        ("observation_success_rate", "Observation Success Rate"),
        ("dynamic_completion_rate",  "Dynamic Completion Rate"),
        ("routine_completion_rate",  "Routine Completion Rate"),
    ]
    metrics_to_plot = [(k, t) for k, t in metrics_to_plot
                       if any(k in v for v in results.values())]

    if not metrics_to_plot:
        logger.warning("results 中无完成率字段，跳过完成率对比图")
        return

    algo_names = list(results.keys())
    n_algos = len(algo_names)
    n_metrics = len(metrics_to_plot)

    x = np.arange(n_metrics)
    width = 0.8 / n_algos

    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 2), 5))
    for i, algo in enumerate(algo_names):
        vals = [results[algo].get(k, 0) * 100 for k, _ in metrics_to_plot]
        bars = ax.bar(x + i * width - (n_algos - 1) * width / 2, vals,
                      width, label=algo,
                      color=COLORS.get(algo, f"C{i}"), alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([t for _, t in metrics_to_plot], rotation=10, ha="right")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("Algorithm Comparison — Completion Rates")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "completion_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"已保存: {path}")


# -----------------------------------------------------------------------
# 5. 卫星利用率对比
# -----------------------------------------------------------------------
def plot_satellite_utilization(sat_logs: dict, horizon_s: float, out_dir: Path):
    """
    绘制各卫星的时间利用率（观测时间占总规划周期的比例）。

    参数
    ----
    sat_logs : {sat_name: List[ScheduleRecord]}
    """
    if not sat_logs:
        return

    sat_names = sorted(sat_logs.keys())
    utils = []
    n_tasks = []
    for name in sat_names:
        total_obs = sum(r.obs_end_s - r.obs_start_s for r in sat_logs[name])
        utils.append(total_obs / horizon_s * 100)
        n_tasks.append(len(sat_logs[name]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # (a) 利用率
    ax = axes[0]
    bars = ax.bar(sat_names, utils, color=COLORS["routine"], alpha=0.85)
    for bar, v in zip(bars, utils):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Utilization (%)")
    ax.set_title("Satellite Time Utilization")
    ax.tick_params(axis="x", rotation=30)

    # (b) 调度任务数
    ax = axes[1]
    bars = ax.bar(sat_names, n_tasks, color=COLORS["dynamic"], alpha=0.85)
    for bar, v in zip(bars, n_tasks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Tasks Scheduled")
    ax.set_title("Tasks per Satellite")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    path = out_dir / "satellite_utilization.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"已保存: {path}")


# -----------------------------------------------------------------------
# 6. 任务规模 vs 完成率热力图
# -----------------------------------------------------------------------
def plot_scale_heatmap(scale_results: list, out_dir: Path, metric: str = "dynamic_completion_rate"):
    """
    绘制 (routine_size × dynamic_size) → metric 的热力图。

    参数
    ----
    scale_results : [{n_routine, n_dynamic, <metric>}, ...]
    """
    if not scale_results:
        return

    df = pd.DataFrame(scale_results)
    if metric not in df.columns:
        logger.warning(f"scale_results 中无 {metric} 列，跳过热力图")
        return

    pivot = df.groupby(["n_routine", "n_dynamic"])[metric].mean().unstack()

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values * 100, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label=f"{metric} (%)")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(r) for r in pivot.index])
    ax.set_xlabel("Dynamic Tasks per Insertion")
    ax.set_ylabel("Routine Tasks")
    title_map = {
        "dynamic_completion_rate": "Dynamic Completion Rate (%)",
        "observation_success_rate": "Observation Success Rate (%)",
        "routine_completion_rate": "Routine Completion Rate (%)",
    }
    ax.set_title(f"Scale Sensitivity — {title_map.get(metric, metric)}")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val*100:.0f}%", ha="center", va="center",
                        fontsize=9, color="black")

    fig.tight_layout()
    path = out_dir / f"scale_heatmap_{metric}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"已保存: {path}")


# -----------------------------------------------------------------------
# 7. 奖励分布箱线图（多算法对比）
# -----------------------------------------------------------------------
def plot_reward_boxplot(results_episodes: dict, out_dir: Path):
    """
    绘制各算法 episode total_reward 的箱线图。

    参数
    ----
    results_episodes : {algo_name: [total_reward, ...]}
    """
    if not results_episodes:
        return

    algo_names = list(results_episodes.keys())
    data = [results_episodes[a] for a in algo_names]
    colors = [COLORS.get(a, f"C{i}") for i, a in enumerate(algo_names)]

    fig, ax = plt.subplots(figsize=(max(6, len(algo_names) * 2), 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(algo_names) + 1))
    ax.set_xticklabels(algo_names)
    ax.set_ylabel("Total Reward")
    ax.set_title("Reward Distribution by Algorithm")
    fig.tight_layout()
    path = out_dir / "reward_boxplot.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"已保存: {path}")


# -----------------------------------------------------------------------
# 8. 综合仪表板：训练 + 评估 + 最终完成率
# -----------------------------------------------------------------------
def plot_dashboard(train_log: pd.DataFrame, eval_log: pd.DataFrame,
                   final_metrics: dict, out_dir: Path, label: str = "MRL-DMS"):
    """
    4 宫格仪表板，适合论文插图或实验报告。

    子图布局:
        [训练 reward 曲线]  [评估完成率曲线]
        [元损失曲线]        [最终指标雷达图/柱状图]
    """
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle(f"MRL-DMS Dashboard — {label}", fontsize=15)

    color = COLORS.get(label, "#2196F3")
    iters = train_log["iter"]
    window = min(10, max(1, len(train_log) // 5))

    # (1,1) 训练 reward
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(iters, train_log["avg_reward"], color=color, alpha=0.4, linewidth=1)
    smoothed = train_log["avg_reward"].rolling(window, min_periods=1).mean()
    ax1.plot(iters, smoothed, color=color, linewidth=2)
    ax1.set_title("Training Reward")
    ax1.set_xlabel("Meta-Iteration")
    ax1.set_ylabel("Avg Reward")

    # (1,2) 评估完成率曲线
    ax2 = fig.add_subplot(gs[0, 1])
    if not eval_log.empty:
        eval_iters = eval_log["iter"]
        for col, color_k, lbl in [
            ("dynamic_completion_rate", "#EF5350", "Dynamic"),
            ("routine_completion_rate", "#42A5F5", "Routine"),
            ("observation_success_rate", "#4CAF50", "Observation"),
        ]:
            if col in eval_log.columns:
                ax2.plot(eval_iters, eval_log[col] * 100,
                         "o-", color=color_k, linewidth=1.8, markersize=5, label=lbl)
        ax2.set_ylim(0, 100)
        ax2.set_title("Eval Completion Rates")
        ax2.set_xlabel("Meta-Iteration")
        ax2.set_ylabel("Rate (%)")
        ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No eval data yet", ha="center", va="center",
                 transform=ax2.transAxes, color="gray")
        ax2.set_title("Eval Completion Rates")

    # (2,1) Meta Loss
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(iters, train_log["meta_loss"], color=COLORS["meta_loss"], alpha=0.4, linewidth=1)
    smoothed_loss = train_log["meta_loss"].rolling(window, min_periods=1).mean()
    ax3.plot(iters, smoothed_loss, color=COLORS["meta_loss"], linewidth=2)
    ax3.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax3.set_title("Meta Loss")
    ax3.set_xlabel("Meta-Iteration")
    ax3.set_ylabel("Loss")

    # (2,2) 最终指标柱状图
    ax4 = fig.add_subplot(gs[1, 1])
    if final_metrics:
        rate_keys = [
            ("observation_success_rate", "Obs. Rate"),
            ("dynamic_completion_rate",  "Dyn. Rate"),
            ("routine_completion_rate",  "Rtn. Rate"),
        ]
        names = [n for k, n in rate_keys if k in final_metrics]
        vals  = [final_metrics[k] * 100 for k, n in rate_keys if k in final_metrics]
        bar_colors = ["#4CAF50", "#EF5350", "#42A5F5"][:len(names)]
        bars = ax4.bar(names, vals, color=bar_colors, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
        ax4.set_ylim(0, 110)
        ax4.set_title("Final Eval Metrics")
        ax4.set_ylabel("Rate (%)")
    else:
        ax4.text(0.5, 0.5, "No final metrics", ha="center", va="center",
                 transform=ax4.transAxes, color="gray")
        ax4.set_title("Final Eval Metrics")

    path = out_dir / "dashboard.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"已保存: {path}")


# -----------------------------------------------------------------------
# 9. 动态任务响应时间分布
# -----------------------------------------------------------------------
def plot_dynamic_response_dist(schedule_log, missions, out_dir: Path):
    """
    绘制动态任务从到达到完成的响应时间分布（直方图）。
    """
    dynamic_missions = {m.id: m for m in missions if m is not None and m.is_dynamic}
    if not dynamic_missions:
        return

    response_times = []
    for rec in schedule_log:
        if rec.mission_id in dynamic_missions:
            m = dynamic_missions[rec.mission_id]
            if hasattr(m, "arrival_time_s") and m.arrival_time_s is not None:
                response_times.append((rec.obs_start_s - m.arrival_time_s) / 60)

    if not response_times:
        logger.warning("无法计算动态任务响应时间（缺少 arrival_time_s 字段）")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(response_times, bins=20, color=COLORS["dynamic"], alpha=0.75, edgecolor="white")
    ax.axvline(np.median(response_times), color="black", linestyle="--",
               linewidth=1.5, label=f"Median: {np.median(response_times):.1f} min")
    ax.set_xlabel("Response Time (minutes)")
    ax.set_ylabel("Count")
    ax.set_title("Dynamic Mission Response Time Distribution")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "dynamic_response_dist.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"已保存: {path}")


# -----------------------------------------------------------------------
# 主函数：读取日志 + 批量生成所有图表
# -----------------------------------------------------------------------
def visualize_run(run_dir: Path, checkpoint: str = None, label: str = "MRL-DMS"):
    """根据一次运行的 run_dir 自动生成所有可生成的图表。"""
    out_dir = run_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    # --- 读取日志 ---
    train_log_path = run_dir / "train_log.csv"
    eval_log_path  = run_dir / "eval_log.csv"

    train_log = pd.read_csv(train_log_path) if train_log_path.exists() else pd.DataFrame()
    eval_log  = pd.read_csv(eval_log_path)  if eval_log_path.exists()  else pd.DataFrame()

    if train_log.empty:
        logger.warning(f"{train_log_path} 不存在或为空，跳过训练曲线")
    else:
        plot_training_curves(train_log, out_dir, label=label)

    plot_eval_curves(eval_log, out_dir, label=label)

    # --- 仪表板 ---
    final_metrics = {}
    if not eval_log.empty:
        final_metrics = eval_log.iloc[-1].to_dict()
    if not train_log.empty:
        plot_dashboard(train_log, eval_log, final_metrics, out_dir, label=label)

    # --- 需要 checkpoint 的图表 ---
    if checkpoint:
        _visualize_with_checkpoint(checkpoint, out_dir)
    else:
        logger.info("未指定 --checkpoint，跳过甘特图和卫星利用率图")

    logger.info(f"全部图表已保存到: {out_dir}")


def _visualize_with_checkpoint(checkpoint: str, out_dir: Path):
    """加载 checkpoint，跑一个 episode，生成调度相关图表。"""
    import torch
    from config import get_default_config
    from algo.mrl_dms import MRLDMSTrainer

    logger.info(f"加载 checkpoint: {checkpoint}")
    config = get_default_config()
    trainer = MRLDMSTrainer(config)
    trainer.setup_data()

    try:
        trainer.load_checkpoint(checkpoint)
    except Exception as e:
        logger.error(f"加载 checkpoint 失败: {e}")
        return

    # 生成一批任务
    routine, dynamic = trainer.mission_gen.generate_episode_missions(
        n_routine=200, n_dynamic_per_insertion=50,
    )
    trainer._precompute_task_vtw(routine, dynamic)

    horizon_s = config.mission.schedule_horizon_hours * 3600

    if trainer.multi_agent:
        # 多星模式
        multi_env = trainer._multi_env
        obs_dict, info_dict = multi_env.reset(options={
            "routine_missions": copy.deepcopy(routine),
            "dynamic_schedule": copy.deepcopy(dynamic),
        })
        actor = trainer._mappo_model.actor
        actor.eval()
        done = False
        with torch.no_grad():
            while not done:
                actions = {}
                for aid, obs in obs_dict.items():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
                    mask  = info_dict.get(aid, {}).get("action_mask",
                                                       np.ones(multi_env.action_dim))
                    mask_t = torch.FloatTensor(mask).unsqueeze(0).to(trainer.device)
                    dist, _ = actor(obs_t, mask_t)
                    actions[aid] = dist.sample().cpu().item()
                obs_dict, _, term_dict, trunc_dict, info_dict = multi_env.step(actions)
                done = all(term_dict.get(a, False) or trunc_dict.get(a, False)
                           for a in obs_dict)

        # 收集各星调度记录
        sat_logs = {}
        dynamic_ids = {m.id for m in routine + [m for _, ms in dynamic for m in ms]
                       if m.is_dynamic} if routine else set()
        for sat_name, env in multi_env.envs.items():
            sat_logs[sat_name] = env.schedule_log
        all_missions = routine + [m for _, ms in dynamic for m in ms]
        dynamic_ids = {m.id for m in all_missions if m.is_dynamic}

        plot_gantt_multi(sat_logs, horizon_s, out_dir, dynamic_ids)
        plot_satellite_utilization(sat_logs, horizon_s, out_dir)

    else:
        # 单星模式
        env = trainer.envs[0]
        obs, info = env.reset(options={
            "routine_missions": copy.deepcopy(routine),
            "dynamic_schedule": copy.deepcopy(dynamic),
        })
        trainer.actor_critic.eval()
        done = False
        with torch.no_grad():
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
                mask  = info.get("action_mask", np.ones(env.action_space.n))
                mask_t = torch.FloatTensor(mask).unsqueeze(0).to(trainer.device)
                dist, _ = trainer.actor_critic(obs_t, mask_t)
                action = dist.sample().cpu().item()
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

        all_missions = routine + [m for _, ms in dynamic for m in ms]
        plot_gantt_single(env.schedule_log, env.sat_config.name,
                          horizon_s, out_dir, all_missions)
        plot_dynamic_response_dist(env.schedule_log, all_missions, out_dir)
        sat_logs = {env.sat_config.name: env.schedule_log}
        plot_satellite_utilization(sat_logs, horizon_s, out_dir)


def compare_runs(run_dirs: list, labels: list, out_dir: Path):
    """对比多次训练运行（不同算法 / 不同超参数）的 eval 曲线。"""
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Multi-Run Comparison", fontsize=14)

    metrics = [
        ("dynamic_completion_rate",  "Dynamic Completion Rate (%)", axes[0]),
        ("routine_completion_rate",  "Routine Completion Rate (%)",  axes[1]),
        ("total_reward",             "Total Reward",                  axes[2]),
    ]

    for run_dir, label in zip(run_dirs, labels):
        eval_path = Path(run_dir) / "eval_log.csv"
        if not eval_path.exists():
            logger.warning(f"{eval_path} 不存在，跳过")
            continue
        df = pd.read_csv(eval_path)
        if df.empty:
            continue
        color = COLORS.get(label, None)
        for col, ylabel, ax in metrics:
            if col not in df.columns:
                continue
            vals = df[col] * 100 if col != "total_reward" else df[col]
            ax.plot(df["iter"], vals, "o-", label=label, color=color,
                    linewidth=1.8, markersize=5)

    for col, ylabel, ax in metrics:
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Meta-Iteration")
        ax.legend()
        if col != "total_reward":
            ax.set_ylim(0, 100)

    fig.tight_layout()
    path = out_dir / "multi_run_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"已保存: {path}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MRL-DMS Visualization")
    parser.add_argument("--run_dir",  type=str, default=None,
                        help="单次运行的日志目录, 例如 runs/mrl_dms_xxxxx")
    parser.add_argument("--run_dirs", type=str, nargs="+", default=None,
                        help="多次运行目录, 与 --labels 配合使用")
    parser.add_argument("--labels",   type=str, nargs="+", default=None,
                        help="与 --run_dirs 对应的标签, 例如 MRL-DMS PPO")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型 checkpoint 路径, 用于生成甘特图")
    parser.add_argument("--label",    type=str, default="MRL-DMS",
                        help="单次运行的标签 (默认 MRL-DMS)")
    parser.add_argument("--out_dir",  type=str, default=None,
                        help="多运行对比图的输出目录 (默认 plots/)")
    args = parser.parse_args()

    if args.run_dirs:
        labels = args.labels or [f"run{i}" for i in range(len(args.run_dirs))]
        out = Path(args.out_dir) if args.out_dir else Path("plots")
        compare_runs(args.run_dirs, labels, out)
    elif args.run_dir:
        visualize_run(Path(args.run_dir), checkpoint=args.checkpoint, label=args.label)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
