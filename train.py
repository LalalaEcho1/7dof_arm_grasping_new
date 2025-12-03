import os
import time
import yaml
import argparse
import numpy as np
import torch
import gymnasium as gym  # 确保导入 gymnasium
import multiprocessing as mp

# 引入 TensorBoard 必要的库
from torch.utils.tensorboard import SummaryWriter

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

torch.set_num_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

# 假设你的环境文件在这里
from envs.seven_dof_arm import SevenDOFArmEnv


# ==========================================
# 1. 新增：全能 TensorBoard 可视化回调
# ==========================================
class TensorboardCallback(BaseCallback):
    """
    集成视频录制、Q值直方图、动作分布监控的回调函数
    """
    def __init__(self, make_env_fn, video_freq=50000, debug_freq=2000, verbose=0):
        super().__init__(verbose)
        self.video_freq = video_freq
        self.debug_freq = debug_freq
        self.make_env_fn = make_env_fn
        self.eval_env = None  # 用于录制视频的独立环境

    def _init_callback(self) -> None:
        # 初始化一个用于评估和录制的独立环境 (DummyVecEnv)
        # 注意：这里必须使用 render_mode='rgb_array'
        self.eval_env = DummyVecEnv([self.make_env_fn(render_mode='rgb_array', render_every=1)])
        
        # 如果训练环境使用了 VecNormalize，评估环境也必须包裹 VecNormalize
        # 但评估时不需要更新统计数据 (training=False)，也不需要归一化奖励 (norm_reward=False)
        if isinstance(self.training_env, VecNormalize):
            self.eval_env = VecNormalize(self.eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    def _on_step(self) -> bool:
        # --- A. 记录直方图 (Q值, 动作, 权重) ---
        if self.num_timesteps % self.debug_freq == 0:
            self._log_histograms()

        # --- B. 录制视频 ---
        if self.num_timesteps % self.video_freq == 0:
            self._record_video()

        return True

    def _log_histograms(self):
        # 获取当前的 logger (SB3 的 logger 并没有直接暴露 add_histogram，我们需要拿到底层的 writer)
        # 注意：这里通过一种 hack 的方式获取 TensorBoard writer
        tb_writer = None
        for format in self.logger.output_formats:
            if format.__class__.__name__ == 'TensorBoardOutputFormat':
                tb_writer = format.writer
                break
        
        if tb_writer is not None:
            with torch.no_grad():
                # 1. 动作分布
                # 获取当前 batch 的动作 (来自 replay buffer 或当前 step)
                # 为了简单，我们直接用当前策略对当前 observation 预测一次
                obs = self.locals['new_obs'] # 获取当前步的 observation
                if isinstance(obs, dict): # 处理 Dict spaces
                    pass 
                else:
                    obs_tensor = torch.as_tensor(obs).to(self.model.device)
                    actions, _ = self.model.actor(obs_tensor) # 预测动作
                    
                    tb_writer.add_histogram('Debug/Action_Distribution', actions, self.num_timesteps)

                    # 2. Q 值分布 (检查 Critic 是否过估计)
                    # 使用 Critic 网络评估这些动作
                    q1_values, q2_values = self.model.critic(obs_tensor, actions)
                    tb_writer.add_histogram('Debug/Q_Values', q1_values, self.num_timesteps)

                    # 3. 记录 Critic Loss (虽然 SB3 自带，但这里可以做更细致的检查)
                    # (SB3 默认已经记录了 train/critic_loss，这里不做重复)

    def _record_video(self):
        if self.verbose > 0:
            print(f"🎥 [TensorboardCallback] 正在录制视频 @ Step {self.num_timesteps}...")

        # >>> 关键步骤：同步 VecNormalize 的统计数据 <<<
        # 如果训练环境是归一化的，评估环境必须拥有相同的均值和方差，否则机器人像是在"盲人摸象"
        if isinstance(self.training_env, VecNormalize):
            # 从并行环境同步 obs_rms 到评估环境
            self.eval_env.obs_rms = self.training_env.obs_rms

        screens = []
        obs = self.eval_env.reset()
        done = False
        
        # 运行一个完整回合
        while True:
            # 渲染帧 (H, W, C)
            # 对于 DummyVecEnv，render 返回的是 list of arrays
            img = self.eval_env.render() 
            if isinstance(img, list): img = img[0] # 取第一个环境的图像
            screens.append(img)

            # 确定性动作选择
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, infos = self.eval_env.step(action)
            
            if done[0]: # DummyVecEnv 返回的是 done 数组
                break

        # 转换格式为 TensorBoard 需要的 (N, T, C, H, W)
        # screens: (T, H, W, C) -> (1, T, C, H, W)
        if len(screens) > 0:
            screens_np = np.array(screens)
            # 转换为 Tensor: [T, H, W, C] -> [T, C, H, W]
            video_tensor = torch.from_numpy(screens_np).permute(0, 3, 1, 2).unsqueeze(0)
            
            # 获取 writer 并写入
            for format in self.logger.output_formats:
                if format.__class__.__name__ == 'TensorBoardOutputFormat':
                    format.writer.add_video('Rollout/Video', video_tensor, self.num_timesteps, fps=30)
                    format.writer.flush()
                    break

    def _on_training_end(self) -> None:
        if self.eval_env is not None:
            self.eval_env.close()


# ==========================================
# 2. 原有的课程学习回调 (保持不变)
# ==========================================
class CurriculumCallback(BaseCallback):
    """
    基于训练进度的课程学习回调
    """
    def __init__(self, total_timesteps, update_every_steps=500, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = float(total_timesteps)
        self.update_every_steps = int(update_every_steps)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.update_every_steps != 0:
            return True

        progress = float(self.num_timesteps) / max(1.0, self.total_timesteps)

        if self.verbose >= 2:
            print(f"\n[Curriculum] 步数: {self.num_timesteps}, 进度: {progress:.2%}")

        try:
            self.training_env.env_method("set_training_progress", progress)
        except Exception as e:
            print(f"[CurriculumCallback] ⚠️ 更新环境参数失败: {e}")

        return True


class EpisodeLoggerCallback(BaseCallback):
    """
    Episode日志回调 - 仅用于控制台输出
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                ep_r = info['episode'].get('r', 0.0)
                ep_l = info['episode'].get('l', 0)
                if self.verbose >= 1:
                    print(f"  > Ep End: R={ep_r:.1f}, L={ep_l}")
        return True


# ==========================================
# 3. 环境工厂与训练逻辑
# ==========================================

def make_env_builder(render_mode=None, render_every=1000):
    """工厂函数，返回一个创建环境的函数"""
    def _init():
        # 注意：render_mode 必须在这里传递给 SevenDOFArmEnv
        env = SevenDOFArmEnv(render_mode=render_mode, model_path="franka/panda.xml", render_every=render_every)
        env = Monitor(env)
        return env
    return _init

def train(config_path="config.yaml", visualize_after=False):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = int(config.get("seed", 42))
    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("🔍 测试环境...")
    # 测试环境创建
    test_env = make_env_builder(render_mode='rgb_array')()
    test_env.reset()
    test_env.close()
    del test_env
    print("✅ 环境测试通过！\n")

    # 创建并行训练环境
    num_cpu = 4 # 建议根据 CPU 核心数调整，2 可能太少
    # 注意：训练环境通常不需要 render_mode (为了速度)，除非为了调试
    env = SubprocVecEnv([make_env_builder(render_mode=None) for _ in range(num_cpu)], start_method='spawn')
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=True)

    print(f"✅ 环境配置: SubprocVecEnv(n={num_cpu}), VecNormalize=True\n")

    total_timesteps = int(config.get("train_params", {}).get("total_timesteps", 2000000))
    learning_rate = float(config["model_params"]["learning_rate"])

    print(f"🚀 开始训练 | 进度驱动课程学习")

    policy_kwargs = config.get("policy_kwargs", {"net_arch": [256, 256]})
    ent_coef_conf = config["model_params"].get("ent_coef", "auto")
    
    # 处理 ent_coef 类型转换
    if isinstance(ent_coef_conf, str) and ent_coef_conf != "auto":
        try:
            ent_coef_value = float(ent_coef_conf)
        except ValueError:
            ent_coef_value = "auto"
    else:
        ent_coef_value = ent_coef_conf

    model = SAC(
        "MlpPolicy",
        env,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=1,
        learning_rate=learning_rate,
        buffer_size=int(config["model_params"]["buffer_size"]),
        batch_size=int(config["model_params"]["batch_size"]),
        gamma=float(config["model_params"].get("gamma", 0.99)),
        tau=float(config["model_params"].get("tau", 0.01)),
        ent_coef=ent_coef_value,
        tensorboard_log="./logs/",
        policy_kwargs=policy_kwargs,
        target_update_interval=int(config.get("target_update_interval", 1)),
    )

    # 回调配置
    curriculum_ctrl = config.get("curriculum_control", {})
    update_every = int(curriculum_ctrl.get("update_every_steps", 500))

    # --- 配置可视化回调 ---
    # video_freq: 比如每 50,000 步录制一次
    # debug_freq: 每 2,000 步记录一次 Q 值分布
    tensorboard_cb = TensorboardCallback(
        make_env_fn=make_env_builder, 
        video_freq=50000, 
        debug_freq=2000,
        verbose=1
    )

    callbacks = [
        CheckpointCallback(save_freq=int(config["callbacks"]["checkpoint_freq"]), save_path="./models/", name_prefix="sac_7dof"),
        CurriculumCallback(total_timesteps, update_every_steps=update_every, verbose=1),
        EpisodeLoggerCallback(verbose=1),
        tensorboard_cb  # <--- 添加新的回调
    ]

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        tb_log_name="sac_7dof",
        log_interval=100,
    )

    os.makedirs("./models/", exist_ok=True)
    model.save("./models/sac_7dof_final")
    try:
        env.save("./models/vec_normalize.pkl")
    except Exception:
        pass
    env.close()

    print("\n✅ 训练完成！")

    if visualize_after:
        _visualize_model("./models/sac_7dof_final.zip")


def _visualize_model(model_path, episodes=3):
    print(f"\n🎮 可视化演示: {model_path}")
    
    # 演示环境必须开启 render_mode='human'
    vis_env = make_env_builder(render_mode='human', render_every=1)()
    
    # 正确的加载方式：必须加载训练时的 VecNormalize 统计数据
    # 否则演示时的动作会非常鬼畜
    if os.path.exists("./models/vec_normalize.pkl"):
        print("   加载 VecNormalize 统计数据...")
        # 必须用 DummyVecEnv 包裹才能使用 VecNormalize.load
        vis_env = DummyVecEnv([lambda: vis_env]) 
        vis_env = VecNormalize.load("./models/vec_normalize.pkl", vis_env)
        vis_env.training = False # 测试模式，不更新统计
        vis_env.norm_reward = False
    else:
        print("⚠️ 未找到 vec_normalize.pkl，使用原始观测值 (可能导致演示效果极差)")

    model = SAC.load(model_path)

    for ep in range(episodes):
        obs = vis_env.reset() # VecNormalize 包裹后 reset 不需要 _
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = vis_env.step(action)
            total_reward += reward
            time.sleep(0.02)
        print(f"演示 Ep {ep + 1}: Reward={total_reward[0]:.2f}") # VecEnv 返回数组
    vis_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--visualize-after", action="store_true")
    args = parser.parse_args()

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    train(config_path=args.config, visualize_after=args.visualize_after)
