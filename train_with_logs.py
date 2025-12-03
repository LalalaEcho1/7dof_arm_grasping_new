#!/usr/bin/env python3
"""
带详细日志的训练脚本
"""

import os
import yaml
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.seven_dof_arm import SevenDOFArmEnv

class TrainingLogger(BaseCallback):
    """训练日志记录器"""
    
    def __init__(self, verbose=0):
        super(TrainingLogger, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.total_episodes = 0
        self.step_count = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # 记录每个episode的奖励
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'episode' in info:
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.total_episodes += 1
                
                # 检查是否成功
                if info.get('is_success', False):
                    self.success_count += 1
                
                # 打印episode信息
                success_rate = self.success_count / max(1, self.total_episodes)
                print(f"Episode {self.total_episodes}: Reward={episode_reward:.2f}, Length={episode_length}, Success={info.get('is_success', False)}, Success Rate={success_rate:.2%}")
                
                # 每10个episode打印统计信息
                if self.total_episodes % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_length = np.mean(self.episode_lengths[-10:])
                    recent_success_rate = np.mean([1 if info.get('is_success', False) else 0 for info in self.locals.get('infos', [])[-10:]])
                    print(f"--- 最近10个episodes统计 ---")
                    print(f"平均奖励: {avg_reward:.2f}")
                    print(f"平均长度: {avg_length:.1f}")
                    print(f"成功率: {recent_success_rate:.2%}")
                    print(f"总步数: {self.step_count}")
                    print("-" * 40)
        
        return True

def make_env(render_mode=None):
    """封装环境创建"""
    def _init():
        env = SevenDOFArmEnv(render_mode=render_mode, model_path='franka/panda.xml')
        env = Monitor(env)
        return env
    return _init

def train_with_logs():
    """带详细日志的训练"""
    print("🚀 开始带日志训练...")
    
    # 加载配置
    try:
        with open('config.yaml') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("错误：找不到config.yaml文件")
        return
    except yaml.YAMLError as e:
        print(f"错误：配置文件格式错误 - {e}")
        return

    # 创建环境
    try:
        env = DummyVecEnv([make_env(render_mode=None)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
    except Exception as e:
        print(f"错误：创建环境失败 - {e}")
        return

    # 优化学习参数
    learning_rate = 3e-4  # 提高学习率
    buffer_size = 100000
    batch_size = 256
    gamma = 0.99
    tau = 0.005

    # 动态目标熵
    temp_env = make_env()()
    action_dim = temp_env.action_space.shape[0]
    temp_env.close()
    target_entropy = -float(action_dim)

    # 创建模型 - 使用更激进的参数
    model = SAC(
        "MlpPolicy",
        env,
        device='cuda',
        verbose=1,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        ent_coef='auto',
        target_entropy=target_entropy,
        tensorboard_log="./logs/",
        policy_kwargs={
            "net_arch": {"pi": [512, 512], "qf": [512, 512]}  # 更大的网络
        },
        target_update_interval=1,
    )

    # 回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="sac_7dof_logs"
    )

    # 创建评估环境
    eval_env = DummyVecEnv([make_env(render_mode=None)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # 训练日志记录器
    training_logger = TrainingLogger()

    # 开始训练
    print("🚀 开始训练...")
    print(f"📊 总时间步: {config['total_timesteps']}")
    print(f"🎯 学习率: {learning_rate}")
    print(f"📈 TensorBoard日志: ./logs/")
    print(f"💾 模型保存: ./models/")
    print("=" * 50)
    
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=[checkpoint_callback, eval_callback, training_logger],
        tb_log_name="sac_7dof_with_logs"
    )
    
    print("=" * 50)
    print("✅ 训练完成！")
    print(f"📈 总episodes: {training_logger.total_episodes}")
    print(f"🎯 总成功率: {training_logger.success_count / max(1, training_logger.total_episodes):.2%}")
    print(f"📊 平均奖励: {np.mean(training_logger.episode_rewards):.2f}")

    # 保存模型
    model.save("./models/sac_7dof_with_logs_final")
    env.save("./models/vec_normalize_with_logs.pkl")

    # 关闭环境
    env.close()
    eval_env.close()

if __name__ == "__main__":
    # 确保保存目录存在
    os.makedirs("./models/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)
    train_with_logs()
