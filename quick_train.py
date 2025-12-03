#!/usr/bin/env python3
"""
快速训练脚本 - 生成训练数据用于TensorBoard显示
"""

import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.seven_dof_arm import SevenDOFArmEnv

def make_env():
    """创建环境"""
    def _init():
        env = SevenDOFArmEnv(render_mode=None)
        env = Monitor(env)
        return env
    return _init

def quick_train():
    """快速训练生成数据"""
    print("🚀 开始快速训练...")
    
    # 创建环境
    env = DummyVecEnv([make_env()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # 创建模型
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        ent_coef='auto',
        tensorboard_log="./logs/",
    )
    
    print("📊 开始训练1000步...")
    model.learn(
        total_timesteps=1000,
        tb_log_name="quick_train"
    )
    
    print("✅ 训练完成！")
    env.close()

if __name__ == "__main__":
    os.makedirs("./logs/", exist_ok=True)
    quick_train()
