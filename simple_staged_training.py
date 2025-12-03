#!/usr/bin/env python3
"""
简化的分阶段训练脚本
"""

import os
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.seven_dof_arm import SevenDOFArmEnv

class EpisodeLoggerCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                ep_r = info['episode']['r']
                ep_l = info['episode']['l']
                print(f"[Episode] reward: {ep_r:.2f}, length: {ep_l}")
        return True

def make_env(render_mode=None):
    def _init():
        env = SevenDOFArmEnv(render_mode=render_mode, model_path='franka/panda.xml')
        env = Monitor(env)
        return env
    return _init

def train_stage(stage_name, timesteps, success_mode="distance", threshold=0.5):
    """分阶段训练"""
    print(f"\n🚀 开始 {stage_name} 训练...")
    print(f"成功条件: {success_mode}, 阈值: {threshold}")
    
    # 创建环境
    env = DummyVecEnv([make_env(render_mode=None)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    # 设置成功条件
    for env_wrapper in env.envs:
        if hasattr(env_wrapper.env, 'set_success_condition'):
            env_wrapper.env.set_success_condition(success_mode, threshold)
    
    # 创建模型
    model = SAC(
        "MlpPolicy",
        env,
        device='cuda',
        verbose=2,
        learning_rate=1e-4,
        buffer_size=200000,
        batch_size=512,
        gamma=0.995,
        tau=0.01,
        ent_coef='auto',
        tensorboard_log="./logs/",
        policy_kwargs={"net_arch": [512, 512, 256]},
        target_update_interval=1,
    )
    
    # 回调函数
    callback = EpisodeLoggerCallback()
    callback._training_env = env
    callback._total_timesteps = timesteps
    
    checkpoint_callback = CheckpointCallback(
        save_freq=timesteps // 4,
        save_path=f"./models/{stage_name}/",
        name_prefix="sac_7dof"
    )
    
    # 开始训练
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback, callback],
        tb_log_name=stage_name,
        log_interval=100
    )
    
    # 保存模型
    model.save(f"./models/{stage_name}_final")
    env.save(f"./models/{stage_name}_vec_normalize.pkl")
    
    env.close()
    print(f"✅ {stage_name} 训练完成！")
    
    return model

def main():
    """主训练流程"""
    os.makedirs("./models/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)
    
    print("🎯 开始分阶段抓取训练...")
    
    # 阶段1：接近训练 (距离<0.5m)
    print("\n" + "="*50)
    print("阶段1：接近训练")
    print("目标：学会接近目标物体")
    print("成功条件：距离 < 0.5m")
    print("="*50)
    train_stage("stage1_approach", timesteps=20000, success_mode="distance", threshold=0.5)
    
    # 阶段2：精确接近 (距离<0.3m)
    print("\n" + "="*50)
    print("阶段2：精确接近")
    print("目标：更精确地接近目标")
    print("成功条件：距离 < 0.3m")
    print("="*50)
    train_stage("stage2_precise", timesteps=20000, success_mode="distance", threshold=0.3)
    
    # 阶段3：接触训练 (单指接触)
    print("\n" + "="*50)
    print("阶段3：接触训练")
    print("目标：学会接触目标物体")
    print("成功条件：单指接触")
    print("="*50)
    train_stage("stage3_contact", timesteps=20000, success_mode="contact")
    
    # 阶段4：抓取训练 (双指接触)
    print("\n" + "="*50)
    print("阶段4：抓取训练")
    print("目标：学会抓取目标物体")
    print("成功条件：双指接触")
    print("="*50)
    train_stage("stage4_grasp", timesteps=20000, success_mode="grasp")
    
    print("\n🎉 分阶段训练完成！")

if __name__ == "__main__":
    main()
