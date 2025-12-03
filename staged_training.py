#!/usr/bin/env python3
"""
分阶段训练脚本 - 渐进式提高抓取难度
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
        
        # 更新课程学习进度
        if hasattr(self, '_training_env') and hasattr(self._training_env, 'envs'):
            progress = self.num_timesteps / self._total_timesteps
            for env in self._training_env.envs:
                if hasattr(env, 'env') and hasattr(env.env, 'set_training_progress'):
                    env.env.set_training_progress(progress)
        
        return True

class SuccessRateCallback(BaseCallback):
    """监控成功率的回调"""
    def __init__(self, eval_freq=1000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.success_count = 0
        self.total_episodes = 0
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # 评估当前成功率
            success_rate = self._evaluate_success_rate()
            print(f"[Step {self.n_calls}] 当前成功率: {success_rate:.1%}")
            
            # 如果成功率足够高，可以调整成功条件
            if success_rate > 0.8 and hasattr(self.training_env, 'envs'):
                self._adjust_success_condition()
        
        return True
    
    def _evaluate_success_rate(self):
        """评估成功率"""
        # 这里可以添加更复杂的评估逻辑
        return self.success_count / max(self.total_episodes, 1)
    
    def _adjust_success_condition(self):
        """调整成功条件"""
        print("🎯 成功率超过80%，建议调整成功条件！")
        # 这里可以动态调整环境的成功条件

def make_env(render_mode=None):
    def _init():
        env = SevenDOFArmEnv(render_mode=render_mode, model_path='franka/panda.xml')
        env = Monitor(env)
        return env
    return _init

def train_stage(stage_name, timesteps, success_threshold=None):
    """分阶段训练"""
    print(f"\n🚀 开始 {stage_name} 训练...")
    
    # 创建环境
    env = DummyVecEnv([make_env(render_mode=None)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    # 如果指定了成功阈值，调整环境
    if success_threshold:
        for env_wrapper in env.envs:
            if hasattr(env_wrapper.env, 'success_threshold'):
                env_wrapper.env.success_threshold = success_threshold
                print(f"设置成功阈值: {success_threshold}")
    
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
    
    success_callback = SuccessRateCallback(eval_freq=1000)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=timesteps // 4,  # 保存4次
        save_path=f"./models/{stage_name}/",
        name_prefix="sac_7dof"
    )
    
    # 开始训练
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback, callback, success_callback],
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
    train_stage("stage1_approach", timesteps=50000, success_threshold=0.5)
    
    # 阶段2：精确接近 (距离<0.3m)
    print("\n" + "="*50)
    print("阶段2：精确接近")
    print("目标：更精确地接近目标")
    print("成功条件：距离 < 0.3m")
    print("="*50)
    train_stage("stage2_precise", timesteps=50000, success_threshold=0.3)
    
    # 阶段3：接触训练 (单指接触)
    print("\n" + "="*50)
    print("阶段3：接触训练")
    print("目标：学会接触目标物体")
    print("成功条件：单指接触")
    print("="*50)
    train_stage("stage3_contact", timesteps=50000, success_threshold="contact")
    
    # 阶段4：抓取训练 (双指接触)
    print("\n" + "="*50)
    print("阶段4：抓取训练")
    print("目标：学会抓取目标物体")
    print("成功条件：双指接触")
    print("="*50)
    train_stage("stage4_grasp", timesteps=50000, success_threshold="grasp")
    
    # 阶段5：抬升训练 (抓取+抬升)
    print("\n" + "="*50)
    print("阶段5：抬升训练")
    print("目标：学会抓取并抬升物体")
    print("成功条件：双指接触 + 抬升")
    print("="*50)
    train_stage("stage5_lift", timesteps=50000, success_threshold="lift")
    
    print("\n🎉 所有阶段训练完成！")

if __name__ == "__main__":
    main()
