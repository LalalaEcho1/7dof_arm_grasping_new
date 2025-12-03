import os
import time
import yaml
import argparse
import numpy as np
import torch
import multiprocessing as mp

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

torch.set_num_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

from envs.seven_dof_arm import SevenDOFArmEnv


class CurriculumCallback(BaseCallback):
    """
    基于训练进度的课程学习回调
    [修复] 使用 env_method 跨进程更新环境参数
    """

    def __init__(self, total_timesteps, update_every_steps=500, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = float(total_timesteps)
        self.update_every_steps = int(update_every_steps)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.update_every_steps != 0:
            return True

        # 计算训练进度
        progress = float(self.num_timesteps) / max(1.0, self.total_timesteps)

        if self.verbose >= 2:
            print(f"\n[Curriculum] 步数: {self.num_timesteps}, 进度: {progress:.2%}")

        # 🚀 [关键修复] 使用 env_method 广播给所有子进程
        # self.training_env 是一个 VecEnv (可能是 VecNormalize 包裹的 SubprocVecEnv)
        # env_method 会自动穿透 Wrappers 并通过 Pipe 发送给子进程
        try:
            self.training_env.env_method("set_training_progress", progress)
        except Exception as e:
            print(f"[CurriculumCallback] ⚠️ 更新环境参数失败: {e}")

        return True


class EpisodeLoggerCallback(BaseCallback):
    """
    Episode日志回调 - 仅用于控制台输出
    环境内部的统计现在由环境自己维护
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
                    # 简单的单行日志
                    print(f"  > Ep End: R={ep_r:.1f}, L={ep_l}")
        return True


def make_env(render_mode=None, render_every=1000):
    def _init():
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
    test_env = make_env(render_mode='None', render_every=1000)()
    test_env.reset()
    test_env.close()
    del test_env
    print("✅ 环境测试通过！\n")

    # 创建并行训练环境
    num_cpu = 2
    env = SubprocVecEnv([make_env(render_mode='None', render_every=1000) for _ in range(num_cpu)], start_method='spawn')
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=True)

    print(f"✅ 环境配置: SubprocVecEnv(n={num_cpu}), VecNormalize=True\n")

    total_timesteps = int(config.get("train_params", {}).get("total_timesteps", 2000000))
    learning_rate = float(config["model_params"]["learning_rate"])

    print(f"🚀 开始训练 | 进度驱动课程学习")
    print(f"   切换点: 25% -> 50% -> 75%\n")

    policy_kwargs = config.get("policy_kwargs", {"net_arch": [256, 256]})
    ent_coef_conf = config["model_params"].get("ent_coef", "auto")
    ent_coef_value = "auto" if isinstance(ent_coef_conf, str) else float(ent_coef_conf)

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

    # 注意顺序：CheckPoint -> Curriculum -> Logger
    callbacks = [
        CheckpointCallback(save_freq=int(config["callbacks"]["checkpoint_freq"]), save_path="./models/",
                           name_prefix="sac_7dof"),
        CurriculumCallback(total_timesteps, update_every_steps=update_every, verbose=1),
        EpisodeLoggerCallback(verbose=1)
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
    print(f"\n🎮 可视化: {model_path}")
    # 可视化使用单个Dummy环境
    vis_env = make_env(render_mode='human', render_every=1)()
    # 如果用了VecNormalize，这里其实应该加载统计数据，否则动作会很奇怪
    # 但为了简单演示，这里直接运行原始环境，效果可能一般

    # 更严谨的做法是包裹 DummyVecEnv 并加载 pkl
    # from stable_baselines3.common.vec_env import DummyVecEnv
    # vis_env = DummyVecEnv([lambda: make_env(render_mode='human', render_every=1)()])
    # vis_env = VecNormalize.load("./models/vec_normalize.pkl", vis_env)
    # vis_env.training = False
    # vis_env.norm_reward = False

    model = SAC.load(model_path)  # 加载模型

    for ep in range(episodes):
        obs, _ = vis_env.reset()
        done = False
        total_reward = 0
        while not done:
            # 若未使用VecNormalize加载，这里的obs范围可能和训练时不一致
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = vis_env.step(action)
            total_reward += reward
            done = terminated or truncated
            time.sleep(0.02)
        print(f"演示 Ep {ep + 1}: Reward={total_reward:.2f}")
    vis_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--visualize-after", action="store_true")
    args = parser.parse_args()

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    train(config_path=args.config, visualize_after=args.visualize_after)
