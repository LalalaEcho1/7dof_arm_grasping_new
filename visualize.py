#!/usr/bin/env python3
"""
可视化脚本：加载训练好的模型并可视化
"""
import os
import argparse
import time
import numpy as np
from stable_baselines3 import SAC
from envs.seven_dof_arm import SevenDOFArmEnv


def visualize_model(model_path, episodes=5, render_every=1, deterministic=True):
    """
    加载模型并可视化
    
    Args:
        model_path: 模型文件路径
        episodes: 要运行的episode数量
        render_every: 每N步渲染一次
        deterministic: 是否使用确定性策略
    """
    print(f"🎮 加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 创建渲染环境
    env = SevenDOFArmEnv(
        render_mode='human',
        model_path="franka/panda.xml",
        render_every=render_every
    )
    
    # 加载模型
    try:
        model = SAC.load(model_path, env=env)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        env.close()
        return
    
    print(f"📺 开始可视化，运行 {episodes} 个episode")
    print("   按 ESC 或关闭窗口可以退出")
    
    total_rewards = []
    success_count = 0
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        print(f"\n--- Episode {ep + 1}/{episodes} ---")
        
        while not (done or truncated):
            # 使用模型预测动作
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # 渲染（环境会自动处理）
            env.render()
            
            # 控制速度
            time.sleep(0.02)
            
            # 检查是否成功
            if done and info.get("is_success", False):
                success_count += 1
                print(f"  ✅ 成功！步数: {step_count}, 奖励: {total_reward:.2f}")
                break
        
        if not (done and info.get("is_success", False)):
            print(f"  Episode结束: 步数={step_count}, 奖励={total_reward:.2f}")
        
        total_rewards.append(total_reward)
    
    env.close()
    
    # 统计信息
    print(f"\n📊 统计信息:")
    print(f"   成功率: {success_count}/{episodes} ({100*success_count/episodes:.1f}%)")
    print(f"   平均奖励: {np.mean(total_rewards):.2f}")
    print(f"   最大奖励: {np.max(total_rewards):.2f}")
    print(f"   最小奖励: {np.min(total_rewards):.2f}")
    print("✅ 可视化结束")


def list_models():
    """列出所有可用的模型"""
    models_dir = "./models"
    if not os.path.exists(models_dir):
        print("❌ models目录不存在")
        return
    
    models = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
    if not models:
        print("❌ 没有找到模型文件")
        return
    
    print("📁 可用的模型文件:")
    for i, model in enumerate(sorted(models), 1):
        model_path = os.path.join(models_dir, model)
        size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"   {i}. {model} ({size:.1f} MB)")
    
    if models:
        latest = sorted(models)[-1]
        print(f"\n💡 提示: 使用最新模型: ./models/{latest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化训练好的模型")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型文件路径（例如: ./models/sac_7dof_final.zip）"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="要运行的episode数量（默认: 5）"
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="每N步渲染一次（默认: 1，即每步都渲染）"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="使用随机策略（默认使用确定性策略）"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用的模型"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
    elif args.model:
        visualize_model(
            args.model,
            episodes=args.episodes,
            render_every=args.render_every,
            deterministic=not args.stochastic
        )
    else:
        # 尝试使用最新的模型
        models_dir = "./models"
        if os.path.exists(models_dir):
            models = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
            if models:
                latest_model = os.path.join(models_dir, sorted(models)[-1])
                print(f"💡 使用最新模型: {latest_model}")
                visualize_model(
                    latest_model,
                    episodes=args.episodes,
                    render_every=args.render_every,
                    deterministic=not args.stochastic
                )
            else:
                print("❌ 没有找到模型文件")
                print("   使用 --list 查看可用模型")
                print("   使用 --model <path> 指定模型路径")
        else:
            print("❌ models目录不存在")
            print("   请先训练模型，或使用 --model <path> 指定模型路径")

