"""
环境诊断脚本：系统性地检查环境设置
"""
import numpy as np
import mujoco
from envs.seven_dof_arm import SevenDOFArmEnv

print("=" * 70)
print("🔍 环境诊断：系统性检查")
print("=" * 70)
print()

# ========== 1. 基础设置检查 ==========
print("1️⃣ 基础设置检查")
print("-" * 70)

env = SevenDOFArmEnv(render_mode=None)
obs, info = env.reset()

# 检查初始位置
eef_pos = env.data.site_xpos[env.end_effector_id].copy()
obj_pos = env.data.xpos[env.target_body_id].copy()
distance = np.linalg.norm(eef_pos - obj_pos)
height_diff = eef_pos[2] - obj_pos[2]
horizontal_error = np.linalg.norm(eef_pos[:2] - obj_pos[:2])

print(f"✅ 目标物体位置: {obj_pos}")
print(f"✅ 末端执行器位置: {eef_pos}")
print(f"✅ 初始距离: {distance:.4f} m")
print(f"✅ 高度差: {height_diff:.4f} m (目标: 0.10m)")
print(f"✅ 水平误差: {horizontal_error:.4f} m (目标: 0.00m)")

if abs(height_diff - 0.10) > 0.02:
    print(f"⚠️  警告: 高度差偏离目标超过2cm!")
if horizontal_error > 0.02:
    print(f"⚠️  警告: 水平误差超过2cm!")

print()

# ========== 2. 奖励函数检查 ==========
print("2️⃣ 奖励函数检查")
print("-" * 70)

# 测试不同距离下的奖励
test_distances = [0.30, 0.20, 0.15, 0.12, 0.08, 0.05, 0.03, 0.01]
print("距离 -> 奖励值:")
for dist in test_distances:
    # 模拟距离
    eef_test = obj_pos + np.array([dist, 0, 0])
    env.data.site_xpos[env.end_effector_id] = eef_test
    mujoco.mj_forward(env.model, env.data)
    
    # 计算奖励（简化版，只计算距离相关奖励）
    distance_reward = env.reward_weights['distance'] * np.exp(-3.0 * dist)
    approach_reward = np.clip(env.reward_weights['approach'] - 1.2 * dist, 0.0, env.reward_weights['approach'])
    
    if dist < 0.08:
        approach_reward += env.reward_weights['approach'] * 2.5
    elif dist < 0.12:
        approach_reward += env.reward_weights['approach'] * 1.5
    elif dist < 0.15:
        approach_reward += env.reward_weights['approach'] * 1.0
    
    proximity_bonus = (
        env.reward_weights['proximity'][0] * 2.5 if dist < 0.05 else
        env.reward_weights['proximity'][0] * 2.0 if dist < 0.08 else
        env.reward_weights['proximity'][0] if dist < 0.12 else
        env.reward_weights['proximity'][1] if dist < 0.20 else
        env.reward_weights['proximity'][2] if dist < 0.30 else 0.0
    )
    
    total_reward = distance_reward + approach_reward + proximity_bonus
    success = dist < 0.05
    
    marker = "✅" if success else "  "
    print(f"{marker} {dist:.2f}m -> {total_reward:.2f} (距离:{distance_reward:.2f}, 接近:{approach_reward:.2f}, 邻近:{proximity_bonus:.2f})")

print()
print("⚠️  注意: 成功条件为 0.05m，但奖励函数在 0.05m 内才有最大奖励")
print("   如果成功率持续为0，可能是:")
print("   1. 初始距离太远，难以在500步内达到0.05m")
print("   2. 奖励信号不够明确，无法引导到0.05m")
print("   3. 动作空间限制，无法精确控制")
print()

# ========== 3. 动作空间检查 ==========
print("3️⃣ 动作空间检查")
print("-" * 70)

action_space = env.action_space
print(f"✅ 动作空间维度: {action_space.shape}")
print(f"✅ 动作空间范围: [{action_space.low}, {action_space.high}]")

# 测试随机动作的影响
print("\n测试随机动作对末端位置的影响:")
for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    new_eef_pos = env.data.site_xpos[env.end_effector_id].copy()
    movement = np.linalg.norm(new_eef_pos - eef_pos)
    print(f"  动作 {i+1}: 末端移动 {movement:.4f} m")
    eef_pos = new_eef_pos

print()

# ========== 4. 成功条件可达性检查 ==========
print("4️⃣ 成功条件可达性检查")
print("-" * 70)

# 重置环境
obs, info = env.reset()
initial_distance = np.linalg.norm(
    env.data.site_xpos[env.end_effector_id] - env.data.xpos[env.target_body_id]
)

print(f"✅ 初始距离: {initial_distance:.4f} m")
print(f"✅ 成功条件: < 0.05 m")
print(f"✅ 需要移动距离: {initial_distance - 0.05:.4f} m")
print(f"✅ 最大步数: {env.max_episode_steps}")

# 计算理论最小步数（假设每步移动0.001m）
min_steps_needed = (initial_distance - 0.05) / 0.001
print(f"✅ 理论最小步数（假设每步0.001m）: {min_steps_needed:.0f}")

if min_steps_needed > env.max_episode_steps * 0.8:
    print(f"⚠️  警告: 理论最小步数接近最大步数，可能难以达到成功条件!")
elif initial_distance > 0.20:
    print(f"⚠️  警告: 初始距离太远（>{0.20}m），可能需要很多步才能接近目标!")

print()

# ========== 5. 观察空间检查 ==========
print("5️⃣ 观察空间检查")
print("-" * 70)

obs_dim = len(obs)
print(f"✅ 观察空间维度: {obs_dim}")
print(f"✅ 观察空间范围: [{env.observation_space.low[0]:.2f}, {env.observation_space.high[0]:.2f}]")

# 检查观察是否包含必要信息
print("\n观察空间组成:")
print("  - 关节位置 (7)")
print("  - 关节速度 (7)")
print("  - 末端位置 (3)")
print("  - 物体位置 (3)")
print("  - 目标位置 (3)")
print("  - 相对速度 (6)")
print("  - 接触标志 (4)")

print()

# ========== 6. 奖励函数与成功条件一致性检查 ==========
print("6️⃣ 奖励函数与成功条件一致性检查")
print("-" * 70)

print("成功条件: distance < 0.05m")
print("奖励函数关键阈值:")
print("  - 0.05m: 最大邻近奖励 (2.5倍)")
print("  - 0.08m: 额外接近奖励 (2.5倍)")
print("  - 0.12m: 额外接近奖励 (1.5倍)")
print("  - 0.15m: 额外接近奖励 (1.0倍)")

print("\n⚠️  潜在问题:")
if initial_distance > 0.15:
    print("  ❌ 初始距离 > 0.15m，无法获得0.15m内的额外奖励")
if initial_distance > 0.12:
    print("  ❌ 初始距离 > 0.12m，无法获得0.12m内的额外奖励")
if initial_distance > 0.08:
    print("  ⚠️  初始距离 > 0.08m，无法获得0.08m内的额外奖励")
if initial_distance > 0.05:
    print("  ⚠️  初始距离 > 0.05m，无法获得最大邻近奖励")

print()

# ========== 7. 建议 ==========
print("=" * 70)
print("💡 诊断建议")
print("=" * 70)
print()

issues = []
if abs(height_diff - 0.10) > 0.02:
    issues.append("1. 初始位置高度不正确")
if horizontal_error > 0.02:
    issues.append("2. 初始位置水平误差过大")
if initial_distance > 0.15:
    issues.append("3. 初始距离太远，难以达到成功条件")
if min_steps_needed > env.max_episode_steps * 0.8:
    issues.append("4. 理论最小步数接近最大步数，时间可能不够")

if issues:
    print("发现的问题:")
    for issue in issues:
        print(f"  ❌ {issue}")
    print()
    print("建议的修复方案:")
    if "初始位置" in str(issues):
        print("  - 检查并修复初始位置优化算法")
    if "初始距离太远" in str(issues) or "理论最小步数" in str(issues):
        print("  - 考虑放宽成功条件到 0.08m 或 0.10m")
        print("  - 或者缩短初始距离到 0.12m 以内")
        print("  - 或者增加最大步数")
    print()
else:
    print("✅ 未发现明显问题")
    print()
    print("如果成功率仍然为0，可能的原因:")
    print("  1. 奖励函数信号不够明确")
    print("  2. 动作空间限制，无法精确控制")
    print("  3. 训练步数不够")
    print("  4. 学习率不合适")
    print("  5. 网络结构不合适")

env.close()
print()
print("=" * 70)
print("诊断完成")
print("=" * 70)

