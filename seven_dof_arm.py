import os
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class SevenDOFArmEnv(gym.Env):
    metadata = {'render_modes': ['None', 'human', 'rgb_array'], 'render_fps': 30}

    def __init__(
            self,
            render_mode='None',
            model_path='franka/panda.xml',
            render_every=1000,
            max_episode_steps=200,  # 改默认值
            is_training=True
    ):
        # ------------------ 基础 ------------------
        self.render_mode = render_mode
        self.render_every = render_every if render_every is not None else 1_000_000
        self._render_counter = 0
        self.np_random = np.random.RandomState()

        # ------------------ 模型 ------------------
        fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise FileNotFoundError(f"Model file not found: {fullpath}")
        self.model = mujoco.MjModel.from_xml_path(fullpath)
        self.data = mujoco.MjData(self.model)

        # ------------------ ID ------------------
        def safe_name2id(objtype, name):
            try:
                return mujoco.mj_name2id(self.model, objtype, name)
            except Exception:
                return -1

        self.end_effector_id = safe_name2id(mujoco.mjtObj.mjOBJ_SITE, "eef_site")
        self.left_tip_site_id = safe_name2id(mujoco.mjtObj.mjOBJ_SITE, "left_tip")
        self.right_tip_site_id = safe_name2id(mujoco.mjtObj.mjOBJ_SITE, "right_tip")
        self.target_geom_id = safe_name2id(mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        self.target_body_id = safe_name2id(mujoco.mjtObj.mjOBJ_BODY, "target_object")

        # 查找夹爪物体ID
        self.left_finger_body_id = safe_name2id(mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_body_id = safe_name2id(mujoco.mjtObj.mjOBJ_BODY, "right_finger")

        # 通过物体ID查找几何体
        self.left_finger_geom_ids = []
        self.right_finger_geom_ids = []
        if self.left_finger_body_id >= 0:
            for i in range(self.model.ngeom):
                if self.model.geom_bodyid[i] == self.left_finger_body_id:
                    self.left_finger_geom_ids.append(i)
        if self.right_finger_body_id >= 0:
            for i in range(self.model.ngeom):
                if self.model.geom_bodyid[i] == self.right_finger_body_id:
                    self.right_finger_geom_ids.append(i)

        self.left_finger_geom_id = self.left_finger_geom_ids[0] if self.left_finger_geom_ids else -1
        self.right_finger_geom_id = self.right_finger_geom_ids[0] if self.right_finger_geom_ids else -1

        # 备用名称
        if self.left_finger_geom_id < 0:
            self.left_finger_geom_id = safe_name2id(mujoco.mjtObj.mjOBJ_GEOM, "left_grip_geom")
            if self.left_finger_geom_id >= 0:
                self.left_finger_geom_ids = [self.left_finger_geom_id]
        if self.right_finger_geom_id < 0:
            self.right_finger_geom_id = safe_name2id(mujoco.mjtObj.mjOBJ_GEOM, "right_grip_geom")
            if self.right_finger_geom_id >= 0:
                self.right_finger_geom_ids = [self.right_finger_geom_id]

        # ------------------ 动作空间 ------------------
        self.arm_actuator_ids = list(range(7))
        self.gripper_actuator_ids = [7]
        self.single_gripper = (len(self.gripper_actuator_ids) == 1)
        action_dim = len(self.arm_actuator_ids) + (1 if self.single_gripper else len(self.gripper_actuator_ids))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        # ------------------ 观察空间 ------------------
        obs_dim = 7 + 7 + 3 + 3 + 3 + 6 + 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # ------------------ 环境状态 ------------------
        self.target_pos = np.zeros(3)
        self.initial_qpos = np.zeros(self.model.nq)
        self.initial_object_height = 0.0
        self.contact_hold_counter = 0
        self.contact_hold_required = 5
        self.max_episode_steps = max_episode_steps
        self.steps = 0
        self.viewer = None

        # ------------------ 🔧 简化的课程学习 ------------------
        self.is_training = is_training
        self.success_mode = "distance"
        self.training_progress = 0.0

        # 成功率统计（仅用于监控）
        self.success_history = deque(maxlen=200)
        self.recent_success_rate = 0.0

        # 机械臂限制
        self.eef_pos_limits = {
            'x': [-0.35, 0.35],
            'y': [-0.35, 0.35],
            'z': [0.0, 0.6]
        }
        self.action_scale_factor = 1.2
        self.joint_angle_limits = np.array([2.0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0])
        self.initial_joint_positions = None

        # 修：重新平衡的奖励权重
        self.reward_weights = {
            'distance': 1.0,  # 降低距离权重
            'approach': 0.0,  # 移除纯接近奖励，防止刷分
            'contact': 5.0,  # 接触给大一点
            'grasp': 10.0,  # 抓取保持给更大
            'lift': 20.0,  # 抬升很大
            'success': 100.0,  #  成功给巨额奖励，确保比任何“刷分”行为都划算
            'step_penalty': -0.1,  # 新增：每步扣分，逼迫它搞快点
            'action_penalty': -0.001,
            'position_penalty': -1.0,
        }

        self.current_episode_success = False

    def _get_obs(self, left_contact=False, right_contact=False, gripper_closed=False, contact_hold_frac=0.0):
        qpos = self.data.qpos[7:14].copy()
        qvel = self.data.qvel[7:14].copy()
        end_effector_pos = self.data.site_xpos[self.end_effector_id].copy() if self.end_effector_id >= 0 else np.zeros(
            3)
        object_pos = self.data.xpos[self.target_body_id].copy() if self.target_body_id >= 0 else np.zeros(3)
        target_pos = self.target_pos.copy()

        end_effector_vel = np.zeros(6)
        if self.end_effector_id >= 0:
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.end_effector_id,
                                     end_effector_vel, 0)
        object_vel = np.zeros(6)
        if self.target_body_id >= 0:
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.target_body_id, object_vel,
                                     0)
        relative_vel = end_effector_vel - object_vel

        contact_flags = np.array(
            [float(left_contact), float(right_contact), float(gripper_closed), float(contact_hold_frac)],
            dtype=np.float32)
        obs = np.concatenate([qpos, qvel, end_effector_pos, object_pos, target_pos, relative_vel, contact_flags])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)

        target_x, target_y, target_z = 0.35, 0.0, 0.025
        self.target_pos = np.array([target_x, target_y, target_z])

        if self.target_body_id >= 0:
            obj_adr = self.model.jnt_qposadr[self.model.body_jntadr[self.target_body_id]]
            self.data.qpos[obj_adr:obj_adr + 3] = self.target_pos
            self.data.qpos[obj_adr + 3:obj_adr + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(self.model, self.data)
            self.initial_object_height = self.data.xpos[self.target_body_id][2]
        else:
            self.initial_object_height = target_z

        target_eef_pos = np.array([target_x - 0.12, target_y, 0.15])
        initial_joint_angles = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, -0.785])

        self.initial_qpos[:] = 0.0
        self.initial_qpos[0:3] = self.target_pos
        self.initial_qpos[3:7] = [1, 0, 0, 0]
        self.initial_qpos[7:14] = initial_joint_angles
        self.initial_qpos[14:16] = [0.04, 0.04]

        self.data.qpos[:] = self.initial_qpos
        mujoco.mj_forward(self.model, self.data)
        self.initial_joint_positions = self.data.qpos[7:14].copy()

        eef = self.data.site_xpos[self.end_effector_id].copy()
        pos_error = eef - target_eef_pos
        total_error = np.linalg.norm(pos_error)

        if total_error > 0.03:
            current_angles = initial_joint_angles.copy()
            learning_rate = 0.1
            max_iter = 50
            for _ in range(max_iter):
                self.data.qpos[7:14] = current_angles
                self.data.qpos[14:16] = [0.04, 0.04]
                mujoco.mj_forward(self.model, self.data)
                eef_current = self.data.site_xpos[self.end_effector_id]
                error_current = eef_current - target_eef_pos
                current_total_error = np.linalg.norm(error_current)
                if current_total_error < 0.015:
                    break
                eps = 0.01
                grad = np.zeros(7)
                for j in [0, 1, 3]:
                    angles_pert = current_angles.copy()
                    angles_pert[j] += eps
                    self.data.qpos[7:14] = angles_pert
                    self.data.qpos[14:16] = [0.04, 0.04]
                    mujoco.mj_forward(self.model, self.data)
                    eef_pert = self.data.site_xpos[self.end_effector_id]
                    error_pert = eef_pert - target_eef_pos
                    grad[j] = (np.linalg.norm(error_pert) - np.linalg.norm(error_current)) / eps
                current_angles[0] -= learning_rate * grad[0] * 0.5
                current_angles[1] -= learning_rate * grad[1]
                current_angles[3] -= learning_rate * grad[3]
                current_angles[0] = np.clip(current_angles[0], -2.0, 2.0)
                current_angles[1] = np.clip(current_angles[1], -1.5, 1.5)
                current_angles[3] = np.clip(current_angles[3], -2.5, 0)
            self.initial_qpos[7:14] = current_angles
            self.data.qpos[:] = self.initial_qpos
            mujoco.mj_forward(self.model, self.data)
            self.initial_joint_positions = self.data.qpos[7:14].copy()

        self.steps = 0
        self.contact_hold_counter = 0
        self._render_counter = 0
        self.current_episode_success = False

        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        terminated = truncated = False

        action = np.asarray(action).flatten()
        expected_dim = len(self.arm_actuator_ids) + (1 if len(self.gripper_actuator_ids) > 0 else 0)

        if len(action) != expected_dim:
            if len(action) > expected_dim:
                action = action[:expected_dim]
            else:
                padded = np.zeros(expected_dim)
                padded[:len(action)] = action
                action = padded

        scaled = np.clip(action, -1.0, 1.0)
        scale_factor = self.action_scale_factor

        for idx, act_id in enumerate(self.arm_actuator_ids):
            if 0 <= act_id < self.model.nu:
                lo, hi = self.model.actuator_ctrlrange[act_id]
                mid = (hi + lo) / 2
                hi_new = mid + (hi - mid) * scale_factor
                lo_new = mid + (lo - mid) * scale_factor
                u = 0.5 * scaled[idx] * (hi_new - lo_new) + (hi_new + lo_new) / 2.0
                self.data.ctrl[act_id] = np.clip(u, lo_new, hi_new)

        gripper_start = len(self.arm_actuator_ids)
        if len(self.gripper_actuator_ids) > 0 and gripper_start < len(scaled):
            gripper_action = scaled[gripper_start]
            for act_id in self.gripper_actuator_ids:
                lo, hi = self.model.actuator_ctrlrange[act_id]
                u = 0.5 * (gripper_action + 1.0) * (hi - lo) + lo
                self.data.ctrl[act_id] = np.clip(u, lo, hi)

        mujoco.mj_step(self.model, self.data)

        if self.target_body_id >= 0:
            has_contact = False
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                g1, g2 = contact.geom1, contact.geom2
                if (g1 == self.target_geom_id or g2 == self.target_geom_id):
                    other_geom = g2 if g1 == self.target_geom_id else g1
                    if other_geom in self.left_finger_geom_ids or other_geom in self.right_finger_geom_ids:
                        has_contact = True
                        break

            if not has_contact:
                obj_adr = self.model.jnt_qposadr[self.model.body_jntadr[self.target_body_id]]
                current_pos = self.data.xpos[self.target_body_id].copy()
                pos_error = np.linalg.norm(current_pos - self.target_pos)

                if pos_error > 0.01:
                    self.data.qpos[obj_adr:obj_adr + 3] = self.target_pos
                    self.data.qpos[obj_adr + 3:obj_adr + 7] = [1, 0, 0, 0]
                    vel_adr = self.model.jnt_dofadr[self.model.body_jntadr[self.target_body_id]]
                    self.data.qvel[vel_adr:vel_adr + 6] = 0.0
                    mujoco.mj_forward(self.model, self.data)

        joint_modified = False
        joint_delta = None
        if self.initial_joint_positions is not None:
            current_joint_pos = self.data.qpos[7:14].copy()
            joint_delta = current_joint_pos - self.initial_joint_positions
            for i in range(len(self.joint_angle_limits)):
                if abs(joint_delta[i]) > self.joint_angle_limits[i]:
                    max_delta = self.joint_angle_limits[i]
                    current_joint_pos[i] = self.initial_joint_positions[i] + (
                        max_delta if joint_delta[i] > 0 else -max_delta)
                    self.data.qpos[7 + i] = current_joint_pos[i]
                    joint_modified = True
            if joint_modified:
                mujoco.mj_forward(self.model, self.data)

        eef_pos = self.data.site_xpos[self.end_effector_id].copy() if self.end_effector_id >= 0 else np.zeros(3)
        object_pos = self.data.xpos[self.target_body_id].copy() if self.target_body_id >= 0 else np.zeros(3)

        if self.left_finger_body_id >= 0 and self.right_finger_body_id >= 0:
            left_tip = self.data.xpos[self.left_finger_body_id].copy()
            right_tip = self.data.xpos[self.right_finger_body_id].copy()
            fingertip_gap = float(np.linalg.norm(left_tip - right_tip))
        elif self.left_tip_site_id >= 0 and self.right_tip_site_id >= 0:
            left_tip = self.data.site_xpos[self.left_tip_site_id].copy()
            right_tip = self.data.site_xpos[self.right_tip_site_id].copy()
            fingertip_gap = float(np.linalg.norm(left_tip - right_tip))
        else:
            fingertip_gap = 0.08
        gripper_closed = fingertip_gap < 0.03

        left_contact = right_contact = False
        if self.target_geom_id >= 0:
            ncon = int(getattr(self.data, "ncon", 0) or 0)
            for i in range(ncon):
                c = self.data.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if (g1 in self.left_finger_geom_ids or g2 in self.left_finger_geom_ids) and (
                        g1 == self.target_geom_id or g2 == self.target_geom_id):
                    left_contact = True
                if (g1 in self.right_finger_geom_ids or g2 in self.right_finger_geom_ids) and (
                        g1 == self.target_geom_id or g2 == self.target_geom_id):
                    right_contact = True
                if left_contact and right_contact:
                    break

        if left_contact and right_contact:
            self.contact_hold_counter += 1
        else:
            self.contact_hold_counter = 0
        contact_hold_frac = min(1.0, self.contact_hold_counter / self.contact_hold_required)

        distance_to_object = float(np.linalg.norm(eef_pos - object_pos))
        height_gain = max(0.0, float(object_pos[2] - self.initial_object_height))

        # =================================================================
        # 🌟 核心修改：奖励重构，防止高分低能
        # =================================================================

        # 1. 基础时间惩罚：逼迫它不要磨蹭
        step_reward = self.reward_weights['step_penalty']

        # 2. 距离奖励：使用 tanh 归一化到 [0, 1]，防止距离无限大导致奖励异常
        # 并且乘以较小的权重，让它只是引导，而不是分数的来源
        dist_shaping = 1.0 - np.tanh(10.0 * distance_to_object)
        distance_reward = self.reward_weights['distance'] * dist_shaping

        # 3. 接触奖励：只在真正接触时给
        contact_reward = 0.0
        if left_contact and right_contact:
            contact_reward = self.reward_weights['contact']
        elif left_contact or right_contact:
            contact_reward = self.reward_weights['contact'] * 0.2

        # 4. 抓取和抬升：条件更严格
        grasp_reward = self.reward_weights['grasp'] if (
                    left_contact and right_contact and contact_hold_frac > 0.5) else 0.0

        lift_bonus = 0.0
        if height_gain > 0.01:
            # 抬得越高分越高，最大限制在 1.0 * weight
            lift_bonus = self.reward_weights['lift'] * min(height_gain / 0.05, 1.0)

        # 5. 动作平滑惩罚
        action_penalty = self.reward_weights['action_penalty'] * np.sum(np.square(action))

        # 6. 边界惩罚
        position_penalty = 0.0
        if not (self.eef_pos_limits['x'][0] <= eef_pos[0] <= self.eef_pos_limits['x'][1] and
                self.eef_pos_limits['y'][0] <= eef_pos[1] <= self.eef_pos_limits['y'][1] and
                self.eef_pos_limits['z'][0] <= eef_pos[2] <= self.eef_pos_limits['z'][1]):
            x_violation = max(0, self.eef_pos_limits['x'][0] - eef_pos[0], eef_pos[0] - self.eef_pos_limits['x'][1])
            y_violation = max(0, self.eef_pos_limits['y'][0] - eef_pos[1], eef_pos[1] - self.eef_pos_limits['y'][1])
            z_violation = max(0, self.eef_pos_limits['z'][0] - eef_pos[2], eef_pos[2] - self.eef_pos_limits['z'][1])
            position_penalty = self.reward_weights['position_penalty'] * (x_violation + y_violation + z_violation)

        # 7. 成功判定
        contact_success = left_contact and right_contact
        grasp_success = contact_success and contact_hold_frac > 0.6
        lift_success = height_gain > 0.025

        if self.success_mode == "distance":
            success = distance_to_object < 0.05  # 判定更严格一点
        elif self.success_mode == "contact":
            success = contact_success
        elif self.success_mode == "grasp":
            success = grasp_success
        elif self.success_mode == "lift":
            success = lift_success
        else:
            success = False

        if success:
            self.current_episode_success = True

        # 🌟 8. 成功奖励：一笔巨款
        success_bonus = self.reward_weights['success'] if success else 0.0

        total_reward = (
                step_reward +
                distance_reward +
                contact_reward +
                grasp_reward +
                lift_bonus +
                success_bonus +
                action_penalty +
                position_penalty
        )

        # 裁剪奖励，防止梯度爆炸
        reward = float(np.clip(total_reward, -10.0, 200.0))

        if success:
            terminated = True

        if self.steps >= self.max_episode_steps:
            truncated = True

        if terminated or truncated:
            self._update_internal_success_rate(self.current_episode_success)

        obs = self._get_obs(left_contact, right_contact, gripper_closed, contact_hold_frac)
        info = {
            "is_success": success,
            "distance": distance_to_object,
            "object_height": float(object_pos[2]),
            "height_gain": height_gain,
            "left_contact": left_contact,
            "right_contact": right_contact,
            "gripper_closed": gripper_closed,
            "reward_total": reward,
        }

        self._render_counter += 1
        if self.render_mode != 'None' and (self._render_counter % max(1, int(self.render_every)) == 0):
            try:
                self._render()
            except Exception:
                pass

        return obs, reward, terminated, truncated, info

    def _update_internal_success_rate(self, is_success):
        """内部使用的成功率更新"""
        self.success_history.append(int(is_success))
        if len(self.success_history) >= 20:
            window_avg = np.mean(list(self.success_history)[-50:])
            alpha = 0.1
            self.recent_success_rate = (1 - alpha) * self.recent_success_rate + alpha * window_avg
        else:
            self.recent_success_rate = np.mean(self.success_history)

    def set_training_progress(self, progress: float):
        """设置训练进度并立即检查是否需要切换"""
        progress = float(np.clip(progress, 0.0, 1.0))
        self.training_progress = progress

        if progress < 0.25:
            target_mode = "distance"
        elif progress < 0.50:
            target_mode = "contact"
        elif progress < 0.75:
            target_mode = "grasp"
        else:
            target_mode = "lift"

        if target_mode != self.success_mode:
            stage_info = {
                "distance": "距离 (0-25%)",
                "contact": "接触 (25-50%)",
                "grasp": "抓取 (50-75%)",
                "lift": "抬升 (75-100%)"
            }
            # 使用 os.getpid() 避免多进程重复打印太多
            if self.steps % 50 == 0:  # 偶尔打印一下
                pass
                # print(f"[Env] 🎯 Phase: {target_mode} ({progress:.1%})")
            self.success_mode = target_mode

    def _render(self):
        if self.render_mode == 'None':
            return
        try:
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                    show_left_ui=False,
                    show_right_ui=False
                )
        except Exception as e:
            pass

    def render(self):
        if self.render_mode == 'human':
            self._render()
            return None
        elif self.render_mode == 'rgb_array' and self.viewer is not None:
            try:
                return self.viewer.read_pixels()
            except Exception:
                pass
        return None

    def close(self):
        if hasattr(self, 'viewer') and self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

    def update_success_rate(self, success_flag):
        self._update_internal_success_rate(success_flag)

    def get_and_reset_episode_success(self):
        success = int(self.current_episode_success)
        self.current_episode_success = False
        return success
