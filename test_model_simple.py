#!/usr/bin/env python3
"""
简单的模型测试 - 使用交互式viewer
对 MuJoCo 仿真模型（XML 文件）进行基础的“体检”和可视化调试。
"""
import mujoco
import mujoco.viewer
import numpy as np

def test_model():
    print("=" * 60)
    print("简单模型测试 - 交互式Viewer")
    print("=" * 60)
    
    model_path = 'envs/assets/franka/panda.xml'
    print(f"\n加载模型: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    print(f"✅ 模型加载成功")
    print(f"   几何体数量: {model.ngeom}")
    
    # 检查group分布
    groups = {}
    for i in range(model.ngeom):
        group = model.geom_group[i]
        groups[group] = groups.get(group, 0) + 1
    
    print(f"\n几何体Group分布:")
    for g, count in sorted(groups.items()):
        print(f"   Group {g}: {count} 个几何体")
    
    print(f"\n使用交互式viewer...")
    print("提示:")
    print("  - 按 'G' 键可以切换显示不同的group")
    print("  - 按 'V' 键可以切换显示模式")
    print("  - 按 'ESC' 或关闭窗口退出")
    print("  - 默认可能显示group=3（collision），需要切换到group=2（visual）")
    print("\n正在启动viewer...")
    
    # 使用交互式viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("✅ Viewer已启动！")
        print("\n在viewer窗口中:")
        print("  1. 按 'G' 键多次，直到看到完整的mesh模型")
        print("  2. 观察机器人结构")
        print("  3. 按 ESC 或关闭窗口退出")
        
        # 运行仿真
        step = 0
        try:
            while viewer.is_running():
                # 随机动作（只控制关节）
                if model.nu > 0:
                    # 为前7个关节生成随机控制信号
                    for i in range(min(7, model.nu)):
                        data.ctrl[i] = np.random.uniform(-0.5, 0.5)
                
                mujoco.mj_step(model, data)
                viewer.sync()
                
                step += 1
                if step % 100 == 0:
                    print(f"  运行中... Step {step}")
                    
        except KeyboardInterrupt:
            print("\n用户中断")
    
    print("\n✅ 测试完成！")

if __name__ == "__main__":
    test_model()

