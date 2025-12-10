import mujoco
import mujoco.viewer

xml_path = "panda_ppo/franka_emika_panda/scene.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("\n=== SITES ===")
for i in range(model.nsite):
    print(i, model.site(i).name)

print("\n=== BODIES ===")
for i in range(model.nbody):
    print(i, model.body(i).name)

print("\n=== JOINTS ===")
for i in range(model.njnt):
    print(i, model.joint(i).name)
    
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
