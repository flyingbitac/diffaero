from textwrap import dedent
import os

import numpy as np

path = os.path.join(os.path.dirname(__file__), "../launch/")

def create_cluttered_world(cylinders):
    head = dedent("""
        <?xml version="1.0" ?>
        <sdf version="1.5">
            <world name="default">
            <!-- A global light source -->
            <include>
                <uri>model://sun</uri>
            </include>
            <!-- A ground plane -->
            <include>
                <uri>model://ground_plane</uri>
            </include>
            <include>
                <uri>model://asphalt_plane</uri>
            </include>
    """)
    tail = dedent("""
            <physics name='default_physics' default='0' type='ode'>
              <gravity>0 0 -9.8066</gravity>
              <ode>
                <solver>
                  <type>quick</type>
                  <iters>10</iters>
                  <sor>1.3</sor>
                  <use_dynamic_moi_rescaling>0</use_dynamic_moi_rescaling>
                </solver>
                <constraints>
                  <cfm>0</cfm>
                  <erp>0.2</erp>
                  <contact_max_correcting_vel>100</contact_max_correcting_vel>
                  <contact_surface_layer>0.001</contact_surface_layer>
                </constraints>
              </ode>
              <max_step_size>0.004</max_step_size>
              <real_time_factor>1</real_time_factor>
              <real_time_update_rate>250</real_time_update_rate>
              <magnetic_field>6.0e-6 2.3e-5 -4.2e-5</magnetic_field>
            </physics>
          </world>
        </sdf>
    """)
    world = "".join([head] + cylinders + [tail])
    with open(os.path.join(path, "testworld.world"), "w") as f:
        f.write(world)
    print("World file generated at", os.path.abspath(os.path.join(path, "testworld.world")))

cylinder_idx = 0
def create_cylinder(x, y, r, h):
    global cylinder_idx
    cylinder = dedent(f"""
        <model name='unit_cylinder_{cylinder_idx}'>
          <pose>{x:.5f} {y:.5f} {h/2+0.02:.5f} 0 -0 0</pose>
          <link name='link'>
            <inertial>
              <mass>100</mass>
              <inertia>
                <ixx>20</ixx>
                <ixy>0</ixy>
                <ixz>0</ixz>
                <iyy>20</iyy>
                <iyz>0</iyz>
                <izz>20</izz>
              </inertia>
              <pose>0 0 0 0 -0 0</pose>
            </inertial>
            <collision name='collision'>
              <geometry>
                <cylinder>
                  <radius>{r:.3f}</radius>
                  <length>{h:.3f}</length>
                </cylinder>
              </geometry>
              <max_contacts>10</max_contacts>
              <surface>
                <contact>
                  <ode/>
                </contact>
                <bounce/>
                <friction>
                  <torsional>
                    <ode/>
                  </torsional>
                  <ode/>
                </friction>
              </surface>
            </collision>
            <visual name='visual'>
              <geometry>
                <cylinder>
                  <radius>{r:.3f}</radius>
                  <length>{h:.3f}</length>
                </cylinder>
              </geometry>
              <material>
                <script>
                  <name>Gazebo/Grey</name>
                  <uri>file://media/materials/scripts/gazebo.material</uri>
                </script>
              </material>
            </visual>
            <self_collide>0</self_collide>
            <enable_wind>0</enable_wind>
            <kinematic>0</kinematic>
          </link>
        </model>
    """).replace("\n", "    \n    ")
    cylinder_idx += 1
    return cylinder, (x, y, r)

B = 2
N = 10
L = 5
R_min = 0.4
R_max = 0.6
H = 10

if __name__ == "__main__":
    print(f"suggested start position: (0, 0)")
    print(f"suggested target position: ({N * L + 2 * B}, {N * L + 2 * B})")
    x_base = np.arange(N) * L + B
    y_base = np.arange(N) * L + B
    x_rand = np.random.rand(N, N) * L
    y_rand = np.random.rand(N, N) * L
    x_base, y_base = np.meshgrid(x_base, y_base, indexing='xy')
    xs, ys = x_base + x_rand, y_base + y_rand
    rs = np.random.rand(N, N) * (R_max - R_min) + R_min
    
    cylinders = []
    with open(os.path.join(path, "cylinder_positions.txt"), "w") as f:
        for i in range(N):
            for j in range(N):
                cylinder, (x, y, r) = create_cylinder(xs[i, j], ys[i, j], rs[i, j], H)
                cylinders.append(cylinder)
                f.write(f"{x:.5f},{y:.5f},{r:.5f}\n")
    create_cluttered_world(cylinders)