This is code for my research on "Reproduction of brush painting technique with wall contact by multicopter".
The code named success.py is the code that makes the simulation work.
My research is as follows.

Research Overview
Recently, UAVs have been used for inspection and repair of construction sites. UAVs used for repair work are equipped with a spraying function, but the conditions of use are such that unevenness and scattering are tolerated. Therefore, we propose a UAV-based coating method using a brush, which is considered the most common coating method, and aim to automate the coating process that can handle detailed areas and complex shapes. In this study, we focused on the brush pressure during application, and analyzed UAV brush painting by manual operation. Based on the issues obtained from the analysis, we proposed a control model for the brush painting operation and confirmed its effectiveness by conducting wall contact experiments in a simulation environment.

We created a model of a drone with a cylindrical link and conducted a wall contact experiment using the created model to confirm that the drone can maintain a posture that satisfies the target pushing force by PID control and can make smooth contact with the wall.


# Note

These programs were created with reference to the following repository.

https://github.com/utiasDSL/gym-pybullet-drones

I created this repository for my own learning.  
This is just a program to simulate the physical behavior of drones with `pybullet`.


Requires `Python>=3.7`.


| Set |  Modules and scripts in this repository | Required modules  |
| --- | ----------------- | ----------------- |
| A | `./b0*_*.py` | `numpy`, `pybullet` |
| Optional | `./util/data_logger.py` | `matplotlib`, `cycler` |

