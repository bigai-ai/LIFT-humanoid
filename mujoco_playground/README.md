## License and Modifications

This repository is a fork of MuJoCo Playground, which is licensed under the Apache License 2.0.
This fork includes modifications by FastTD3 to support the custom tasks `T1LowDimJoystickFlatTerrain` and `T1LowDimJoystickRoughTerrain`.

All original copyright remains with DeepMind Technologies Limited.
See individual file headers for details on changes and license terms.

## Installation

conda create -n mujoco_playground python=3.10 -c conda-forge -y
conda activate mujoco_playground
pip install -r requirements_playground.txt
pip install -e .

