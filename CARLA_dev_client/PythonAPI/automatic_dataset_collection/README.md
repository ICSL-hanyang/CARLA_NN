## Introduction

This source code is a modified implementation of CARLA automatic control example.

It is modified to collect front camera images along with various driving status data, such as throttle and steering angle, withc matching simulation timestamps.

Modification is made by Byung Chan Choi on 2021. 


[Source code](https://github.com/ICSL-hanyang/CARLA_NN/tree/master/CARLA_dev_client/PythonAPI/automatic_dataset_collection) is located at the repository, [ICSL-hanyang/CARLA_NN](https://github.com/ICSL-hanyang/CARLA_NN), which is forked from [original CARLA repository](https://github.com/carla-simulator/carla).

Original CARLA automatic control example follows MIT license.


## Requirements
Ubuntu 18.04

[CARLA Simulator](https://carla.org/) (0.9.12)

[OpenCV-Python](https://pytorch.org/) (Higher than 4.5.3)

[pygame](https://www.pyqtgraph.org/)

[libomp-dev](https://matplotlib.org/)

## Simulation Environment Setup
#### 1. Installing pre-requsites

Update pip3 and pip to the latest version

```bash
pip3 install –upgrade pip
pip install –upgrade pip
```

Install OpenCV-Python using pip (OpenCV-Python is for image I/O control)

```bash
pip install opencv-python
pip install opencv-contrib-python
```

Install pygame using pip (pygame is for rendering CARLA in a separate window)

```bash
pip3 install pygame
pip install pygame
```

Install libomp-dev (libomp is OpenMP library for parallel programming in CARLA environment)

```bash
sudo apt-get update
sudo apt-get install libomp-dev
```

#### 2. Setting up CARLA simulation with Package Installation

Follow the instructions in [official CARLA package installation guide](https://carla.readthedocs.io/en/0.9.12/start_quickstart/).

- Move to CARLA release download list (https://github.com/carla-simulator/carla/blob/master/Docs/download.md)

- For CARLA 0.9.12, move to https://github.com/carla-simulator/carla/releases/tag/0.9.12/

- Download pre-compiled CARLA 0.9.12 release (CARLA_0.9.12.tar.gz)

- Download additional assets (AdditionalMaps_0.9.12.tar.gz)

- Unzip CARLA simulator (CARLA_0.9.12.tar.gz)

- Move additional asset package (AdditionalMaps_0.9.12.tar.gz) to Import folder of CARLA simulator folder and run the following script to extract the contents

```bash
cd PATH_TO_CARLA_0.9.12
./ImportAssets.sh
```

- Move to CARLA simulator directory

- Run CARALA engine server by executing CarlaUE4.sh shell script

```bash
./CarlaUE4.sh
```

#### 3. Cloning ICSL_NN/CARLA repository

Clone ICSL-hanyang/CARLA_NN repository in a separate directory

```bash
git clone https://github.com/ICSL-hanyang/CARLA_NN.git
```


## Running the Code

Move to CARLA 0.9.12 root directory and run CARLA engine server

```bash
cd PATH_TO_CARLA_0.9.12
./CarlaUE4.sh
```

Move to PythonAPI directory of ICSL-hanyang/CARLA_NN repository

```bash
cd PATH_TO_ICSL-hanyang/CARLA_NN/CARLA_dev_client/PythonAPI
```

Move to automatic_dataset_collection directory in PythonAPI

```bash
cd ~/automatic_dataset_collection
```

Specify following parameters in execution shell script for automatic dataset collection (run.sh)

- seed : Seed number for random generator / This will determine what type of vehicle will be used for simulation
- collection_mode : Initial dataset collection mode (Possible options : 'training', 'validation', 'test')

```bash
./run.sh
```

## Configuring CARLA Simulation

#### Disable graphic rendering of CARLA engine server

Run configuration code with rendering option to disable graphic rendering of engine server in order to speed up the simulation

```bash
cd PATH_TO_ICSL-hanyang/CARLA_NN/CARLA_dev_client/PythonAPI
cd ~/util
python3 config.py --no-rendering
```

#### Change the map of CARLA environment

Run configuration code with map option to change the map of the environment

```bash
cd PATH_TO_ICSL-hanyang/CARLA_NN/CARLA_dev_client/PythonAPI
cd ~/util
python3 config.py --map XX
```