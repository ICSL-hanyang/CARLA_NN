# CARLA_NN

Deep Neural Network-based Autonomous Driving Resarch using CARLA.

# Environment Setup Procedure

0. Pre-requisites
    - Update pip3 and pip to the latest version
        - pip3 install --upgrade pip
        - pip install --upgrade pip
    - Install pygame using pip
        - pip3 install pygame
        - pip install pygame
    - Install libomp-dev
        - sudo apt-get update
        - sudo apt-get install libomp-dev
        
1. Install CARLA by following 'Package Installation' in CARLA installation instructions (https://carla.readthedocs.io/en/0.9.12/start_quickstart/#carla-installation)


2. Run CARLA Engine Server (./CarlaUE4.sh)

3. Run client-based custom code while CARLA engine is running