# CARLA_NN

Deep Neural Network-based Autonomous Driving Resarch using CARLA.

# Next Lane Image Prediction using CNN-based U-Net

![image](https://user-images.githubusercontent.com/10843389/139206038-cbad48c1-0eb1-4081-abf2-a360b4669ef3.png)

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
        
1. Install CARLA by following 'Package Installation' in CARLA installation instructions
    - https://carla.readthedocs.io/en/0.9.12/start_quickstart/#carla-installation


2. Run CARLA Engine Server 
    - ./CarlaUE4.sh

3. Clone CARLA_NN repository
    - git clone https://github.com/ICSL-hanyang/CARLA_NN.git

4. Run client-based custom code while CARLA engine is running
