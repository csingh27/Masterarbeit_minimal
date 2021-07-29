# IMPORT PACKAGES

from PIL import Image, ImageTk
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import itertools
import inspect
import os
import torch as T
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)
import gc
import gym
from stable_baselines3 import SAC
import dmc2gym
import stable_baselines_logging
import time
import matplotlib
matplotlib.use('agg') # prevent memory leaks 


if __name__ == '__main__':

    # CONDA ENVIRONMENT : 
    # /home/ubuntu/anaconda3/envs/rlpyt/lib/python3.8/site-packages/dm_control/suite/

    # PATH TO DMC2GYM :
    # print(inspect.getfile(dmc2gym)) # built in path

    # DEFINE PATHS
    location_minimal = "/home/ubuntu/Masterarbeit_minimal/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Minimal/output"
    location_intermediate = "/home/ubuntu/Masterarbeit_minimal/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Intermediate/output"
    location_full = "/home/ubuntu/Masterarbeit_minimal/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Full/output"
    path_to_output = location_minimal + '/output'

    # DEFINE ENVIRONMENTS

    ######### SAMPLE  ######################

    # EXPERIMENT #SAMPLE
    # env_id = 'pendulum'
    # env = gym.make('Pendulum-v0')   

    ######### MINIMAL ######################

    # FIXED INITIALIZATION #
    env_id_m_0_1 = 'cloth_sewts_minimal_0_1'
    env_id_m_0_2 = 'cloth_sewts_minimal_0_2'
    env_id_m_0_3 = 'cloth_sewts_minimal_0_3'
    env_id_m_0_4 = 'cloth_sewts_minimal_0_4'

    reward_max_m_0_1 = 499
    reward_max_m_0_2 = 960
    reward_max_m_0_3 = 960   
    reward_max_m_0_4 = 990

    location_SAC_m_0_1 = location_minimal + "/SAC_m_0_1/"
    location_m_0_1 = location_SAC_m_0_1 + "cloth_sewts_minimal"

    location_SAC_m_0_2 = location_minimal + "/SAC_m_0_2/"
    location_m_0_2 = location_SAC_m_0_2 + "cloth_sewts_minimal"

    location_SAC_m_0_3 = location_minimal + "/SAC_m_0_3/"
    location_m_0_3 = location_SAC_m_0_3 + "cloth_sewts_minimal"

    location_SAC_m_0_4 = location_minimal + "/SAC_m_0_4/"
    location_m_0_4 = location_SAC_m_0_4 + "cloth_sewts_minimal"

    # RANDOM INITIALIZATION #
    env_id_m_1_1 = 'cloth_sewts_minimal_1_1'
    env_id_m_1_2 = 'cloth_sewts_minimal_1_2'
    env_id_m_1_3 = 'cloth_sewts_minimal_1_3'

    reward_max_m_1_1 = 499
    reward_max_m_1_2 = 960
    reward_max_m_1_3 = 960

    location_SAC_m_1_1_a = location_minimal + "/SAC_m_1_1_a/"
    location_m_1_1_a = location_SAC_m_1_1_a + "cloth_sewts_minimal"

    location_SAC_m_1_1_b = location_minimal + "/SAC_m_1_1_b/"
    location_m_1_1_b = location_SAC_m_1_1_b + "cloth_sewts_minimal"

    location_SAC_m_1_2 = location_minimal + "/SAC_m_1_2/"
    location_m_1_2 = location_SAC_m_1_2 + "cloth_sewts_minimal"

    location_SAC_m_1_3 = location_minimal + "/SAC_m_1_3/"
    location_m_1_3 = location_SAC_m_1_3 + "cloth_sewts_minimal"

    location_SAC_m_1_4 = location_minimal + "/SAC_m_1_4/"
    location_m_1_4 = location_SAC_m_1_4 + "cloth_sewts_minimal"

    # PREMANIPULATION #
    env_id_m_2_1 = 'cloth_sewts_minimal_2_1'
    env_id_m_2_2 = 'cloth_sewts_minimal_2_2'

    reward_max_m_2_2 = -2
    reward_max_m_2_2 = -20

    location_SAC_m_2_2 = location_minimal + "/SAC_m_2_2/"
    location_m_2_2 = location_SAC_m_2_2 + "cloth_sewts_minimal"
    location_SAC_m_2_1 = location_minimal + "/SAC_m_2_1/"
    location_m_2_1 = location_SAC_m_2_1 + "cloth_sewts_minimal"


    ######### INTERMEDIATE  ######################

    # FIXED INITIALIZATION #
    env_id_i_0_1 = 'cloth_sewts_intermediate_0_1'
    env_id_i_0_2 = 'cloth_sewts_intermediate_0_2'

    reward_max_i_0_1 = 1490
    reward_max_i_0_2 = 499

    location_SAC_i_0_1 = location_intermediate + "/SAC_i_0_1/"
    location_i_0_1 = location_SAC_i_0_1 + "cloth_sewts_minimal"

    location_SAC_i_0_2 = location_intermediate + "/SAC_i_0_2/"
    location_i_0_2 = location_SAC_i_0_2 + "cloth_sewts_minimal"

    # RANDOM INITIALIZATION #
    env_id_i_1_1 = 'cloth_sewts_intermediate_1_1'
    env_id_i_1_2 = 'cloth_sewts_intermediate_1_2'
    env_id_i_1_3 = 'cloth_sewts_intermediate_1_3'

    reward_max_i_1_1 = 1490
    reward_max_i_1_2 = 499
    reward_max_i_1_3 = 499

    location_SAC_i_1_1 = location_intermediate + "/SAC_i_1_1/"
    location_i_1_1 = location_SAC_i_1_1 + "cloth_sewts_minimal"

    location_SAC_i_1_2 = location_intermediate + "SAC_i_1_2/"
    location_i_1_2 = location_SAC_i_1_2 + "cloth_sewts_minimal"

    location_SAC_i_1_3 = "/home/ubuntu/Masterarbeit_minimal/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Intermediate/output/SAC_i_1_3/"
    location_i_1_3 = location_SAC_i_1_3 + "cloth_sewts_minimal"


    # PREMANIPULATION #
    env_id_i_2_1 = 'cloth_sewts_intermediate_2_1'
    env_id_i_2_2 = 'cloth_sewts_intermediate_2_2'
    env_id_i_2_3 = 'cloth_sewts_intermediate_2_3'
    env_id_i_2_4 = 'cloth_sewts_intermediate_2_4'
    env_id_i_2_5 = 'cloth_sewts_intermediate_2_5'
    env_id_i_2_6 = 'cloth_sewts_intermediate_2_6'
    env_id_i_2_7 = 'cloth_sewts_intermediate_2_7'
    env_id_i_2_8 = 'cloth_sewts_intermediate_2_8'
    env_id_i_2_9 = 'cloth_sewts_intermediate_2_9'
    env_id_i_2_10 = 'cloth_sewts_intermediate_2_10'
    env_id_i_2_12 = 'cloth_corner'

    reward_max_i_2_1 = 499

    location_SAC_i_2_1 = "/home/ubuntu/Masterarbeit_minimal/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Intermediate/output/SAC_i_2_1/"
    location_i_2_1 = location_SAC_i_2_1 + "cloth_sewts_minimal"


    env_id_i_2_2 = 'cloth_sewts_intermediate_2_2'

    reward_max_i_2_2 = 1499

    location_SAC_i_2_2 = "/home/ubuntu/Masterarbeit_minimal/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Intermediate/output/SAC_i_2_2/"
    location_i_2_2 = location_SAC_i_2_2 + "cloth_sewts_minimal"

    reward_max_i_2_3 = 1275

    location_SAC_i_2_3 = "/home/ubuntu/Masterarbeit_minimal/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Intermediate/output/SAC_i_2_3/"
    location_i_2_3 = location_SAC_i_2_3 + "cloth_sewts_minimal"

    reward_max_i_2_4 = 1275

    location_SAC_i_2_4 = "/home/ubuntu/Masterarbeit_minimal/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Intermediate/output/SAC_i_2_4/"
    location_i_2_4 = location_SAC_i_2_4 + "cloth_sewts_minimal"

    reward_max_i_2_5 = 1275

    location_SAC_i_2_5 = "/home/ubuntu/Masterarbeit_minimal/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Intermediate/output/SAC_i_2_5/"
    location_i_2_5 = location_SAC_i_2_5 + "cloth_sewts_minimal"

    reward_max_i_2_6 = 1700

    location_SAC_i_2_6 = "/home/ubuntu/Masterarbeit_minimal/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Intermediate/output/SAC_i_2_6/"
    location_i_2_6 = location_SAC_i_2_6 + "cloth_sewts_minimal"

    reward_max_i_2_7 = 1700

    location_SAC_i_2_7 = "/home/ubuntu/Masterarbeit_minimal/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Intermediate/output/SAC_i_2_7/"
    location_i_2_7 = location_SAC_i_2_7 + "cloth_sewts_minimal"

    reward_max_i_2_8 = 0.95

    location_SAC_i_2_8 = "/home/ubuntu/Masterarbeit_minimal/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Intermediate/output/SAC_i_2_8/"
    location_i_2_8 = location_SAC_i_2_8 + "cloth_sewts_minimal"

    reward_max_i_2_9 = 0.95

    location_SAC_i_2_9 = "/home/ubuntu/Masterarbeit_minimal/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Intermediate/output/SAC_i_2_9/"
    location_i_2_9 = location_SAC_i_2_9 + "cloth_sewts_minimal"

    reward_max_i_2_10 = 0.95

    location_SAC_i_2_10 = "/home/ubuntu/Masterarbeit_minimal/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Intermediate/output/SAC_i_2_10/"
    location_i_2_10 = location_SAC_i_2_10 + "cloth_sewts_minimal"

    reward_max_i_2_12 = 0.9

    location_SAC_i_2_12 = "/home/ubuntu/Masterarbeit_minimal/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Intermediate/output/SAC_i_2_12/"
    location_i_2_12 = location_SAC_i_2_12 + "cloth_sewts_minimal"

    ######### FULL  ######################

    # FIXED INITIALIZATION #
    env_id_f_0_1 = 'cloth_sewts_full_0_1'
    env_id_f_0_2 = 'cloth_sewts_full_0_2'
    
    reward_max_f_0_1 = 499
    reward_max_f_0_2 = 499

    location_SAC_f_0_1 = "/home/ubuntu/Masterarbeit_minimal/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Full/output/SAC_f_0_1/"
    location_f_0_1 = location_SAC_f_0_1 + "cloth_sewts_minimal"

    location_SAC_f_0_2 = "/home/ubuntu/Masterarbeit_minimal/Code/Implementation/Experiment_files/Experiment_cloth_sewts/Full/output/SAC_f_0_2/"
    location_f_0_2 = location_SAC_f_0_2 + "cloth_sewts_minimal"

    # RANDOM INITIALIZATION #
    env_id_f_1_1 = 'cloth_sewts_minimal_1_1'
    env_id_f_1_2 = 'cloth_sewts_minimal_1_2'
    env_id_f_1_3 = 'cloth_sewts_minimal_1_3'

    reward_max_f_1_1 = 499
    reward_max_f_1_2 = 499
    reward_max_f_1_3 = 499

    location_SAC_f_1_1 = "/home/ubuntu/Masterarbeit_minimal/Code/Implementation/Experiment_files/Experiment cloth_sewts/Full/output/SAC_f_1_1/"
    location_f_1_1 = location_SAC_f_1_1 + "cloth_sewts_minimal"

    location_SAC_f_1_2 = "/home/ubuntu/Masterarbeit_minimal/Code/Implementation/Experiment_files/Experiment cloth_sewts/Full/output/SAC_f_1_2/"
    location_f_1_2 = location_SAC_f_1_2 + "cloth_sewts_minimal"

    location_SAC_f_1_3 = "/home/ubuntu/Masterarbeit_minimal/Code/Implementation/Experiment_files/Experiment cloth_sewts/Full/output/SAC_f_1_3/"
    location_f_1_3 = location_SAC_f_1_3 + "cloth_sewts_minimal"

    # PREMANIPULATION #
    env_id_f_2_1 = 'cloth_sewts_minimal_2_1'
    env_id_f_2_1 = 'cloth_sewts_minimal_2_1'
    env_id_f_2_1 = 'cloth_sewts_minimal_2_1'

    reward_max_f_2_1 = 499

    location_SAC_f_2_1 = "/home/ubuntu/Masterarbeit_minimal/Code/Implementation/Experiment_files/Experiment cloth_sewts/Full/output/SAC_f_2_1/"
    location_f_2_1 = location_SAC_f_2_1 + "cloth_sewts_minimal"


    ######################################

    # DEFINE  VARIABLES
    n_games = 50
    reward_collect = []
    step_collect = []
    accuracy_collect = [] 
    observation_ideal = [0., 0., 0, -0.03, 0., 0.017, -0.06, 0., 0]
    observation_ideal = [0., 0., 0.017, -0.03, 0., 0.017, -0.06, 0., 0.017]
    ######################################

    # SELECT EXPERIMENTS 

    # CONFIG m_0_1
    # env_id = [env_id_m_0_1]
    # reward_set = [reward_max_m_0_1]
    # location_set = [location_m_0_1] 
    # location_set_SAC = [location_SAC_m_0_1]

    # CONFIG m_0_2
    # env_id = [env_id_m_0_2]
    # reward_set = [reward_max_m_0_2]
    # location_set = [location_m_0_2] 
    # location_set_SAC = [location_SAC_m_0_2]

    # CONFIG m_0_3
    # env_id = [env_id_m_0_3]
    # reward_set = [reward_max_m_0_3]
    # location_set = [location_m_0_3] 
    # location_set_SAC = [location_SAC_m_0_3]

    # CONFIG m_0_4
    # env_id = [env_id_m_0_4]
    # reward_set = [reward_max_m_0_4]
    # location_set = [location_m_0_4] 
    # location_set_SAC = [location_SAC_m_0_4]
    
    # CONFIG m_1_1
    # env_id = [env_id_m_1_1_a]
    # reward_set = [reward_max_m_1_1]
    # location_set = [location_m_1_1_a] 
    # location_set_SAC = [location_SAC_m_1_1_a]

    # CONFIG m_1_2
    # env_id = [env_id_m_1_2]
    # reward_set = [reward_max_m_1_2]
    # location_set = [location_m_1_2] 
    # location_set_SAC = [location_SAC_m_1_2]

    # CONFIG m_1_4
    # predict
    # env_id = [env_id_m_1_1_a, env_id_m_1_1_b]
    # train
    # env_id = [env_id_m_1_2]
    # reward_set = [reward_max_m_1_2]
    # location_set = [location_m_1_4] 
    # location_set_SAC = [location_SAC_m_1_4]


    # CONFIG m_2_1
    # env_id = [env_id_m_2_1]
    # reward_set = [reward_max_m_2_1]
    # location_set = [location_m_2_1] 
    # location_set_SAC = [location_SAC_m_2_1]

    # CONFIG m_2_2
    # env_id = [env_id_m_2_2]
    # reward_set = [reward_max_m_2_2]
    # location_set = [location_m_2_2] 
    # location_set_SAC = [location_SAC_m_2_2]

    # CONFIG i_0_1
    # env_id = [env_id_i_0_1]
    # reward_set = [reward_max_i_0_1]
    # location_set = [location_i_0_1] 
    # location_set_SAC = [location_SAC_i_0_1]

    # CONFIG i_1_1
    # env_id = [env_id_i_1_1]
    # reward_set = [reward_max_i_1_1]
    # location_set = [location_i_1_1] 
    # location_set_SAC = [location_SAC_i_1_1]

    # CONFIG i_2_2
    # env_id = [env_id_i_2_2]
    # reward_set = [reward_max_i_2_2]
    # location_set = [location_i_2_2] 
    # location_set_SAC = [location_SAC_i_2_2]

    # CONFIG i_2_3
    # env_id = [env_id_i_2_3]
    # reward_set = [reward_max_i_2_3]
    # location_set = [location_i_2_3] 
    # location_set_SAC = [location_SAC_i_2_3]

    # CONFIG i_2_4
    # env_id = [env_id_i_2_4]
    # reward_set = [reward_max_i_2_4]
    # location_set = [location_i_2_4] 
    # location_set_SAC = [location_SAC_i_2_4]

    # CONFIG i_2_5
    # env_id = [env_id_i_2_5]
    # reward_set = [reward_max_i_2_5]
    # location_set = [location_i_2_5] 
    # location_set_SAC = [location_SAC_i_2_5]

    # CONFIG i_2_6
    # env_id = [env_id_i_2_6]
    # reward_set = [reward_max_i_2_6]
    # location_set = [location_i_2_6] 
    # location_set_SAC = [location_SAC_i_2_6]

    # CONFIG i_2_7
    # env_id = [env_id_i_2_7]
    # reward_set = [reward_max_i_2_7]
    # location_set = [location_i_2_7] 
    # location_set_SAC = [location_SAC_i_2_7]

    # CONFIG i_2_8
    # env_id = [env_id_i_2_8]
    # reward_set = [reward_max_i_2_8]
    # location_set = [location_i_2_8] 
    # location_set_SAC = [location_SAC_i_2_8]

    # CONFIG i_2_9
    # env_id = [env_id_i_2_9]
    # reward_set = [reward_max_i_2_9]
    # location_set = [location_i_2_9] 
    # location_set_SAC = [location_SAC_i_2_9]

    # CONFIG i_2_10
    env_id = [env_id_i_2_10]
    reward_set = [reward_max_i_2_10]
    location_set = [location_i_2_10] 
    location_set_SAC = [location_SAC_i_2_10]

    # CONFIG i_2_12
    # env_id = [env_id_i_2_12]
    # reward_set = [reward_max_i_2_12]
    # location_set = [location_i_2_12] 
    # location_set_SAC = [location_SAC_i_2_12]

    for i in range(len(location_set_SAC)): 

        if not os.path.exists(location_set_SAC[i]):
            os.mkdir(location_set_SAC[i])

        # LOAD EXPERIMENT
        os.chdir(location_set_SAC[i])
        env = dmc2gym.make(domain_name=env_id[i], task_name='easy', seed=1) #dmc2gym package
        # DEFINE MODEL

        tensorboard_log = location_set_SAC[i] + "/Log"
        if not os.path.exists(tensorboard_log):
            os.mkdir(tensorboard_log)

        model = SAC("MlpPolicy", env, learning_starts=1024, 
                tensorboard_log=tensorboard_log,
                batch_size=256, verbose=1)

        print("##################################################")
        print("ENVIRONMENT",i + 1," /",location_set,"\n")

        for j in range(n_games):
            
            # INITIALIZE VARIABLES

            done = False
            step = 0
            reward = 0
            dist = 0
            print("GAME no.",j + 1,"/n/n/n")

            # LOAD MODEL IF ALREADY EXISTING

            if os.path.isfile(location_set[i] + '.zip'):
               model = SAC.load(location_set[i])

            # SET ENVIRONMENT
            model.set_env(env)
            os.chdir(location_set_SAC[i])

            # TRAIN MODEL
            model.learn(total_timesteps=600000, reset_num_timesteps = True, callback = stable_baselines_logging.ImageRecorderCallback(), log_interval=1) # log_interval = no. of episodes

            # EXPERIMENT #
            model.save(location_set[i])

            observation = env.reset()  

            for z in range(200):
                # env.step(np.array([0.,0.]))
                # FOR EXPERIMENT cloth_sewts_minimal_2_1
                # env.step(np.array([0.,0.,0.,0.,0.,0.]))
                # FOR EXPERIMENT cloth_sewts_minimal_2_1
                # env.step(np.array([0.,0.]))
                # FOR EXPERIMENT cloth_sewts_minimal_2_1
                # env.step(np.array([0.,0.]))
                env.step(np.array([0.,0.,0.,0.,0.,0.,0.]))

            while done is False :

                
                # MODEL PREDICTION
                
                action, observation_ = model.predict(observation, deterministic=False)
                
                # UPDATE STEPS
                
                step = step + 1
                print("STEP",step,"/n")
                # action = np.array([0.,0.,0.,0.,0.,0.,0.])
                # TAKE STEP
                # action = np.array([0.,0.,0.,0.,0.,0.])
                
                # action = np.array([0.,0.,0.,0.,0.,0.,0.])
                print(action)
                obs, reward, done, info = env.step(action)
                 
                # PRINTING RESULTS

                print("Evaluating : step ",step)
                print("observation")
                print(obs)
                print("action")
                print(action)
                print("reward")
                print(reward)

                # config m_2_1            
                # dist1 = abs(obs[0] - observation_ideal[2])
                # dist2 = abs(obs[1]- observation_ideal[5])
                # config m_0_1, config m_1_1
                # dist1 = np.sqrt((obs[0]- observation_ideal[0])**2 + (obs[1]- observation_ideal[1])**2)
                # config m_0_1, config m_1_1
                # dist2 = dist1
                # config m_0_2, config m_0_3, config m_1_2
                # dist2 = np.sqrt((obs[3]- observation_ideal[3])**2 + (obs[4]- observation_ideal[4])**2)
                # dist = (dist1 + dist2) / 2
                # accuracy = (1 - dist) * 100
                # REDEFINED ACCURACY for i_2_3                
                # accuracy = 

                # UPDATE OBSERVATION

                observation = obs

                # TERMINAL CONDITION
                # SAVE FIGURES
                
                slope_G00_G01 = "G00_G01 Slope - " + str((observation[4] - observation[1])/(observation[3]-observation[0]))
                slope_G01_G02 = "G01_G02 Slope - " + str((observation[7] - observation[4])/(observation[6]-observation[3]))
                slope_G00_G02 = "G00_G02 Slope - " + str((observation[7] - observation[1])/(observation[6]-observation[0]))
                string4 = "obs : G00xy - " + str(observation[:2]) +  " | G01z - " + str(observation[3:5]) 
                string5 = " G02xy - " + str(observation[6:8])    

                image = env.render(mode="rgb_array", height=256, width=256) 
                plt.imshow(image)
                string1_0 = "obs : G00z - " + str(observation[2]) +  " | G01z - " + str(observation[5]) +  "|  G02z - " + str(observation[8])  
                string1_1 = "obs : G10z - " + str(observation[20]) +  " | G11z - " + str(observation[35]) +  "|  G12z - " + str(observation[38])  

                string2 = "step :" + str(step)
                string3 = "reward :" + str(reward)
                # string4 = "accuracy :" + str(accuracy)
                plt.text(-90, -20, string1_0, bbox=dict(fill=False, edgecolor='black', linewidth=1))
                plt.text(-90, 0, string1_1, bbox=dict(fill=False, edgecolor='black', linewidth=1))
                plt.text(-90, 20, string2, bbox=dict(fill=False, edgecolor='black', linewidth=1))
                plt.text(-90, 40, string3, bbox=dict(fill=False, edgecolor='black', linewidth=1))
                plt.text(-90, 60, slope_G00_G01, bbox=dict(fill=False, edgecolor='black', linewidth=1))
                plt.text(-90, 80, slope_G01_G02, bbox=dict(fill=False, edgecolor='black', linewidth=1))
                plt.text(-90, 100, slope_G00_G02, bbox=dict(fill=False, edgecolor='black', linewidth=1))
                plt.text(-90, 120, string4, bbox=dict(fill=False, edgecolor='black', linewidth=1))
                plt.text(-90, 140, string5, bbox=dict(fill=False, edgecolor='black', linewidth=1))


                # plt.text(-90, 40, string4, bbox=dict(fill=False, edgecolor='black', linewidth=1))

                # plt.show()
                plt.axis('off')
                plt.show(block=False)
                game_folder = "Game_" + str(j+1)
                if not os.path.exists(game_folder):
                    os.mkdir(game_folder)
                game_location = location_set_SAC[i] + game_folder
                name = game_location + "/" + str(time.time()) + ".png"
                plt.savefig(name)
                print("Done /n")
                plt.close()

                print(i)
                
                if step == 800 or reward > reward_set[i]: # EXPERIMENT_1 # EXPERIMENT_0
                    reward_collect.append(reward)
                    step_collect.append(step)
                    # accuracy_collect.append(accuracy)
                    done = True
                    print("\n####################################DONE################################\n")
                
            j = j + 1

        i = i + 1
        average_reward = sum(reward_collect)/len(reward_collect)
        average_step = sum(step_collect)/len(step_collect)
        # average_accuracy = sum(accuracy_collect)/len(accuracy_collect)
        with open("test.txt", "a") as myfile:
            myfile.write("\nAverage Reward " + str(average_reward) )
            myfile.write("\nAverage Step " + str(average_step) )
            # myfile.write("\nAverage Accuracy " + str(average_accuracy) )
         
