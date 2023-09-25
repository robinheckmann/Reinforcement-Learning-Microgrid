import gym
import environmentSofia
import numpy as np
import pandas as pd
import pickle as pkl
from stable_baselines3 import DDPG, PPO, DQN, A2C, TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.logger import configure
from vars import *
from stable_baselines3.common.callbacks import EvalCallback

import argparse
import os

def run(ckpt, model_type, dynamic, soft, transfer, noisy, eval):

    eval = False
    print(transfer)
    transfer_path = ''
    if transfer:
        transfer_path = 'transfer/'

    save_dir = 'data/output/' + model_type + '/simulation1/' + transfer_path
    
    new_logger = configure(save_dir, ["stdout", "csv"])

    env = gym.make('environmentSofia/GridWorld-v0')
    

   

    if model_type == "DDPG":
        print("=========== DDPG ========== ")

        n_actions = 4  # replace with your actual action space dimension
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(3), sigma=0.1 * np.ones(3))

        LEARNING_RATE = 1e-6  # noch weiter reduzierte Lernrate 1e-7
        BATCH_SIZE = 16  # weiter erhöhte Batchgröße 64
        TAU = 0.001  # weiter erhöhter Tau-Wert 0.001

        model = DDPG("MlpPolicy", env)

        
    # policy_kwargs=dict(net_arch=[FC_1_DIMS, FC_2_DIMS, FC_3_DIMS]), 
    if model_type == "PPO":
        print("=========== PPO ========== ")
        #PPO.load(f"{save_dir}/PPO_tutorial")
        if transfer:
            model = PPO.load("data/output/PPO/simulation1/model")
            print("model loaded")
            print(model)
        else:
            model = PPO("MlpPolicy", env)
    
    if model_type == "DQN":
        print("=========== DQN ========== ")
        #PPO.load(f"{save_dir}/PPO_tutorial")
        #model = DQN("MlpPolicy", env)

    if model_type == "A2C":
        print("=========== A2C ========== ")
        #PPO.load(f"{save_dir}/PPO_tutorial")
        model = A2C("MlpPolicy", env)

    if model_type == "SAC":
        print("=========== A2C ========== ")
        #PPO.load(f"{save_dir}/PPO_tutorial")
        model = SAC("MlpPolicy", env)
    
    if model_type == "TD3":
        print("=========== TD3 ========== ")
        #PPO.load(f"{save_dir}/PPO_tutorial")
        LEARNING_RATE = 1e-7  # noch weiter reduzierte Lernrate 1e-7
        BATCH_SIZE = 16  # weiter erhöhte Batchgröße 64
        TAU = 0.001  # weiter erhöhter Tau-Wert 0.001
        model = TD3("MlpPolicy", env, device=device)
   
     
    #model = PPO("MlpPolicy", env)
    
    eval_callback = EvalCallback(env, best_model_save_path=save_dir, log_path=save_dir, eval_freq=5000)


    env = gym.make('environmentSofia/GridWorld-v0')
    model.set_env(env)
    vec_env = model.get_env()
    obs = vec_env.reset()

    model.set_logger(new_logger)
    model.learn(total_timesteps=1000000, log_interval=1)

    vec_env = model.get_env()
    obs = vec_env.reset()


    print("======== TEST PHASE ========")
    
    hydrogen_produced = [env.hydrogen]
    storage_state = [env.storage]
    prices = [env.price]
    power_from_grid = [env.power_from_grid]
    sun_power = [env.sun_power]
    wind_power = [env.wind_generation]
    moles = [env.moles]
    natural_gas = [env.natural_gas]
    dates = [env.date]
    pv_generation = [env.pv_generation]
    natural_gas_prices = [env.natural_gas_price]
    gas_consumptions = [env.gas_consumption]
    ammonia_produced = [env.ammonia]
    cost = [env.cost]

    profits = [env.profit]
    actions = [[0,0,0]]
    action1 = [0]
    action2 = [0]
    action3 = [0]
    rewards = [0]
    
    for i in range(24*14):
        print(i)
        action, _states = model.predict(obs, deterministic=True) 
        obs, reward, done, info = vec_env.step(action)
     
        action = action[0]
        action1.append(action[0])
        action2.append(action[1])
        action3.append(action[2])  
        actions.append(action)
        rewards.append(reward)
        moles.append(env.moles)
        profits.append(env.profit)
        pv_generation.append(env.pv_generation)
        natural_gas_prices.append(env.natural_gas_price)
        gas_consumptions.append(env.gas_consumption)
        hydrogen_produced.append(env.hydrogen)
        ammonia_produced.append(env.ammonia)
        natural_gas.append(env.natural_gas)
        storage_state.append(env.storage)
        sun_power.append(env.sun_power)
        wind_power.append(env.wind_generation)
        dates.append(env.date)
        power_from_grid.append(env.power_from_grid)
        prices.append(env.price)
        cost.append(env.cost)
       
        
   

    eval_data = pd.DataFrame()
    eval_data['Actions'] = actions
    eval_data['Rewards'] = rewards
    eval_data['Action1'] = action1
    eval_data['Action2'] = action2
    eval_data['Action3'] = action3
    eval_data['PV Generation'] = pv_generation
    eval_data['Datetime'] = dates
    eval_data['Gas Consumption'] = gas_consumptions
    eval_data['Cost'] = cost
    eval_data['Prices'] = prices
    eval_data['Profits'] = profits
    eval_data['Ammonia'] = ammonia_produced
    eval_data['Prices Natural gas'] = natural_gas_prices
    eval_data['Moles'] = moles         
    eval_data['Storage'] = storage_state
    eval_data['Power'] = power_from_grid
    eval_data['Sun Power'] = sun_power
    eval_data['Hydrogen'] = hydrogen_produced
    eval_data['Wind Power'] = wind_power
    eval_data['Natural Gas'] = natural_gas

    with open(os.getcwd() + '/data/output/' + model_type + '/simulation1/' + model_type + '_eval.pkl', 'wb') as f:
        pkl.dump(eval_data, f)        


    
     
    

    action1 = []
    action2 = []
    action3 = []
    sun_powers = []
    prices = []
    battery_levels = []
    moless = []
    gas_prices = []

     
    for sun_power in np.arange(0,110000000,10000000):
        print("Sun Power", sun_power)
        for price in np.arange(0,110,1):
            print("Price", price)
            for battery_level in np.arange(0,5500000,500000):
                for moles in np.arange(0,51000,1000):
                    for gas_price in np.arange(0,11,1):
                    
                        state = [sun_power, price, battery_level, moles, gas_price]
                
                        
                        action, _states = model.predict(state, deterministic=True) 
                        action1.append(action[0])
                        action2.append(action[1])
                        action3.append(action[2])
                        sun_powers.append(sun_power)      
                        prices.append(price)
                        battery_levels.append(battery_level)
                        
                        moless.append(moles)
                        gas_prices.append(gas_price)

    eval_data = pd.DataFrame()
    eval_data['Action1'] = action1
    eval_data['Action2'] = action2
    eval_data['Action3'] = action3
    eval_data['Grid Prices'] = prices
    eval_data['Battery Level'] = battery_levels
    eval_data['Moles'] = moless
    eval_data['Gas Prices'] = gas_prices
    eval_data['Sun Power'] = sun_power  

    with open(os.getcwd() + '/data/output/' + model_type + '/simulation1/' + model_type + '_policy_eval.pkl', 'wb') as f:
        pkl.dump(eval_data, f)

  


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=False)
    parser.add_argument("--model_type", default='PPO')
    parser.add_argument("--dynamic", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--soft", default=True,type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--transfer", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--noisy", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--eval", default=False, type=lambda x: (str(x).lower() == 'true'))
   
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(**vars(args))


    