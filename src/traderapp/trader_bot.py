# import requirements
import pandas as pd
import numpy as np
import time
import os
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import math
import random
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle

def trader_bot(mode, epochs=50, split=0.8, initial_cash=1000):
    """
    input: mode (test or train), number of epochs and train/test split
    runs 'epochs' number of episodes in the rl trading 
    outputs: the agent (including model and losses) and portfolio returns
    """
    import numpy as np


    from rl_classes import Agent, Environment



    models_dir = 'trader_bot_model'
    num_episodes = epochs
    mkdir_(models_dir)
    
    data = get_data()
    n_timesteps, n_stocks = data.shape
    
    # split the train/test data
    n_train = int(n_timesteps * split)
    train_data = data[:n_train]
    test_data = data[n_train:]
    
    
    env = Environment(train_data, initial_cash)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = Agent(state_size, action_size)
    scaler = get_scaler(env)
    
    # store the final value of the portfolio 
    portfolio = []
    
    if mode == 'test':
        # load the previous model
        with open(f'{models_dir}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        # remake the env with test data
        env = Environment(test_data, initial_cash)
        
        # set epsilon to 0
        agent.epsilon = 0
        
        # load model
        from tensorflow import keras
        agent.model = load_model(models_dir)
       
    # play the game num_episodes times
    
    for e in tqdm(range(num_episodes)):
        start = time.time()
        val = play_one_episode(agent, env, mode, scaler)
        duration = time.time()-start
        if e%10:
            print(f"episode: {e + 1}/{num_episodes}, episode end value: {val[0]:.2f}, duration: {duration}")

        portfolio.append(val) # append episode end portfolio value
 
    if mode == 'train':
        # save the model
        agent.model.save(models_dir)
        
        # save the scaler
        with open(f'{models_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
     
    return agent, portfolio


def get_scaler(env):
    """
    generate random states to use to fit the StandardScalar
    """
    states = []
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break
    scaler = StandardScaler()
    scaler.fit(states)
    return scaler

def mkdir_(directory):
    """
    creates directory to store the model and the fitted scaler
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
             
def get_data():
    """
    function to extract the stock prices for the next step
    """
    df = pd.read_csv('../../data/fin_data/nflx_full.csv',
                        parse_dates=['Date']).set_index('Date')
    df = df['2015':]['Close']
    return df.values.reshape(-1,1)        