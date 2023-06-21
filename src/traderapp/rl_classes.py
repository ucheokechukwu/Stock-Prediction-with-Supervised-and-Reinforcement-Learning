import numpy as np

class Environment:
    """
    NFLX trading Environment
    state space: 
        0 -> number of NFLX shares owned
        1 -> price of NFLX shares
        2 -> cash balance
    action space: 
        0 -> sell
        1 -> hold
        2 -> buy
    """

    def __init__(self, data, initial_cash=10000):
        # data
        self.history = data
        # the number of steps is the len of the data
        self.n_step, self.n_stock = self.history.shape
        
        # attributes
        self.initial_cash = initial_cash
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None
        
        self.action_space = np.arange(3) 
        
        self.action_list = np.arange(3) 
        
        # size of state
        self.state_dim = 3
        self.reset()
        
    def reset(self):
        self.cur_step = 0
        self.stock_owned = 0
        self.stock_price = self.history[self.cur_step]
        self.cash_in_hand = self.initial_cash
        
        return self._get_obs()
    
    def step(self, action):
        # error if the action is not valid
        assert action in self.action_space
#         print('step', self.cur_step)
        
        # save previous value
        prev_val = self._get_val()
        
        # price update
        self.cur_step += 1
        self.stock_price = self.history[self.cur_step]
        
        # perform the trade
        self._trade(action)
        
        # get the new value & calc reward
        cur_val = self._get_val()
        reward = cur_val - prev_val
        
        # check if done 
        done = self.cur_step == self.n_step -1
        
        # store the current value
        info = {'current_value': cur_val}
        
        # same structure as OpenGym API
        return self._get_obs(), reward, done, info
    
    def _get_obs(self):
        # returns the state parameters - num_stock, price, cash
        obs = np.empty(self.state_dim)
        obs[0] = self.stock_owned
        obs[1] = self.stock_price
        obs[2] = self.cash_in_hand
        return obs
      
    def _get_val(self):
        # calculate portfolio value
        return self.stock_owned * self.stock_price + self.cash_in_hand
    
    def _trade(self, action):
        # 0 - sell
        # 1 - hold
        # 2 - buy
        
        sell_index = []
        buy_index = []
        action_vec = self.action_list[action]
        if action_vec == 0: 
            # sell all 
            self.cash_in_hand += self.stock_price * self.stock_owned
            self.stock_owned = 0
            
        if action_vec == 2:
            # buy minimum of 2 stocks/max of 5 stocks (possible hyperparameter value)
            if self.cash_in_hand > (2*self.stock_price):
                buy_stocks = 2
#                 buy_stocks = np.floor(self.cash_in_hand / self.stock_price)
                self.stock_owned += buy_stocks
                self.cash_in_hand -=  buy_stocks * self.stock_price
            else:
                pass
def get_data():
    """
    function to extract the stock prices for the next step
    """
    df = pd.read_csv('../data/nflx_full.csv',
                        parse_dates=['Date']).set_index('Date')
    df = df['2015':]['Close']
    return df.values.reshape(-1,1)


class Agent():
    """
    Agent class
    """

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.97 # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.005 ###########
#         self.model_name = model_name
        self.model = self._model() # will be overwritten in test mode
        
    def act(self, state):
        import numpy as np
        if np.random.rand() <=self.epsilon:
            return np.random.choice(self.action_size)
        # if not epsilon, perform greedy action
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) # returns action which gives the largest Q value
    
    def train(self, state, action, reward, next_state, done):
        if done:
            target = reward 
        else:
            target = reward * self.gamma + np.amax(self.model.predict(next_state), axis=1)
        # target_full : number of samples, num_outputs
        target_full = self.model.predict(state)
        target_full[0, action] = target
        
        # run one training step
        self.model.fit(state, target_full, epochs=1, verbose=0, steps_per_epoch=1)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            
            
    def _model(self):
        from keras.models import Sequential
        from keras.models import load_model
        from keras.layers import Dense 
        from keras.optimizers import Adam 
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
        return model
  
        
        



   