import pandas as pd
import numpy as np
import gym
# These imports are related to OpenAI Gym, a popular library for developing and comparing reinforcement learning environments.
from gym import spaces
from stable_baselines3 import PPO
#This import statement is for the Proximal Policy Optimization (PPO) algorithm, 
#which is one of the reinforcement learning algorithms available in the Stable Baselines3 library. 
#It's used for training RL agents.

# Loading the  historical stock data and preprocess it as needed
data = pd.read_csv('^NSEI.csv')
# Checking the Null values and removing them.
print("Number of NaN values in the data:")
print(data.isnull().sum())
data = data.ffill()
print("Number of NaN values in the data:")
print(data.isnull().sum())



# Calculate additional features
# Moving averages
data['50-day_MA'] = data['Close'].rolling(window=50).mean()
#This creates a window of 50 days and  then mean it 
#This moving average provides valuable insights into the stock's
#recent price trend and helps smoothen out short-term fluctuations, making it easier to identify the underlying trend.
data['200-day_MA'] = data['Close'].rolling(window=200).mean()

# Relative Strength Index (RSI)
# Implementing  RSI calculation here
def calculate_rsi(data, window=14):
    close_prices = data['Close']
    daily_price_changes = close_prices.diff()

    # Separate positive and negative price changes
    gains = daily_price_changes.where(daily_price_changes > 0, 0)
    losses = -daily_price_changes.where(daily_price_changes < 0, 0)

    # Calculate the average gain and average loss over the window period
    avg_gain = gains.rolling(window=window, min_periods=1).mean()
    avg_loss = losses.rolling(window=window, min_periods=1).mean()

    # Avoid division by zero by adding a small epsilon value
    epsilon = 1e-10

    # Calculate the relative strength (RS)
    rs = avg_gain / (avg_loss + epsilon)

    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi

 
data['RSI'] = calculate_rsi(data, window=200)
#RSI helps identify overbought (RSI > 70) and oversold (RSI < 30) conditions, aiding in potential trend reversal predictions.
#It confirms trend strength - RSI > 50 in uptrends, RSI < 50 in downtrends, guiding traders in trend-following strategies.
#RSI divergence offers early signals of trend changes, indicating possible trade entry or exit points.
# Normalize the data
data['50-day_MA'] = (data['50-day_MA'] - data['50-day_MA'].mean()) / data['50-day_MA'].std()
data['200-day_MA'] = (data['200-day_MA'] - data['200-day_MA'].mean()) / data['200-day_MA'].std()
#we transform the values of the 50-day and 200-day moving averages so that they have a mean of 0 and 
#a standard deviation of 1. This process makes it easier to compare and use these features in various
#machine learning algorithms and trading strategies, as the data is now on a standardized scale
data


# Step 3: Action Space

# Define the action granularity
action_granularity = 0.05  # 5% of the portfolio value

# Define the bounds for buying and selling actions
buy_action_bound = 0.2  # 20% of the portfolio value (Maximum 20% of portfolio can be invested in a single stock)
sell_action_bound = 0.2  # 20% of the current holding (Maximum 20% of current holding can be sold)

# Encode the action space
action_space = np.arange(-1, 2) * action_granularity  # Action space: [-0.05, 0, 0.05]

# Print the action space
print("Action Space:", action_space)

# Action Mapping
def map_action(action):
    if action < 0:
        return 'SELL'  # Sell a percentage of the current holding (up to sell_action_bound)
    elif action == 0:
        return 'HOLD'  # Hold current position
    else:
        return 'BUY'   # Buy a percentage of the portfolio value (up to buy_action_bound)

# Test the action mapping function
action = 0.1
print("Action:", action, "Mapped Action:", map_action(action))


#  the initial portfolio value (at the beginning of the training episode)
initial_portfolio_value = 1000000  

#  the function to calculate the reward
def calculate_reward(portfolio_value):
    # Calculate the percentage return of the portfolio
    percentage_return = (portfolio_value - initial_portfolio_value) / initial_portfolio_value

    # Assign reward based on percentage return
    if percentage_return > 0:
        # Positive return, provide a positive reward
        reward = percentage_return
    else:
        # Negative return, provide a negative reward (penalty)
        reward = percentage_return - 0.01  # Apply a penalty for negative returns

    return reward


class StockTradingEnvironment(gym.Env):
    def __init__(self, data):
        super(StockTradingEnvironment, self).__init__()

        self.data = data
        self.current_step = 0
        self.max_steps = len(data) - 1

        # Define the action space (continuous)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Define the observation space (state representation)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self._next_observation()

    def step(self, action):
        # Execute the action and get the next state, reward, done flag, and additional info
        self.current_step += 1
        done = self.current_step >= self.max_steps

        if done:
            return self._next_observation(), 0, done, {}

        reward = self._get_reward(action)
        obs = self._next_observation()

        return obs, reward, done, {}

    def _next_observation(self):
        return np.array([self.data.iloc[self.current_step]['Close']])

    def _get_reward(self, action):
        # Implement the reward function based on your desired stock return optimization
        # In this example, we provide a simple reward for positive returns
        current_price = self.data.iloc[self.current_step]['Close']
        next_price = self.data.iloc[self.current_step + 1]['Close']
        return (next_price - current_price) / current_price


 
env = StockTradingEnvironment(data)

# Create the RL agent using PPO
model = PPO("MlpPolicy", env, verbose=1)

# Train the RL agent using historical data
model.learn(total_timesteps=10000)  # Adjust total_timesteps as per need

# Save the trained model for later use
model.save("stock_trading_agent")

# After training, you can evaluate the agent's performance using a separate testing dataset
# Load the testing dataset
testing_data = pd.read_csv('^NSEI_test.csv')

# Check for NaN values in the testing data and handle them (if any)
testing_data = testing_data.ffill()

# Create a new instance of the environment with testing data
testing_env = StockTradingEnvironment(testing_data)

# Reset the environment to get the initial observation
obs = testing_env.reset()

done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = testing_env.step(action)
    total_reward += reward

print("Total reward during testing:", total_reward)
 
