import numpy as np

def get_action(Q, pre_observation):
    return np.random.choice(np.ravel(np.argwhere(Q[pre_observation] == np.amax(Q[pre_observation]))))

def get_value(Q, pre_observation, action, N, reward):
    return Q[pre_observation, action] + ((1 / N[action]) * (reward - Q[pre_observation, action]))

class MultiArmedBandit:
    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon

    def fit(self, env, steps=1000):
        Q = np.zeros((env.observation_space.n, env.action_space.n))
        N = np.zeros((env.action_space.n, 1))
        rewards = np.zeros((100,1))

        r = int(steps / 100)
        total_reward = 0
        current_reward = 0
        currrent_index = 0
        
        observation = env.reset()

        step=1
        while step<steps:
          pre_observation = observation
          e = self.epsilon
          rand = np.random.random()

          if (rand < e):
            action = env.action_space.sample()
          else:
            action = get_action(Q, pre_observation)

          observation, reward, _, _ = env.step(action)

          N[action] += 1
          new_val = get_value(Q, pre_observation, action, N, reward)

          for row in Q:
            row[action] = new_val

          if (current_reward < r):
            total_reward += reward
            current_reward += 1
          else:
            rewards[currrent_index] = total_reward / r
            total_reward = 0 + reward
            current_reward = 1
            currrent_index += 1

          step+=1

        rewards[currrent_index] = total_reward / r

        return Q, rewards

    def predict(self, env, Q):
        s = np.empty([])
        a = np.empty([])
        r = np.empty([])

        observation = env.reset()
        done=False
        while (not(done)):
          action = np.argmax(Q[observation])
          observation, reward, done, _ = env.step(action)

          s = np.append(s, observation)
          a = np.append(a, action)
          r = np.append(r, reward)

        s=np.delete(s,0)
        a=np.delete(a,0)
        r=np.delete(r,0)

        return s, a, r
