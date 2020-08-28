import numpy as np

def get_action(Q, pre_observation):
    return np.random.choice(np.ravel(np.argwhere(Q[pre_observation] == np.amax(Q[pre_observation]))))

def big_boi(Q, pre_observation, action, Q_copy, reward, discount, observation):
    return Q[pre_observation,action]+((1 /Q_copy[pre_observation,action])*(reward+(discount*np.amax(Q[observation]))-Q[pre_observation, action]))


class QLearning:
    def __init__(self, epsilon=0.2, discount=0.95, adaptive=False):
        self.epsilon = epsilon
        self.discount = discount
        self.adaptive = adaptive

    def fit(self, env, steps=1000):
        Q = np.zeros((env.observation_space.n, env.action_space.n))
        rewards = np.zeros((100,1))

        Q_copy = np.zeros((env.observation_space.n, env.action_space.n))

        reeeee = np.floor(steps / 100)
        reward_value = 0
        current_reward = 0
        current_index = 0
        
        observation = env.reset()
        
        step=0
        while step<steps:
          pre_observation = observation

          if (np.random.random() < self._get_epsilon(step/steps)):
            action = env.action_space.sample()
          else:
            action=get_action(Q, pre_observation)

          Q_copy[pre_observation, action] += 1

          observation, reward, done, _ = env.step(action)

          Q[pre_observation,action] = big_boi(Q, pre_observation, action, Q_copy, reward, self.discount, observation)

          if (current_reward < reeeee):
            reward_value += reward
            current_reward += 1
          else:
            rewards[current_index]=reward_value/reeeee
            reward_value =0+reward
            current_reward=1
            current_index+=1

          if done:
              observation = env.reset()

          step+=1

          rewards[current_index] = reward_value/reeeee
        
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

    def _get_epsilon(self, progress):
        return self._adaptive_epsilon(progress) if self.adaptive else self.epsilon

    def _adaptive_epsilon(self, progress):
        return (1-progress) * self.epsilon
