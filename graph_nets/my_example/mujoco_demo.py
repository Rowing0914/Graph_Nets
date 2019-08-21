# run this from the terminal and make sure you are loading appropriate environment variables
# $ echo $LD_LIBRARY_PATH

import gym

env = gym.make("HalfCheetah-v2")
state = env.reset()
done = False
for t in range(1):
# while not done:
    # env.render()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    print(state)
