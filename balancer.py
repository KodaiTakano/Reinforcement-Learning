import gym

env = gym.make('CartPole-v1')

# 20エピソード
for episode in range(20):
    # 初期化
    observation = env.reset()
    # 100ステップ
    for step in range(100):
        env.render()
        print(observation)
        # action_spaceの中からランダムに行動を決定
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        # 倒立振り子の棒が倒れるとdone=True
        if done:
            break
    print('Episode finished after {} timesteps'.format(step+1))
env.close()
