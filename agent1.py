import gym

env = gym.make('CartPole-v1')
steps = []

# 20エピソード
for episode in range(20):
    # 初期化
    observation = env.reset()
    # 100ステップ
    for step in range(100):
        env.render()
        # th=棒の角度
        _, _, th, _ = observation
        # 棒が左に倒れたら支点を左に、右に倒れたら右に
        if th < 0:
            action = 0
        else:
            action = 1
        observation, reward, done, info = env.step(action)
        # 倒立振り子の棒が倒れるとdone=True
        if done:
            break
    print('Episode {} finished after {} timesteps'.format(episode+1, step+1))
    steps.append(step+1)
env.close()
