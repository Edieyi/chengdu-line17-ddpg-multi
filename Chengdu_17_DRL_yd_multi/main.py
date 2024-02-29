import numpy as np
import matplotlib.pyplot as plt
import agent_ddpg
import agent2_ddpg
import agent3_ddpg
from Chengdu_17_train_model import Train
from Chengdu_17_line_parameters import Section
import torch
from State_update import MultiStateNode

torch.backends.cudnn.deterministic = True

obs_dim = 3
act_dim = 1
act_bound = [-1, 1]

agent1 = agent_ddpg.ddpg1(obs_dim, act_dim, act_bound)
agent2 = agent2_ddpg.ddpg2(obs_dim, act_dim, act_bound)
agent3 = agent3_ddpg.ddpg3(obs_dim, act_dim, act_bound)
line = Section["Section1"]
train = Train()

MAX_EPISODE = 1000

update_every = 20
batch_size = 64
noise = 0.1
total_rewardList = []
total_positionList = [[] for _ in range(3)]
total_velocityList = [[] for _ in range(3)]
total_timeList = []
total_state_list = []
total_slopeList = []
total_accList = []
episode_rewards = []

for episode in range(MAX_EPISODE):
    total_actionList = []
    update_cnt = 0
    rewardList = []
    positionList = []
    velocityList = []
    timeList = []
    state_list = []
    slopeList = []
    accList = []
    actionList = []
    maxReward = -np.inf
    total_energy = 0
    t_power = 0
    r_power = 0
    ep_reward = 0
    results = []
    act = 0

    action1 = 0.0
    action2 = 0.0
    action3 = 0.0

    reward_1 = 0.0
    reward_2 = 0.0
    reward_3 = 0.0

    t_energy_1 = 0.0
    t_energy_2 = 0.0
    t_energy_3 = 0.0
    r_energy_1 = 0.0
    r_energy_2 = 0.0
    r_energy_3 = 0.0

    done_1 = 0
    done_2 = 0
    done_3 = 0

    speed = [[] for _ in range(3)]
    position = [[] for _ in range(3)]
    times = []

    next_state_1 = np.array([0, 0, 0])
    next_state_2 = np.array([0, 0, 0])
    next_state_3 = np.array([0, 0, 0])

    initial_states = [
        np.array([0, 0, 0]),  # 初始状态列车1
        np.array([0, 0, 0]),  # 初始状态列车2（120s后发车）
        np.array([0, 0, 0])  # 初始状态列车3（240s后发车）
    ]

    last_state = [
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0])
    ]

    multi_state_node = MultiStateNode(initial_states, 0, episode, line, train, agent1, agent2, agent3)
    state_list.append(multi_state_node)

    step = 1
    max_step = (line.scheduled_time + 240) / multi_state_node.delt_t

    while step <= max_step:

        if multi_state_node.states[0][0] < line.length:
            next_state_1, t_energy_1, r_energy_1, energy_1, action1, reward_1, done_1 = multi_state_node.state_step(1)
            t = multi_state_node.states[0][2]
        else:
            next_state_1 = [line.length, 0, (multi_state_node.step + 1) * multi_state_node.delt_t]
            t = multi_state_node.states[0][2]

        #print(action1)
        multi_state_node.states[1][2] = t
        multi_state_node.states[2][2] = t

        if multi_state_node.states[0][2] >= 120:
            if multi_state_node.states[1][0] < line.length:
                next_state_2, t_energy_2, r_energy_2, energy_2, action2, reward_2, done_2 = multi_state_node.state_step(2)

                if multi_state_node.states[0][2] >= 240:
                    next_state_3, t_energy_3, r_energy_3, energy_3, action3, reward_3, done_3 = multi_state_node.state_step(3)

                else:
                    multi_state_node.states[2] = np.array([0.0, 0.0, t])

            else:
                next_state_2 = [line.length, 0, (multi_state_node.step + 1) * multi_state_node.delt_t]
        else:
            multi_state_node.states[1] = np.array([0.0, 0.0, t])


        t_power += t_energy_1 + t_energy_2 + t_energy_3
        r_power += r_energy_1 + r_energy_2 + r_energy_3
        total_energy = t_power + r_power

        ep_reward += reward_1 + reward_2 + reward_3

        position[0].append(multi_state_node.states[0][0])
        speed[0].append(multi_state_node.states[0][1])
        position[1].append(multi_state_node.states[1][0])
        speed[1].append(multi_state_node.states[1][1])
        position[2].append(multi_state_node.states[2][0])
        speed[2].append(multi_state_node.states[2][1])
        times.append(multi_state_node.states[0][2])
        actionList.append(action1)

        # 经验回放
        agent1.replay_buffer.store(np.array(multi_state_node.states[0].copy(), dtype=object),
                                   np.array(action1.copy(), dtype=object), np.array(reward_1.copy(), dtype=object),
                                   np.array(next_state_1.copy(), dtype=object), done_1)
        agent2.replay_buffer.store(np.array(multi_state_node.states[1], dtype=object), np.array(action2, dtype=object),
                                   np.array(reward_2, dtype=object), np.array(next_state_2, dtype=object), done_2)
        agent3.replay_buffer.store(np.array(multi_state_node.states[2], dtype=object), np.array(action3, dtype=object),
                                   np.array(reward_3, dtype=object), np.array(next_state_3, dtype=object), done_3)

        # 网络更新
        if episode > 100 and step % update_every == 0:
            if agent1.replay_buffer.size >= batch_size:
                for _ in range(2):
                    batch = agent1.replay_buffer.sample_batch(batch_size)
                    agent1.update(data=batch)

            if agent2.replay_buffer.size >= batch_size:
                for _ in range(2):
                    batch = agent2.replay_buffer.sample_batch(batch_size)
                    agent2.update(data=batch)

            if agent3.replay_buffer.size >= batch_size:
                for _ in range(2):
                    batch = agent3.replay_buffer.sample_batch(batch_size)
                    agent3.update(data=batch)

        if done_1:
            total_positionList.append(position)
            total_velocityList.append(speed)
            total_timeList.append(timeList.copy())
            total_actionList.append(actionList)
            actionList.clear()
            position.clear()
            speed.clear()
            timeList.clear()
            episode_rewards.append(ep_reward)
            break

        next_state = [next_state_1, next_state_2, next_state_3]
        multi_state_node = MultiStateNode(next_state, step, episode, line, train, agent1, agent2, agent3)
        results.append(multi_state_node.states)
        step += 1

    if episode % 20 == 0:
        print('Episode:', episode, '奖励:', ep_reward, '总运行能耗：', total_energy)
    # if episode == 100:
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(results[100][0][2], speed[0], label="Train 1")
    #     plt.plot(times, speed[1], label="Train 2")
    #     plt.plot(times, speed[2], label="Train 3")
    #     plt.xlabel('Time (s)')  # 设置 x 轴标签
    #     plt.ylabel('Speed (m/s)')  # 设置 y 轴标签
    #     plt.title('Speed vs Time')  # 设置图形标题
    #     plt.legend()  # 显示图例
    #     plt.grid(True)  # 添加网格线
    #     plt.show()  # 显示图形
        print(results[1])
        print(results[500])



# 绘制 ep_reward 的图形
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('Episode Rewards Over Time')
plt.show()



