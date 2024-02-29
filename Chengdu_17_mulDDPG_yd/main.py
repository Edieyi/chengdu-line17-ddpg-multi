import numpy as np
import matplotlib.pyplot as plt
import agent1_ddpg
import agent2_ddpg
import agent3_ddpg
from train_model import Train1
from train_model import Train2
from train_model import Train3
from line_parameters import Section
import torch
from state_update import MultiStateNode
import time
import pandas as pd

torch.backends.cudnn.deterministic = True
np.random.seed(1)
obs_dim = 7
act_dim = 1
act_bound = [-1, 1]

agent1 = agent1_ddpg.ddpg1(obs_dim, act_dim, act_bound)
agent2 = agent2_ddpg.ddpg2(obs_dim, act_dim, act_bound)
agent3 = agent3_ddpg.ddpg3(obs_dim, act_dim, act_bound)
line = Section["Section1"]
train1 = Train1()
train2 = Train2()
train3 = Train3()
time_interval = 120

MAX_EPISODE = 500
update_every = 20
batch_size = 64
noise = 0.1
total_positionList = [[] for _ in range(3)]
total_velocityList =[]
total_timeList = []
episode_rewards = []

start_time = time.time()  # 记录程序开始时间
for episode in range(MAX_EPISODE):
    update_cnt = 0

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
    total_energy = 0.0

    t_power = 0
    r_power = 0
    ep_reward = 0

    t_power_1 = 0
    t_power_2 = 0
    t_power_3 = 0

    r_power_1 = 0
    r_power_2 = 0
    r_power_3 = 0

    power_1 = 0
    power_2 = 0
    power_3 = 0

    done_1 = 0
    done_2 = 0
    done_3 = 0

    next_state_1 = np.array([0, 0, 0])
    next_state_2 = np.array([0, 0, 0])
    next_state_3 = np.array([0, 0, 0])

    times = []
    speed = [[] for _ in range(3)]
    position = [[] for _ in range(3)]

    initial_states = [
        np.array([0, 0, 0, 0, 0, 0, 0]),  # 初始状态列车1
        np.array([0, 0, 0, 0, 0, 0, 0]),  # 初始状态列车2（120s后发车）
        np.array([0, 0, 0, 0, 0, 0, 0])  # 初始状态列车3（240s后发车）
    ]

    multi_state_node = MultiStateNode(initial_states, 0, episode, line, train1, train2, train3, agent1, agent2, agent3)

    step = 1
    max_step = (line.scheduled_time + time_interval * 2) / multi_state_node.delt_t

    while step <= max_step + 1:
        # 检查列车3是否到站
        if multi_state_node.states[2][0] >= line.length:
            break  # 如果到站，跳出循环

        # 更新列车1的状态
        if multi_state_node.states[0][0] < line.length:
            next_state_1, t_energy_1, r_energy_1, energy_1, action1, reward_1 = multi_state_node.state_step(1)
        else:
            next_state_1 = [line.length, 0, (multi_state_node.step + 1) * multi_state_node.delt_t]

        # 更新列车2的状态
        next_state_2 = np.array([0, 0, multi_state_node.states[0][2] + multi_state_node.delt_t])
        if multi_state_node.states[0][2] >= time_interval:
            if multi_state_node.states[1][0] <= line.length:
                next_state_2, t_energy_2, r_energy_2, energy_2, action2, reward_2 = multi_state_node.state_step(2)
            else:
                next_state_2 = [line.length, 0, (multi_state_node.step + 1) * multi_state_node.delt_t]

        # 更新列车3的状态
        if multi_state_node.states[0][2] >= time_interval * 2:
            if multi_state_node.states[2][0] <= line.length:
                next_state_3, t_energy_3, r_energy_3, energy_3, action3, reward_3 = multi_state_node.state_step(3)
                if next_state_3[0] > line.length:
                    next_state_3 = [line.length, 0, (multi_state_node.step + 1) * multi_state_node.delt_t]
            else:
                next_state_3 = [line.length, 0, (multi_state_node.step + 1) * multi_state_node.delt_t]

        t_power_1 += t_energy_1
        t_power_2 += t_energy_2
        t_power_3 += t_energy_3
        r_power_1 += r_energy_1
        r_power_2 += r_energy_2
        r_power_3 += r_energy_3
        power_1 = t_power_1 + r_power_1
        power_2 = t_power_2 + r_power_2
        power_3 = t_power_3 + r_power_3

        t_power += t_energy_1 + t_energy_2 + t_energy_3
        r_power += r_energy_1 + r_energy_2 + r_energy_3
        total_energy = t_power + r_power

        ep_reward += reward_1 + reward_2 + reward_3

        train1_next_state = np.array([next_state_1[0], next_state_1[1], next_state_1[2], 0, 0, next_state_2[0], next_state_2[1]])
        train2_next_state = np.array([next_state_2[0], next_state_2[1], next_state_2[2], next_state_1[0], next_state_1[1], next_state_3[0], next_state_3[1]])
        train3_next_state = np.array([next_state_3[0], next_state_3[1], next_state_3[2], next_state_2[0], next_state_2[1], 0, 0])

        next_states = [train1_next_state, train2_next_state, train3_next_state]
        # 经验回放
        agent1.replay_buffer.store(np.array(multi_state_node.states[0].copy(), dtype=object),
                                   np.array(action1, dtype=object), np.array(reward_1, dtype=object),
                                   np.array(train1_next_state.copy(), dtype=object), done_1)
        agent2.replay_buffer.store(np.array(multi_state_node.states[1], dtype=object), np.array(action2, dtype=object),
                                   np.array(reward_2, dtype=object), np.array(train2_next_state, dtype=object), done_2)
        agent3.replay_buffer.store(np.array(multi_state_node.states[2], dtype=object), np.array(action3, dtype=object),
                                   np.array(reward_3, dtype=object), np.array(train3_next_state, dtype=object), done_3)

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

        times.append(multi_state_node.states[0][2])
        position[0].append(multi_state_node.states[0][0])
        speed[0].append(multi_state_node.states[0][1])
        position[1].append(multi_state_node.states[1][0])
        speed[1].append(multi_state_node.states[1][1])
        position[2].append(multi_state_node.states[2][0])
        speed[2].append(multi_state_node.states[2][1])

        total_positionList.append(position.copy())
        total_velocityList.append(speed.copy())
        total_timeList.append(times.copy())
        episode_rewards.append(ep_reward)


        multi_state_node = MultiStateNode(next_states, step, episode, line, train1, train2, train3, agent1, agent2, agent3)
        step += 1

    position.clear()
    speed.clear()
    times.clear()

    if episode % 20 == 0:
        print('Episode:', episode, '奖励:', ep_reward, '总运行能耗：', total_energy,
              '头车运行能耗：', power_1, '中间车运行能耗：', power_2, '尾车运行能耗：', power_3)

end_time = time.time()  # 记录程序结束时间
total_time = end_time - start_time  # 计算程序运行时间

print("程序运行时间：", total_time, "秒")

# 速度-时间图
plt.figure(figsize=(10, 6))
plt.plot(total_timeList[-1], total_velocityList[-1][0], label='Train 1')
plt.plot(total_timeList[-1], total_velocityList[-1][1], label='Train 2')
plt.plot(total_timeList[-1], total_velocityList[-1][2], label='Train 3')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Velocity-Time Graph for Three Trains')
plt.legend()
plt.grid(True)
plt.show()

# 速度-位置图
plt.figure(figsize=(10, 6))
plt.plot(total_positionList[-1][0], total_velocityList[-1][0], label='Train 1')
plt.plot(total_positionList[-1][1], total_velocityList[-1][1], label='Train 2')
plt.plot(total_positionList[-1][2], total_velocityList[-1][2], label='Train 3')
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.title('Velocity-Position Graph for Three Trains')
plt.legend()
plt.grid(True)
plt.show()


# 将列表转换为 pandas DataFrame
# 将列表转置后再转换为 pandas DataFrame
df_position = pd.DataFrame(np.transpose(total_positionList[-1]), columns=['Train1_Position', 'Train2_Position', 'Train3_Position'])
df_velocity = pd.DataFrame(np.transpose(total_velocityList[-1]), columns=['Train1_Velocity', 'Train2_Velocity', 'Train3_Velocity'])
df_time = pd.DataFrame(np.transpose(total_timeList[-1]), columns=['Time'])

# 创建 Excel writer 对象
excel_writer = pd.ExcelWriter('train_data.xlsx', engine='xlsxwriter')

# 将 DataFrame 写入 Excel 文件的不同 sheet 中
df_position.to_excel(excel_writer, sheet_name='Position', index=False)
df_velocity.to_excel(excel_writer, sheet_name='Velocity', index=False)
df_time.to_excel(excel_writer, sheet_name='Time', index=False)

# 保存 Excel 文件
excel_writer.save()

print("数据已成功保存到 train_data.xlsx 文件中。")






