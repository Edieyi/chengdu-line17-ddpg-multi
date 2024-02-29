import numpy as np
from Chengdu_17_line_parameters import Section1
from Chengdu_17_train_model import Train
import matplotlib.pyplot as plt

np.random.seed(0)
class MultiStateNode():
    def __init__(self, states, step):
        self.departure_time_interval = 120  # 发车间隔，单位为s
        self.headway = 0  # 两列车间行车间隔
        self.delt_t = 1  # 时间间隔为1s
        self.line = Section1()
        self.train_model = Train()

        self.states = states
        self.step = step
        self.max_step = self.line.scheduled_time / self.delt_t
        self.action = np.array(0).reshape(1)

    def get_action(self, train_id):
        step2 = ((self.step + 1) * self.delt_t - self.departure_time_interval * (train_id - 1)) / self.delt_t - 1
        if step2 <= (1 / 3) * self.max_step:
            self.action = np.random.uniform(0, 0.8)
            return self.action
        elif (1 / 3) * self.max_step <= step2 <= (2 / 3) * self.max_step:
            self.action = np.random.uniform(-1, 1)
            return self.action
        else:
            self.action = np.random.uniform(-0.5, 0)
            return self.action

    def calculate_speed(self, x):
        sorted_positions = sorted(self.line.speed_limit.keys())
        for i in range(len(sorted_positions) - 1):
            start_position = sorted_positions[i]
            end_position = sorted_positions[i + 1]
            if start_position <= x < end_position:
                return self.line.speed_limit[start_position]
        return self.line.speed_limit[sorted_positions[-1]]

    def calculate_slope(self, x):
        sorted_positions = sorted(self.line.gradient.keys())
        for i in range(len(sorted_positions) - 1):
            start_position = sorted_positions[i]
            end_position = sorted_positions[i + 1]
            if start_position <= x < end_position:
                return self.line.gradient[start_position]
        return self.line.gradient[sorted_positions[-1]]

    def calculate_curve(self, x):
        sorted_positions = sorted(self.line.curve.keys())
        for i in range(len(sorted_positions) - 1):
            start_position = sorted_positions[i]
            end_position = sorted_positions[i + 1]
            if start_position <= x < end_position:
                return self.line.curve[start_position]
        return self.line.curve[sorted_positions[-1]]

    def cal_slope_acc(self, position):
        gradient = self.calculate_slope(position)
        return -9.8 * gradient / 1000

    def cal_radius_acc(self, position):
        curve = self.calculate_curve(position)
        if curve != 0:
            return - 3 * 9.8 / (5 * curve)
        else:
            return 0

    def cal_acc(self, position, velocity, action):
        slope_acc = self.cal_slope_acc(position)
        radius_acc = self.cal_radius_acc(position)

        if action > 0:  # 牵引状态
            Cur_traction = self.train_model.get_max_traction_force(velocity * 3.6) * abs(action)
            tra_acc = Cur_traction / self.train_model.weight
            return tra_acc + slope_acc + radius_acc
        elif action < 0:  # 制动状态
            Cur_brake = self.train_model.get_max_brake_force(velocity * 3.6) * abs(action)
            bra_acc = - 1.5 * Cur_brake / self.train_model.weight
            return bra_acc + slope_acc + radius_acc
        else:  # 惰行状态
            return slope_acc + radius_acc

    def get_next_state(self, position, velocity, acc):
        v0 = velocity
        vt = v0 + acc * self.delt_t
        x = abs((vt * vt - v0 * v0) / (2 * acc))
        if vt < 0:
            vt = np.array(0).reshape(1)
        return [x + position, vt, (self.step + 1) * self.delt_t]

    def cal_energy(self, velocity, action):
        ave_v = velocity
        if action < 0:
            return self.train_model.get_re_power(self.delt_t, ave_v * 3.6, action)
        else:
            return self.train_model.get_traction_power(self.delt_t, ave_v * 3.6, action)

    def state_step(self, train_id):
        position, velocity, _ = self.states[train_id - 1]
        action = self.get_action(train_id)
        acc = self.cal_acc(position, velocity, action)
        next_state = self.get_next_state(position, velocity, acc)
        self.states[train_id - 1] = next_state
        energy = self.cal_energy(velocity, action)
        return next_state, energy,action


if __name__ == '__main__':
    initial_states = [
        np.array([0, 0, 0]),  # 初始状态列车1
        np.array([0, 0, 0]),  # 初始状态列车2（120s后发车）
        np.array([0, 0, 0])  # 初始状态列车3（240s后发车）
    ]

    initial_step = 1
    multi_state_node = MultiStateNode(initial_states, initial_step)

    speed = [[] for _ in range(3)]
    position = [[] for _ in range(3)]
    total_energy = 0
    times = []
    line = Section1()
    max_step = (line.scheduled_time + 240) / multi_state_node.delt_t
    node_list = []

    energy_1 = 0
    energy_2 = 0
    energy_3 = 0

    step = 1
    while step <= max_step + 1:
        if multi_state_node.states[0][0] < line.length:
            next_state_1, energy_1, action1 = multi_state_node.state_step(1)

            t = multi_state_node.states[0][2]
        else:
            next_state_1 = [line.length, 0, (multi_state_node.step + 1) * multi_state_node.delt_t]
            t = multi_state_node.states[0][2]

        multi_state_node.states[1][2] = t
        multi_state_node.states[2][2] = t

        if multi_state_node.states[0][2] >= 120:
            if multi_state_node.states[1][0] < line.length:
                next_state_2, energy_2, action2 = multi_state_node.state_step(2)

                if multi_state_node.states[0][2] >= 240:
                    next_state_3, energy_3, action3 = multi_state_node.state_step(3)

                else:
                    multi_state_node.states[2] = np.array([0, 0, t])
            else:
                next_state_2 = [line.length, 0, (multi_state_node.step + 1) * multi_state_node.delt_t]
        else:
            multi_state_node.states[1] = np.array([0, 0, t])

        position[0].append(multi_state_node.states[0][0])
        speed[0].append(multi_state_node.states[0][1])
        position[1].append(multi_state_node.states[1][0])
        speed[1].append(multi_state_node.states[1][1])
        position[2].append(multi_state_node.states[2][0])
        speed[2].append(multi_state_node.states[2][1])
        times.append(multi_state_node.states[0][2])


        node_list.append(multi_state_node.states)
        next_state = [multi_state_node.states[0],multi_state_node.states[1],multi_state_node.states[2]]
        print(next_state)

        multi_state_node = MultiStateNode(next_state, step)

        # total_energy += energy_1 + energy_2 + energy_3
        step += 1


    print("结束")
    print(node_list)

    plt.figure(figsize=(10, 6))
    plt.plot(times, speed[0], label="Train 1")
    plt.plot(times, speed[1], label="Train 2")
    plt.plot(times, speed[2], label="Train 3")
    plt.xlabel('Time (s)')  # 设置 x 轴标签
    plt.ylabel('Speed (m/s)')  # 设置 y 轴标签
    plt.title('Speed vs Time')  # 设置图形标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 添加网格线
    plt.show()  # 显示图形
    plt.figure(figsize=(10, 6))

    plt.plot(position[0], speed[0], label="Train 1")
    plt.plot(position[1], speed[1], label="Train 2")
    plt.plot(position[2], speed[2], label="Train 3")
    plt.xlabel('Position (m)')  # 设置 x 轴标签
    plt.ylabel('Speed (m/s)')  # 设置 y 轴标签
    plt.title('Speed vs Position')  # 设置图形标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 添加网格线
    plt.show()  # 显示图形
