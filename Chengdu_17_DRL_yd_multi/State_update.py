import numpy as np
from Chengdu_17_line_parameters import Section1
from Chengdu_17_train_model import Train

class MultiStateNode():
    def __init__(self, states, step, episode, line, train, agent1, agent2, agent3):
        self.departure_time_interval = 120  # 发车间隔，单位为s
        self.headway = 0  # 两列车间行车间隔
        self.delt_t = 1  # 时间间隔为1s
        self.line = line
        self.train_model = train
        self.states = states
        self.step = step
        self.max_step = (line.scheduled_time + self.departure_time_interval * 2) / self.delt_t
        self.episode = episode
        self.agent1 = agent1
        self.agent2 = agent2
        self.agent3 = agent3
        self.noise = 0.1
        self.reward = 0.0
        self.energy = 0.0
        self.t_energy = 0.0
        self.r_energy = 0.0

    def get_action(self, train_id, cur_state):
        step2 = ((self.step + 1) * self.delt_t - self.departure_time_interval * (train_id - 1)) / self.delt_t - 1
        if step2 <= (1 / 3) * self.max_step:
            action = np.array([np.random.uniform(0, 1)], dtype=np.float32)
            return action
        elif (1 / 3) * self.max_step <= step2 <= (2 / 3) * self.max_step:
            if train_id == 1:
                action = self.agent1.get_action(cur_state, self.noise)
                return action
            if train_id == 2:
                action = self.agent2.get_action(cur_state, self.noise)
                return action
            if train_id == 3:
                action = self.agent3.get_action(cur_state, self.noise)
                return action
        else:
            action = np.array([np.random.uniform(-1, 0)], dtype=np.float32)
            return action

    def get_last_node(self, state_list):
        if len(state_list) > 1:
            self.last_node_state_0 = state_list[self.step - 1].states[0]
            self.last_node_state_1 = state_list[self.step - 1].states[1]
            self.last_node_state_2 = state_list[self.step - 1].states[2]
            self.last_node_state = [self.last_node_state_0, self.last_node_state_1, self.last_node_state_2]
        else:
            self.last_node_state = [
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0])
    ]

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
        return [x + position, vt, (self.step + 1) * self.delt_t]

    def cal_energy(self, cur_state, next_state, action):
        ave_v = (cur_state[1] + next_state[1]) / 2
        if action < 0:
            self.t_energy = np.array(0.0).reshape(1)
            self.r_energy = self.train_model.get_re_power(self.delt_t, ave_v * 3.6, action)
            self.energy = self.t_energy + self.r_energy

        else:
            self.t_energy = self.train_model.get_traction_power(self.delt_t, ave_v * 3.6, action)
            self.r_energy = np.array(0.0).reshape(1)
            self.energy = self.t_energy + self.r_energy

    def cal_reward(self, next_state):
        if self.step == self.max_step:  #最后一阶段
            done = 1
            if abs(next_state[1]-self.line.length) <= 3:
                stop_punish = 10
            else:
                stop_punish = -100
            self.reward = abs(self.r_energy) + stop_punish
        else:
            done = 0  # 运行过程中
            self.reward = abs(self.r_energy)
            return self.reward, done

    def state_step(self, train_id):
        position, velocity, t = self.states[train_id - 1]
        cur_state = np.array([position, velocity, t], dtype=np.float32)  # 将列表转换为 numpy 数组
        action = self.get_action(train_id, cur_state)
        acc = self.cal_acc(position, velocity, action)
        next_state = self.get_next_state(position, velocity, acc)
        next_state = np.array(next_state, dtype=np.float32)
        self.cal_energy(cur_state, next_state, action)
        self.reward, done = self.cal_reward(next_state)
        return next_state, self.t_energy, self.r_energy, self.energy, action, self.reward, done
