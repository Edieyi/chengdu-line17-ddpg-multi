import numpy as np

class MultiStateNode():
    def __init__(self, states, step, episode, line,train1,train2,train3, agent1, agent2, agent3):
        self.time_interval = 120
        self.delt_t = 1
        self.line = line
        self.states = states
        self.train1 = train1
        self.train2 = train2
        self.train3 = train3
        self.agent1 = agent1
        self.agent2 = agent2
        self.agent3 = agent3
        self.noise = 0.1

        self.departure_time_interval = 120
        self.step = step
        self.max_step = (line.scheduled_time + self.departure_time_interval * 2) / self.delt_t
        self.max_step_0 = line.scheduled_time / self.delt_t
        self.episode = episode
        self.action = np.array(0).reshape(1)

        self.reward = 0.0
        self.energy = 0.0
        self.t_energy = 0.0
        self.r_energy = 0.0

    def get_action(self, train_id, cur_state):
        step2 = ((self.step + 1) * self.delt_t - self.departure_time_interval * (train_id - 1)) / self.delt_t - 1
        if train_id == 1:
            if step2 <= (1 / 3) * self.max_step_0:
                self.action = np.random.uniform(0, 0.8)
                return self.action
            elif (1 / 3) * self.max_step_0 <= step2 <= (2 / 3) * self.max_step_0:
                self.action = self.agent1.get_action(cur_state, self.noise)
                return self.action
            else:
                self.action = np.random.uniform(-0.5, 0)
                return self.action
        if train_id == 2:
            if step2 <= (1 / 3) * self.max_step_0:
                self.action = np.random.uniform(0, 0.8)
                return self.action
            elif (1 / 3) * self.max_step_0 <= step2 <= (2 / 3) * self.max_step_0:
                self.action = self.agent2.get_action(cur_state, self.noise)
                return self.action
            else:
                self.action = np.random.uniform(-0.5, 0)
                return self.action
        if train_id == 3:
            if step2 <= (1 / 3) * self.max_step_0:
                self.action = np.random.uniform(0, 0.8)
                return self.action
            elif (1 / 3) * self.max_step_0 <= step2 <= (2 / 3) * self.max_step_0:
                self.action = self.agent3.get_action(cur_state, self.noise)
                return self.action
            else:
                self.action = np.random.uniform(-0.5, 0)
                return self.action
        # if self.episode > 100:
        #     if train_id == 1:
        #         self.action = self.agent1.get_action(cur_state, self.noise)
        #         return self.action
        #     elif train_id == 2:
        #         self.action = self.agent2.get_action(cur_state, self.noise)
        #         return self.action
        #     elif train_id == 3:
        #         self.action = self.agent3.get_action(cur_state, self.noise)
        #         return self.action
        # else:
        #     self.action = np.random.uniform(-0.5, 0)
        #     return self.action



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

    def cal_acc(self, position, velocity, action, train_id):
        slope_acc = self.cal_slope_acc(position)
        radius_acc = self.cal_radius_acc(position)

        if train_id == 1:
            if action > 0:  # 牵引状态
                Cur_traction = self.train1.get_max_traction_force(velocity * 3.6) * abs(action)
                tra_acc = Cur_traction / self.train1.weight
                return tra_acc + slope_acc + radius_acc
            elif action < 0:  # 制动状态
                Cur_brake = self.train1.get_max_brake_force(velocity * 3.6) * abs(action)
                bra_acc = - 1.5 * Cur_brake / self.train1.weight
                return bra_acc + slope_acc + radius_acc
            else:  # 惰行状态
                return slope_acc + radius_acc
        elif train_id == 2:
            if action > 0:  # 牵引状态
                Cur_traction = self.train2.get_max_traction_force(velocity * 3.6) * abs(action)
                tra_acc = Cur_traction / self.train2.weight
                return tra_acc + slope_acc + radius_acc
            elif action < 0:  # 制动状态
                Cur_brake = self.train2.get_max_brake_force(velocity * 3.6) * abs(action)
                bra_acc = - 1.5 * Cur_brake / self.train2.weight
                return bra_acc + slope_acc + radius_acc
            else:  # 惰行状态
                return slope_acc + radius_acc
        elif train_id == 3:
            if action > 0:  # 牵引状态
                Cur_traction = self.train3.get_max_traction_force(velocity * 3.6) * abs(action)
                tra_acc = Cur_traction / self.train3.weight
                return tra_acc + slope_acc + radius_acc
            elif action < 0:  # 制动状态
                Cur_brake = self.train3.get_max_brake_force(velocity * 3.6) * abs(action)
                bra_acc = - 1.5 * Cur_brake / self.train3.weight
                return bra_acc + slope_acc + radius_acc
            else:  # 惰行状态
                return slope_acc + radius_acc

    def atp_limit(self, cur_state, next_state):
        limit_speed_next = self.calculate_speed(next_state[0])
        length = next_state[0] - cur_state[0]
        atp_limitspeed = np.sqrt(
                limit_speed_next * limit_speed_next / 3.6 / 3.6 - 2 * self.train1.max_bra_acc * length ) * 3.6
        return atp_limitspeed

    def get_next_state(self, position, velocity, acc):
        v0 = velocity
        if position > 0 and velocity < 0:
            vt = np.array(0).reshape(1)
            x = vt * self.delt_t
        else:
            if acc == 0:
                vt = v0
                x = vt * self.delt_t
            else:
                vt = v0 + acc * self.delt_t
                x = abs((vt * vt - v0 * v0) / (2 * acc))

        return [x + position, vt, (self.step + 1) * self.delt_t]

    def cal_energy(self, cur_state, next_state, action, train_id):
        ave_v = (cur_state[1] + next_state[1]) / 2
        if train_id == 1:
            if action < 0:
                self.t_energy = np.array(0.0).reshape(1)
                self.r_energy = self.train1.get_re_power(self.delt_t, ave_v * 3.6, action)
                self.energy = self.t_energy + self.r_energy

            else:
                self.t_energy = self.train1.get_traction_power(self.delt_t, ave_v * 3.6, action)
                self.r_energy = np.array(0.0).reshape(1)
                self.energy = self.t_energy + self.r_energy
        if train_id == 2:
            if action < 0:
                self.t_energy = np.array(0.0).reshape(1)
                self.r_energy = self.train2.get_re_power(self.delt_t, ave_v * 3.6, action)
                self.energy = self.t_energy + self.r_energy

            else:
                self.t_energy = self.train2.get_traction_power(self.delt_t, ave_v * 3.6, action)
                self.r_energy = np.array(0.0).reshape(1)
                self.energy = self.t_energy + self.r_energy
        if train_id == 3:
            if action < 0:
                self.t_energy = np.array(0.0).reshape(1)
                self.r_energy = self.train3.get_re_power(self.delt_t, ave_v * 3.6, action)
                self.energy = self.t_energy + self.r_energy

            else:
                self.t_energy = self.train3.get_traction_power(self.delt_t, ave_v * 3.6, action)
                self.r_energy = np.array(0.0).reshape(1)
                self.energy = self.t_energy + self.r_energy

    def cal_reward(self, cur_state, next_state, train_id):
        next_limit_speed = self.calculate_speed(next_state[0])
        step0 = ((self.step + 1) * self.delt_t - self.departure_time_interval * (train_id - 1)) / self.delt_t - 1

        if train_id == 1 or train_id == 2:
            if next_state[0] == self.line.length and next_state[1] == 0: # 到站
                if abs(cur_state[0] - self.line.length) <= 3:
                    stop_punish = 10
                else:
                    stop_punish = -100
                self.reward = abs(self.r_energy) + stop_punish
            else:
                if next_state[1] >= next_limit_speed:
                    speed_punish = -10
                else:
                    speed_punish = 0
                self.reward = abs(self.r_energy) + speed_punish
            return self.reward
        if train_id == 3:
            if self.step == self.max_step:
                if abs(cur_state[0] - self.line.length) <= 3:
                    stop_punish = 10
                else:
                    stop_punish = -100
                self.reward = abs(self.r_energy) + stop_punish
            if step0 < self.max_step_0:
                if next_state[1] >= next_limit_speed:
                    speed_punish = -10
                else:
                    speed_punish = 0
                self.reward = abs(self.r_energy) + speed_punish
            return self.reward


    def state_step(self, train_id):
        cur_train_position, cur_train_velocity, t, last_train_position, last_train_velocity, next_train_position, next_train_velocity = self.states[train_id - 1]
        cur_state = np.array([cur_train_position, cur_train_velocity, t], dtype=np.float32)
        cur_state0 = np.array([cur_train_position, cur_train_velocity, t, last_train_position, last_train_velocity, next_train_position, next_train_velocity], dtype=np.float32)
        action = self.get_action(train_id, cur_state0)
        acc = self.cal_acc(cur_train_position, cur_train_velocity, action, train_id)
        next_state = self.get_next_state(cur_train_position, cur_train_velocity, acc)
        next_state = np.array(next_state, dtype=np.float32)
        self.cal_energy(cur_state, next_state, action, train_id)
        self.reward = self.cal_reward(cur_state, next_state, train_id)
        return next_state, self.t_energy, self.r_energy, self.energy, action, self.reward




