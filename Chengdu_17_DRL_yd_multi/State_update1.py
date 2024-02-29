import numpy as np
from Chengdu_17_line_parameters import Section1
from Chengdu_17_train_model import Train
import matplotlib.pyplot as plt

class multi_StateNode():
    def __init__(self, states, step):
        self.departure_time_interval = 120  # 发车间隔，单位为s
        self.headway = 0  # 两列车间行车间隔
        self.clock = 0
        self.delt_t = 0.5  # 时间间隔为1s
        self.line = Section1()
        self.train_model = Train()

        self.tra_acc = 0
        self.bra_acc = 0
        self.coa_acc = 0
        self.slope_acc = 0  # 坡度加速度
        self.radius_acc = 0  # 曲率加速度
        self.acc = 0.0
        self.slope = 0.0
        self.energy = 0.0

        self.step = step  # 当前步长
        self.action = np.array(0).reshape(1)
        self.train1_cur_state = states[0]
        self.train1_next_state = np.zeros(3)
        self.train2_cur_state = states[1]
        self.train2_next_state = np.zeros(3)
        self.train3_cur_state = states[2]
        self.train3_next_state = np.zeros(3)
        self.train_next_state = np.zeros(3)


    def get_action(self):
        self.action = np.random.uniform(-1, 1)
        self.action = np.array([self.action], dtype=np.float32)
    # 计算某位置处的限速
    def calculate_speed(self, x):
        sorted_positions = sorted(self.line.speed_limit.keys())

        for i in range(len(sorted_positions) - 1):
            start_position = sorted_positions[i]
            end_position = sorted_positions[i + 1]

            if start_position <= x < end_position:
                return self.line.speed_limit[start_position]

        return self.line.speed_limit[sorted_positions[-1]]

    # 计算某位置处的坡度
    def calculate_slope(self, x):
        sorted_positions = sorted(self.line.gradient.keys())

        for i in range(len(sorted_positions) - 1):
            start_position = sorted_positions[i]
            end_position = sorted_positions[i + 1]

            if start_position <= x < end_position:
                return self.line.gradient[start_position]

        return self.line.gradient[sorted_positions[-1]]

    # 计算某位置处的曲率
    def calculate_curve(self, x):
        sorted_positions = sorted(self.line.curve.keys())

        for i in range(len(sorted_positions) - 1):
            start_position = sorted_positions[i]
            end_position = sorted_positions[i + 1]

            if start_position <= x < end_position:
                return self.line.curve[start_position]

        return self.line.curve[sorted_positions[-1]]

    # 计算坡度加速度
    def cal_slope_acc(self,position):  # 当前位置，单位为m
        gradient = self.calculate_slope(position)
        self.slope_acc = -9.8 * gradient / 1000  # g_ acc = 9.8 * g /1000
        self.slope = gradient / 10

    # 计算曲线加速度
    def cal_radius_acc(self, position):  # 当前位置，单位为m
        curve = self.calculate_curve(position)
        if curve != 0:
            self.radius_acc = - 3 * 9.8 / (5 * curve)  # c_acc = 3g/5R
        else:
            self.radius_acc = 0

    # 计算合加速度
    def cal_acc(self, flag):
        if flag == 1:
            self.cal_slope_acc(self.train1_cur_state[0])
            self.cal_radius_acc(self.train1_cur_state[0])

            if self.action > 0:  # 牵引状态
                # cur_state[1]为m/s
                Cur_traction = self.train_model.get_max_traction_force(self.train1_cur_state[1] * 3.6) * abs(
                    self.action)
                self.tra_acc = Cur_traction / self.train_model.weight
                self.bra_acc = 0
                self.acc = self.tra_acc + self.bra_acc + self.slope_acc + self.radius_acc
            elif self.action < 0:  # 制动状态
                # print(self.cur_state[1] * 3.6)
                Cur_brake = self.train_model.get_max_brake_force(self.train1_cur_state[1] * 3.6) * abs(self.action)
                # print(Cur_brake)
                self.bra_acc = - 1.5 * Cur_brake / self.train_model.weight
                self.tra_acc = 0
                self.acc = self.tra_acc + self.bra_acc + self.slope_acc + self.radius_acc
            else:  # 惰行状态
                self.tra_acc = 0
                self.bra_acc = 0
                self.acc = self.tra_acc + self.bra_acc + self.slope_acc + self.radius_acc
        elif flag == 2:
            self.cal_slope_acc(self.train2_cur_state[0])
            self.cal_radius_acc(self.train2_cur_state[0])

            if self.action > 0:  # 牵引状态
                # cur_state[1]为m/s
                Cur_traction = self.train_model.get_max_traction_force(self.train2_cur_state[1] * 3.6) * abs(
                    self.action)
                self.tra_acc = Cur_traction / self.train_model.weight
                self.bra_acc = 0
                self.acc = self.tra_acc + self.bra_acc + self.slope_acc + self.radius_acc
            elif self.action < 0:  # 制动状态
                # print(self.cur_state[1] * 3.6)
                Cur_brake = self.train_model.get_max_brake_force(self.train2_cur_state[1] * 3.6) * abs(self.action)
                # print(Cur_brake)
                self.bra_acc = - 1.5 * Cur_brake / self.train_model.weight
                self.tra_acc = 0
                self.acc = self.tra_acc + self.bra_acc + self.slope_acc + self.radius_acc
            else:  # 惰行状态
                self.tra_acc = 0
                self.bra_acc = 0
                self.acc = self.tra_acc + self.bra_acc + self.slope_acc + self.radius_acc
        elif flag == 3:
            self.cal_slope_acc(self.train3_cur_state[0])
            self.cal_radius_acc(self.train3_cur_state[0])

            if self.action > 0:  # 牵引状态
                # cur_state[1]为m/s
                Cur_traction = self.train_model.get_max_traction_force(self.train3_cur_state[1] * 3.6) * abs(
                    self.action)
                self.tra_acc = Cur_traction / self.train_model.weight
                self.bra_acc = 0
                self.acc = self.tra_acc + self.bra_acc + self.slope_acc + self.radius_acc
            elif self.action < 0:  # 制动状态
                # print(self.cur_state[1] * 3.6)
                Cur_brake = self.train_model.get_max_brake_force(self.train3_cur_state[1] * 3.6) * abs(self.action)
                # print(Cur_brake)
                self.bra_acc = - 1.5 * Cur_brake / self.train_model.weight
                self.tra_acc = 0
                self.acc = self.tra_acc + self.bra_acc + self.slope_acc + self.radius_acc
            else:  # 惰行状态
                self.tra_acc = 0
                self.bra_acc = 0
                self.acc = self.tra_acc + self.bra_acc + self.slope_acc + self.radius_acc

            # 控制加速度绝对值在1.5以内
        if self.acc > 1.5:
            self.acc = np.array(1.5).reshape(1)
        if self.acc < -1.5:
            self.acc = np.array(-1.5).reshape(1)

    def get_next_state(self, flag):
        if flag == 1:
            v0 = self.train1_cur_state[1]
            vt = v0 + self.acc * self.delt_t
            x = abs((vt * vt - v0 * v0) / (2 * self.acc))

            self.train1_next_state[0] = x + self.train1_cur_state[0]
            self.train1_next_state[1] = vt
            self.train1_next_state[2] = (self.step + 1) * self.delt_t
        elif flag == 2:
            v0 = self.train2_cur_state[1]
            vt = v0 + self.acc * self.delt_t
            x = abs((vt * vt - v0 * v0) / (2 * self.acc))

            self.train2_next_state[0] = x + self.train2_cur_state[0]
            self.train2_next_state[1] = vt
            self.train2_next_state[2] = (self.step + 1) * self.delt_t
        elif flag == 3:
            v0 = self.train3_cur_state[1]
            vt = v0 + self.acc * self.delt_t
            x = abs((vt * vt - v0 * v0) / (2 * self.acc))

            self.train3_next_state[0] = x + self.train3_cur_state[0]
            self.train3_next_state[1] = vt
            self.train3_next_state[2] = (self.step + 1) * self.delt_t

        self.train_next_state = np.vstack(self.train1_next_state, self.train2_next_state, self.train3_next_state)

    def cal_energy(self):
        ave_v = (self.train1_cur_state[1] + self.train1_next_state[1]) / 2
        if self.action < 0:
            r_energy = self.train_model.get_re_power(self.delt_t, ave_v * 3.6, self.action)
            t_energy = np.array(0.0).reshape(1)
        else:
            r_energy = np.array(0.0).reshape(1)
            t_energy = self.train_model.get_traction_power(self.delt_t, ave_v * 3.6, self.action)
        self.energy = r_energy + t_energy


    def state_step(self,flag):
        self.get_action()
        self.cal_acc(flag)
        self.get_next_state(flag)
        self.cal_energy()


    # def state_step_multi(self):
    #     # 第一辆车启动
    #     self.get_action()
    #     self.cal_acc(self.train1_cur_state[0])
    #     self.get_next_state()
    #     energy1, t_energy1, r_energy1 = self.cal_energy()
    #     # 时钟开始转动
    #     if self.train1_next_state[2] == self.departure_time_interval:
    #         self.get_action()
    #         self.cal_acc(self.train2_cur_state[0])
    #     if self.train2_next_state[2] == self.departure_time_interval:
    #         self.get_action()
    #         self.cal_acc(self.train3_cur_state[0])

if __name__ == '__main__':
    #initial_state = np.array([0, 0, 0])  # 例如，初始位置为49，速度为10m/s，时间为10秒
    initial_states = [
        np.array([0, 0, 0]),  # 初始状态列车1
        np.array([0, 0, 120]),  # 初始状态列车2（120s后发车）
        np.array([0, 0, 240])  # 初始状态列车3（240s后发车）
    ]
    initial_step = 1
    multi_StateNode = multi_StateNode(initial_states, initial_step)
    speed = []
    position = []
    total_energy = 0
    line = Section1()
    speed2 = []
    position2 = []
    speed3 = []
    position3 = []

    while multi_StateNode.train1_cur_state[0] < line.length:
        multi_StateNode.state_step(1)
        position.append(multi_StateNode.train1_cur_state[0])
        speed.append(multi_StateNode.train1_cur_state[1])
        print("当前状态:", multi_StateNode.train1_cur_state)
        total_energy = total_energy + multi_StateNode.energy

        if multi_StateNode.train1_cur_state[2] >= 120:
            multi_StateNode.state_step(2)
            position2.append(multi_StateNode.train2_cur_state[0])
            speed2.append(multi_StateNode.train2_cur_state[1])
        if multi_StateNode.train1_cur_state[2] >= 240:
            multi_StateNode.state_step(3)
            position3.append(multi_StateNode.train3_cur_state[0])
            speed3.append(multi_StateNode.train3_cur_state[1])


        #print("累计能耗：", total_energy)
        multi_StateNode.step += 1
        multi_StateNode.train1_cur_state = multi_StateNode.train_next_state[0].copy()
        multi_StateNode.train2_cur_state = multi_StateNode.train_next_state[1].copy()
        multi_StateNode.train3_cur_state = multi_StateNode.train_next_state[2].copy()


    print("结束")
    plt.plot(position, speed)
    plt.xlabel('Position')  # 设置 x 轴标签
    plt.ylabel('Speed')  # 设置 y 轴标签
    plt.title('Speed vs Position')  # 设置图形标题
    plt.show()  # 显示图形



