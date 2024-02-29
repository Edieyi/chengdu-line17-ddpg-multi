import numpy as np
import torch
from torch.optim import Adam
import agent_core as core
from copy import deepcopy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn


class ddpg1:
    def __init__(self, obs_dim, act_dim, act_bound):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_bound = act_bound
        self.ac = core.MLPActorCritic
        self.seed = 0

        self.replay_size = 1000000
        self.gamma = 0.99  # 折扣因子
        self.polyak = 0.995  # 目标网络滑动平均的权重
        self.pi_lr = 0.01  # 演员网络学习率
        self.q_lr = 0.01  # 网络的学习率
        self.act_noise = 0.01

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # 构建演员评论家网络
        self.actor_critic = self.ac(self.obs_dim, self.act_dim, self.act_bound).to(device)

        # 复制参数到目标网络
        self.target_actor_critic = deepcopy(self.actor_critic).to(device)

        self.pi_optimizer = Adam(self.actor_critic.pi.parameters(), lr=self.pi_lr)
        self.q_optimizer = Adam(self.actor_critic.q.parameters(), lr=self.q_lr)

        for i in self.target_actor_critic.parameters():
            i.requires_grad = False

        # 构建经验回放缓冲区
        self.replay_buffer = ReplayBuffer1(obs_dim, act_dim, size=self.replay_size)

    # 计算值函数损失
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = self.actor_critic.q(o, a)

        with torch.no_grad():
            q_pi_targ = self.target_actor_critic.q(o, self.target_actor_critic.pi(o))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        loss_q = ((q - backup) ** 2).mean()

        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info

    # 计算策略损失
    def compute_loss_pi(self, data):
        o = data['obs']
        q_pi = self.actor_critic.q(o, self.actor_critic.pi(o))
        return -q_pi.mean()

    # 网络权重更新函数
    def update(self, data):
        # 更新值函数网络的权重
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        for p in self.actor_critic.q.parameters():
            p.requires_grad = False

        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        for p in self.actor_critic.q.parameters():
            p.requires_grad = True

        # 更新目标网络，用polyak滑动平均
        with torch.no_grad():
            for p, p_targ in zip(self.actor_critic.parameters(), self.actor_critic.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    # 动作选取
    def get_action(self, o, noise_scale):
        a = self.actor_critic.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, self.act_bound[0], self.act_bound[1])

# 经验缓放区
class ReplayBuffer1:
    """
    FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        # 观察值
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        # 动作值
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        # 奖励
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # 标志，表示环境是否结束
        self.done_buf = np.zeros(size, dtype=np.float32)
        # ptr 表示当前存储位置，size 表示当前存储的样本数量，max_size 表示缓冲区的最大容量。
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        # done: 表示环境是否结束的标志（1表示结束，0表示未结束）
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    # 从回放缓冲区中随机采样一个批次的经验数据
    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k,v in batch.items()}





