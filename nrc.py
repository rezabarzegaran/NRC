import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations_with_replacement

# Specify the GPU to use and memory usage
selected_gpu = 2
usage_percentage = 0.1

device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("mps is available and set")
elif torch.cuda.is_available():
    device = torch.device(f"cuda:{selected_gpu}")
    total_memory = torch.cuda.get_device_properties(selected_gpu).total_memory
    max_memory = int(total_memory * usage_percentage)
    torch.cuda.set_per_process_memory_fraction(usage_percentage)
    torch.cuda.empty_cache()
    print(f"Selected GPU: {torch.cuda.get_device_name(selected_gpu)}")
    print(f"GPU Memory Limit: {max_memory} bytes")
else:
    print("cpu is selected")


# Define the Agent class
class Agent:
    def __init__(self, agent_id, L, n, m, neighbors, comm_neighbors, W, K_initial, xi0, alpha,
                 learning_mode='constant'):
        """
        Initialize an agent in the multi-agent system.

        Parameters:
        - agent_id: Unique ID of the agent.
        - n: Dimension of the state.
        - m: Dimension of the control input.
        - neighbors: List of agent's neighbors.
        - comm_neighbors: List of agents it communicates with.
        - W: Weight matrix for communication.
        - K_initial: Initial control gain matrix.
        - xi0: Initial state value.
        - alpha: Learning rate for parameter updates.
        - learning_mode: 'constant' 使用固定学习率，'adaptive' 则使用自适应学习率（例如 RMSProp）
        """
        self.id = agent_id
        self.L = L
        self.n = n
        self.m = m
        self.neighbors = neighbors
        self.comm_neighbors = comm_neighbors
        self.W = W.to(device)
        self.K = torch.tensor(K_initial, device=device, dtype=torch.float32).view(m, -1)
        self.alpha = alpha

        # Initialize state estimate Z_i(0)
        self.Z = torch.zeros((L, 1), device=device, dtype=torch.float32)
        self.Z[self.id, 0] = xi0

        # Dimension of the theta parameter: phi dimension is determined by state+control dimension
        self.theta_dim = self._get_phi_dim()
        self.theta = torch.zeros((1, self.theta_dim), device=device, dtype=torch.float32)
        self.learning_mode = learning_mode

        # 如果使用自适应学习率，则初始化 RMSProp 所需参数
        if self.learning_mode != 'constant':
            self.adaptive_beta = 0.9  # 衰减因子
            self.adaptive_epsilon = 1e-8  # 防止除零的小常数
            self.grad_sq_accum = torch.zeros_like(self.theta)  # 累积梯度平方

    def _get_phi_dim(self):
        """
        Calculate the dimension of the phi parameter.
        """
        size = self.n + self.L  # 拼接后向量的长度：状态维度 + L
        return size * (size + 1) // 2

    def get_phi(self, ZU1, ZU2):
        """
        Generate the feature vector phi using XU (state and control input).

        Parameters:
        - ZU1: Concatenation of state and control input at current time.
        - ZU2: Concatenation of state and control input at next time.

        Returns:
        - phi: Feature vector of upper triangular elements.
        """
        size = ZU1.shape[0]
        quadratic_terms1 = []
        for i, j in combinations_with_replacement(range(size), 2):
            quadratic_terms1.append(ZU1[i] * ZU1[j])
        y1 = torch.tensor(quadratic_terms1, device=device, dtype=torch.float32)

        quadratic_terms2 = []
        for i, j in combinations_with_replacement(range(size), 2):
            quadratic_terms2.append(ZU2[i] * ZU2[j])
        y2 = torch.tensor(quadratic_terms2, device=device, dtype=torch.float32)

        phi = y1 - y2
        return phi.view(1, -1)

    def update_theta(self, phi, g):
        """
        Update the theta parameter using gradient descent with an adaptive learning rate (RMSProp).

        Parameters:
        - phi: Feature vector.
        - g: Stage cost.
        """
        # 计算预测值和误差
        pred = self.theta @ phi.T  # (1,1)
        error = pred - g  # 标量误差
        grad = error * phi  # 梯度

        if self.learning_mode == 'constant':
            effective_rate = self.alpha
        else:
            # 使用 RMSProp 自适应学习率
            self.grad_sq_accum = self.adaptive_beta * self.grad_sq_accum + (1 - self.adaptive_beta) * (grad ** 2)
            effective_rate = self.alpha / (torch.sqrt(self.grad_sq_accum) + self.adaptive_epsilon)

        # 更新 theta 参数（逐元素更新）
        self.theta = self.theta - effective_rate * grad

    def reconstruct_H_and_update_K(self):
        """
        Reconstruct matrix H from theta and update the control gain K.
        """
        u = (-self.K @ self.Z)
        z = self.Z.clone()
        zu1 = torch.cat((z, u), dim=0)
        zu = zu1.view(-1, 1)
        size = zu.shape[0]

        # 构造对称矩阵 H
        H = torch.zeros((size, size), device=device, dtype=torch.float32)
        indices = torch.triu_indices(size, size)
        H[indices[0], indices[1]] = self.theta.view(-1)
        H = H + H.T - torch.diag(torch.diag(H))

        start_block = size - self.m
        # 提取 H_22 和 H_21
        H_22 = H[start_block:size, start_block:size]
        H_21 = H[start_block:size, 0:start_block]
        H_22_inv = torch.linalg.pinv(H_22)
        K_new = -H_22_inv @ H_21

        self.K = K_new.clone()

    def get_cost(self, X, U):
        P = torch.eye(1, device=device, dtype=torch.float32)
        R = torch.eye(1, device=device, dtype=torch.float32)
        g = X.T @ P @ X + U.T @ R @ U
        return g

    def state_tracking_update(self, x_observations, mode):
        """
        Update the agent's state estimation.

        Parameters:
        - x_observations: Observed states of all agents.
        """
        Z_temp = torch.zeros((4, 1), device=device)
        for j in range(4):
            Z_temp[j, 0] = x_observations[j]

        new_Z = torch.zeros_like(self.Z)
        for j in range(4):
            new_Z += self.W[self.id, j] * Z_temp if j in self.comm_neighbors or j == self.id else 0 * Z_temp

        if self.learning_mode == 'estimate':
            self.Z = new_Z
        else:
            self.Z = Z_temp


# Define the MultiAgentSystem class
class MultiAgentSystem:
    def __init__(self, agents, A_global, B_global, W, K_opt, alpha=0.01, epsilon_K=1e-3):
        """
        Initialize the multi-agent system.

        Parameters:
        - agents: List of agents in the system.
        - A_global: System dynamics matrix A.
        - B_global: System dynamics matrix B.
        - W: Communication weight matrix.
        - K_opt: Optimal control gain matrix.
        - alpha: Learning rate.
        - epsilon_K: Convergence threshold for K.
        """
        self.agents = agents
        self.A_global = torch.tensor(A_global, device=device, dtype=torch.float32)
        self.B_global = torch.tensor(B_global, device=device, dtype=torch.float32)
        self.W = W.to(device)
        self.K_opt = torch.tensor(K_opt, device=device, dtype=torch.float32)
        self.alpha = alpha
        self.epsilon_K = epsilon_K
        self.L = len(agents)
        self.final_K = None

    def get_stage_cost_global(self, X, U):
        P = torch.eye(4, device=device)
        R = torch.eye(4, device=device)
        g = X.T @ P @ X + U.T @ R @ U
        return g

    def generate_excitation_noise(self, t, mode='decay', c=0.9999, rho=1.0):
        if mode == 'decay':
            v_p = c ** (t * rho)
        elif mode == 'no_noise':
            v_p = 0
        else:
            v_p = 1.0
        noise = 0.1 * torch.randn((self.L, 1), device=device) * v_p
        return noise

    def run_simulation(self, X0, N, Q, T, mode='decay', policy_save_path='final_policy.pt', tracking_mode='estimate',
                       control_mode='no_clamp'):
        X_traj = torch.zeros((T + 1, self.L), device=device)
        X_traj[0] = torch.tensor(X0, device=device, dtype=torch.float32)
        G = torch.zeros((T + 1, self.L), device=device)
        Q_val = torch.zeros((T + 1, self.L), device=device)
        k_diff_history = []

        q = 0
        converged = False
        t = 0
        while q < Q and not converged:
            old_thetas = [agent.theta.clone() for agent in self.agents]
            q += 1
            # Policy evaluation (ST-E)
            for p in range(N):
                if t + 1 > T:
                    break

                noise = self.generate_excitation_noise(t, mode=mode)
                U_list = []
                for i, agent in enumerate(self.agents):
                    u_i_no_noise = (-agent.K @ agent.Z)
                    u_i = u_i_no_noise + noise[i]
                    U_list.append(u_i)
                U = torch.cat(U_list, dim=0)

                X_cur = X_traj[t].view(-1, 1).clone()
                X_next = self.A_global @ X_cur + self.B_global @ U
                if control_mode == 'clamp':
                    X_next = torch.clamp(X_next, -0.025, 0.025)
                X_traj[t + 1] = X_next.view(-1).clone()

                x_observations = X_traj[t + 1].view(-1).clone()

                phi_list = []
                for i, agent in enumerate(self.agents):
                    z_curr = agent.Z
                    u_curr = U_list[i]
                    zu_curr = torch.cat((z_curr, u_curr), dim=0)
                    agent.state_tracking_update(x_observations, tracking_mode)
                    z_next = agent.Z
                    u_i_next = (-agent.K @ agent.Z)
                    zu_next = torch.cat((z_next, u_i_next), dim=0)
                    phi = agent.get_phi(zu_curr, zu_next)
                    phi_list.append(phi)

                for i, agent in enumerate(self.agents):
                    g = agent.get_cost(X_cur[i], U[i])
                    agent.update_theta(phi_list[i], g)

                    G[t, i] = g.clone()

                    Q_e = agent.theta @ phi_list[i].T - g
                    Q_val[t, i] = Q_e.clone()
                t = t + 1
            # Policy improvement (ST-Q)
            for i, agent in enumerate(self.agents):
                agent.reconstruct_H_and_update_K()

            system_K = torch.cat([agent.K for agent in self.agents], dim=0)
            k_diff = torch.norm(system_K - self.K_opt).item()
            k_diff_history.append(k_diff)
            print(f"After Q={q}, ||K - K_opt|| = {k_diff}")

            max_theta_diff = 0.0
            for i, agent in enumerate(self.agents):
                diff = torch.norm(agent.theta - old_thetas[i]).item()
                if diff > max_theta_diff:
                    max_theta_diff = diff
            if max_theta_diff < self.epsilon_K:
                converged = True

        self.final_K = torch.cat([agent.K for agent in self.agents], dim=0)
        policy_dict = {'final_K': self.final_K.cpu().numpy()}
        torch.save(policy_dict, policy_save_path)
        print(f"Final policy saved to {policy_save_path}")
        print(f"Final policy is {self.final_K.cpu().numpy()}")

        return [X_traj.cpu().numpy(), G.cpu().numpy(), Q_val.cpu().numpy(), k_diff_history]

    def run(self, X0, T):
        X_traj = torch.zeros((T + 1, self.L), device=device)
        X_traj[0] = torch.tensor(X0, device=device, dtype=torch.float32)

        for i, agent in enumerate(self.agents):
            agent.K = self.final_K[i].clone()

        t = 0
        while t < T:
            U_list = []
            for i, agent in enumerate(self.agents):
                u_i = (-agent.K @ agent.Z)
                U_list.append(u_i)
            U = torch.cat(U_list, dim=0).view(-1, 1)

            X_cur = X_traj[t].view(-1, 1).clone()
            X_next = self.A_global @ X_cur + self.B_global @ U
            X_traj[t + 1] = X_next.view(-1).clone()

            x_observations = X_traj[t + 1].view(-1).clone()
            for i, agent in enumerate(self.agents):
                agent.state_tracking_update(x_observations, mode='noestimate')
            t = t + 1
        return X_traj.cpu().numpy()


def plot_k_diff_evolution(k_diff_history, mode='save'):
    plt.figure(figsize=(10, 6))
    plt.plot(k_diff_history, linewidth=3, label='||K - K_opt||')
    plt.xlabel('Policy Iteration Steps')
    plt.ylabel('||K - K_opt||')
    plt.title('Convergence of K to Optimal K')
    plt.legend()
    plt.grid(True)
    if mode == 'save':
        plt.savefig('PolicyIterationSteps.png')
        plt.close()
    else:
        plt.show()


def plot_X_traj(X_traj, mode='save'):
    plt.figure(figsize=(10, 6))
    plt.title('X trajectory')
    N = X_traj.shape[1]
    for n in range(N):
        plt.subplot(N, 1, n + 1)
        plt.plot(X_traj[:, n], linewidth=3, label=f'Agent {n}')
        plt.ylabel('||X - X0||')
        plt.legend()
        plt.grid(True)
    plt.xlabel('Time (t)')
    if mode == 'save':
        plt.savefig('Xtrajectory.png')
        plt.close()
    else:
        plt.show()


def plot_G(G, mode='save'):
    plt.figure(figsize=(10, 6))
    plt.title('Cost Value G')
    N = G.shape[1]
    for n in range(N):
        plt.subplot(N, 1, n + 1)
        plt.plot(G[:, n], linewidth=3, label=f'Agent {n}')
        plt.ylabel('G')
        plt.legend()
        plt.grid(True)
    plt.xlabel('Time (t)')
    if mode == 'save':
        plt.savefig('G.png')
        plt.close()
    else:
        plt.show()


def plot_Q(Q, mode='save'):
    plt.figure(figsize=(10, 6))
    plt.title('Estimation value Q')
    N = Q.shape[1]
    for n in range(N):
        plt.subplot(N, 1, n + 1)
        plt.plot(Q[:, n], linewidth=3, label=f'Agent {n}')
        plt.ylabel('Q error')
        plt.legend()
        plt.grid(True)
    plt.xlabel('Time (t)')
    if mode == 'save':
        plt.savefig('Q.png')
        plt.close()
    else:
        plt.show()


L = 4  # number of agents
n = 1  # number of outputs
m = 1

learning_rate = 0.001

A_global = np.array([
    [0.2, 0.4, 0.1, 0.01],
    [0.4, 0.2, 0.3, 0.1],
    [0.1, 0.3, 0.3, 0.4],
    [0.2, 0.1, 0.5, 0.3]
])

B_global = np.eye(4)
K_opt = np.array([
    [0.1223193, 0.22787562, 0.07794001, 0.02514448],
    [0.22666936, 0.12788015, 0.18230773, 0.07137375],
    [0.07961969, 0.1869382, 0.19443193, 0.2340778],
    [0.12120244, 0.07423371, 0.2838279, 0.1755628]
])

K_better = np.array([
    [0.9, 0.6, 0.05, 1.],
    [0.6, 0.5, 0.6, 0.2],
    [2., 0.15, 0.8, 1.5],
    [0.15, 0.1, 0.35, 0.5]
])

W = np.array([
    [0.5, 0.5, 0, 0],
    [0.5, 0.3, 0.2, 0],
    [0, 0.2, 0.2, 0.6],
    [0, 0, 0.6, 0.4]
])
row_sums = W.sum(axis=1, keepdims=True)
W = W / row_sums

comm_neighbors = {
    0: [1],
    1: [0, 2],
    2: [1, 3],
    3: [2]
}

X0 = [0.01, 0.01, 0.01, 0.01]

agents = []
for i in range(L):
    agent = Agent(
        agent_id=i,
        L=L,
        n=n,
        m=m,
        neighbors=[],
        comm_neighbors=comm_neighbors[i],
        W=torch.tensor(W, dtype=torch.float32),
        K_initial=K_better[i],
        xi0=X0[i],
        alpha=learning_rate,
        learning_mode='adaptive'  # 将学习模式设置为 'adaptive'
    )
    agents.append(agent)

mas = MultiAgentSystem(agents, A_global, B_global, torch.tensor(W, dtype=torch.float32), K_opt, alpha=0.001,
                       epsilon_K=1e-8)

N = 1000
Q = 5
T = Q * N

print(f"Total number of agents are {L}.")

[X_traj, G, Q_error, K_diff_history] = mas.run_simulation(X0=X0, N=N, Q=Q, T=T, mode='decay',
                                                          policy_save_path='final_policy.pt', tracking_mode='estimate',
                                                          control_mode='no_clamp')
X_traj_final = mas.run(X0=X0, T=T)

plot_k_diff_evolution(K_diff_history, mode='save')
plot_X_traj(X_traj, mode='save')
plot_G(G, mode='save')
plot_Q(Q_error, mode='save')
plot_X_traj(X_traj_final, mode='show')
