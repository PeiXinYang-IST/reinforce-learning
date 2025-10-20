import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

class MCEpsilonGreedyMaze:
    """基于ε-greedy蒙特卡洛方法的网格世界路径搜索（静态障碍物）"""
    def __init__(self, size=10, goal=(9, 9),
                 num_static_obstacles=2,  # 静态障碍物数量
                 obstacle_cost=2, distance_threshold=1.5,
                 epsilon=0.1):
        self.size = size
        self.goal = goal  # 终点固定
        self.start_for_path = (0, 0)  # 路径生成的固定起点
        
        # 动作定义（上、右、下、左）
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.num_actions = len(self.actions)
        
        # 静态障碍物参数
        self.num_static_obstacles = num_static_obstacles
        self.obstacles = set()
        self._init_static_obstacles()
        
        # 代价层参数
        self.obstacle_cost = obstacle_cost
        self.distance_threshold = distance_threshold
        self.cost_layer = self._compute_cost_layer()
        
        # MC核心参数（使用ε-greedy）
        self.Q = defaultdict(float)  # 状态-动作价值函数 Q(s,a)
        self.returns = defaultdict(list)  # 存储每个(s,a)的回报
        self.gamma = 0.9  # 折扣因子
        self.distance_reward_factor = 10  # 距离奖励因子
        self.epsilon = epsilon  # ε-greedy参数
        
        # 环境状态
        self.episodes = 0  # 记录已完成的轨迹数
        self.q_history = []  # 记录Q值变化（背景核心）
        
        # 初始化学习历史
        self._record_q_values()  # 初始记录一次Q值

    # ---------------------- 基础环境逻辑 ----------------------
    def is_terminal(self, state):
        return state == self.goal
    
    def is_obstacle(self, state):
        return state in self.obstacles
    
    def _init_static_obstacles(self):
        """初始化静态障碍物（固定位置）"""
        occupied = {self.goal, self.start_for_path}  # 避免起点和终点
        for _ in range(self.num_static_obstacles):
            while True:
                x = np.random.randint(self.size)
                y = np.random.randint(self.size)
                if (x, y) not in occupied:
                    self.obstacles.add((x, y))
                    occupied.add((x, y))
                    break
    
    def _compute_cost_layer(self):
        """计算静态障碍物的代价层（固定）"""
        cost_layer = np.zeros((self.size, self.size), dtype=bool)
        for x in range(self.size):
            for y in range(self.size):
                state = (x, y)
                if self.is_obstacle(state) or self.is_terminal(state):
                    continue
                for (ox, oy) in self.obstacles:
                    dist = np.sqrt((x-ox)**2 + (y-oy)**2)
                    if dist <= self.distance_threshold:
                        cost_layer[x, y] = True
                        break
        return cost_layer
    
    def in_cost_layer(self, state):
        x, y = state
        return self.cost_layer[x, y]
    
    def euclidean_distance(self, s1, s2):
        return np.sqrt((s1[0]-s2[0])**2 + (s1[1]-s2[1])**2)
    
    def step(self, state, action):
        x, y = state
        dx, dy = action
        new_x, new_y = x + dx, y + dy
        
        current_dist = self.euclidean_distance(state, self.goal)
        
        if not (0 <= new_x < self.size and 0 <= new_y < self.size):
            new_x, new_y = x, y
            reward = -10
        elif self.is_obstacle((new_x, new_y)):
            new_x, new_y = x, y
            reward = -10
        elif self.is_terminal((new_x, new_y)):
            reward = 100
        else:
            reward = -1
            new_dist = self.euclidean_distance((new_x, new_y), self.goal)
            reward += self.distance_reward_factor * (current_dist - new_dist)
            if self.in_cost_layer((new_x, new_y)):
                reward -= self.obstacle_cost
        
        return (new_x, new_y), reward

    # ---------------------- MC核心逻辑（ε-greedy）----------------------
    def choose_epsilon_greedy_action(self, state):
        """使用ε-greedy策略选择动作"""
        if np.random.rand() < self.epsilon:
            # 随机选择动作（探索）
            return np.random.choice(self.num_actions)
        else:
            # 选择当前最优动作（利用）
            q_values = [self.Q[(state, a)] for a in range(self.num_actions)]
            return np.argmax(q_values)
    
    def generate_episode_with_epsilon_greedy(self):
        """生成一条轨迹（使用ε-greedy策略）"""
        episode = []
        
        # 确保起点可用（避免障碍物阻挡）
        start_state = self.start_for_path
        if self.is_obstacle(start_state) or self.is_terminal(start_state):
            # 如果起点被障碍物阻挡，寻找最近的可用点
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_state = (start_state[0] + dx, start_state[1] + dy)
                if 0 <= new_state[0] < self.size and 0 <= new_state[1] < self.size and not self.is_obstacle(new_state) and not self.is_terminal(new_state):
                    start_state = new_state
                    break
        
        state = start_state
        steps = 0
        max_steps = self.size * self.size * 2
        
        while not self.is_terminal(state) and steps < max_steps:
            action = self.choose_epsilon_greedy_action(state)
            next_state, reward = self.step(state, self.actions[action])
            episode.append((state, action, reward))
            state = next_state
            steps += 1
        
        return episode

    def first_visit_mc_control_with_epsilon_greedy(self, num_episodes=500):
        for _ in range(num_episodes):
            episode = self.generate_episode_with_epsilon_greedy()
            self.episodes += 1
            
            # 反向计算回报，更新Q值
            states_actions_in_episode = [(s, a) for (s, a, _) in episode]
            G = 0
            for t in reversed(range(len(episode))):
                s, a, r = episode[t]
                G = self.gamma * G + r
                if (s, a) not in states_actions_in_episode[:t]:
                    self.returns[(s, a)].append(G)
                    self.Q[(s, a)] = np.mean(self.returns[(s, a)])
            
            # 每10个episode记录一次Q值
            if self.episodes % 10 == 0:
                self._record_q_values()

    # ---------------------- 记录背景状态 ----------------------
    def _record_q_values(self):
        v = np.zeros((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                s = (x, y)
                if self.is_terminal(s):
                    v[x, y] = 100
                elif self.is_obstacle(s):
                    v[x, y] = 0
                else:
                    v[x, y] = max([self.Q[(s, a)] for a in range(self.num_actions)])
        self.q_history.append(v)

    # ---------------------- 实时生成最优路径 ----------------------
    def get_latest_optimal_path(self):
        path = [self.start_for_path]
        current = self.start_for_path
        steps = 0
        max_steps = self.size * self.size
        
        while not self.is_terminal(current) and steps < max_steps:
            if self.is_obstacle(current) or current not in [(x,y) for x in range(self.size) for y in range(self.size)]:
                break
            
            q_values = [self.Q[(current, a)] for a in range(self.num_actions)]
            best_action_idx = np.argmax(q_values)
            best_action = self.actions[best_action_idx]
            
            next_state, _ = self.step(current, best_action)
            
            if len(path) >= 2 and next_state == path[-2]:
                sorted_actions = np.argsort(q_values)[::-1]
                for idx in sorted_actions:
                    if idx != best_action_idx:
                        next_state, _ = self.step(current, self.actions[idx])
                        if next_state != path[-2]:
                            break
            
            path.append(next_state)
            current = next_state
            steps += 1
        
        path = [p for p in path if not self.is_obstacle(p)]
        return path if len(path) > 1 else [self.start_for_path]

    # ---------------------- 可视化：静态障碍物背景 ----------------------
    def visualize_learning_with_static_obstacles(self, total_episodes=500):
        self.first_visit_mc_control_with_epsilon_greedy(total_episodes)
        
        if not self.q_history:
            print("没有学习历史记录，无法可视化")
            return
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
        cmap = LinearSegmentedColormap.from_list('custom_bg', colors, N=100)
        
        im_bg = ax.imshow(self.q_history[0], cmap=cmap, interpolation='nearest', vmin=0, vmax=100)
        
        # 绘制静态障碍物
        obstacle_patches = []
        for (x, y) in self.obstacles:
            patch = plt.Rectangle((y-0.5, x-0.5), 1, 1, color='black', alpha=0.9)
            obstacle_patches.append(patch)
            ax.add_patch(patch)
        
        # 绘制静态代价层
        cost_patches = []
        for x in range(self.size):
            for y in range(self.size):
                if self.cost_layer[x, y] and (x,y) not in self.obstacles and (x,y) != self.goal:
                    patch = plt.Rectangle((y-0.5, x-0.5), 1, 1, color='yellow', alpha=0.4)
                    cost_patches.append(patch)
                    ax.add_patch(patch)
        
        initial_path = self.get_latest_optimal_path()
        path_x = [p[1] for p in initial_path]
        path_y = [p[0] for p in initial_path]
        path_line, = ax.plot(path_x, path_y, 'g-', linewidth=3, marker='o', markersize=6, markerfacecolor='green')
        
        ax.text(self.start_for_path[1], self.start_for_path[0], '起点', ha='center', va='center', 
                color='white', fontsize=14, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='blue'))
        ax.text(self.goal[1], self.goal[0], '终点', ha='center', va='center', 
                color='black', fontsize=14, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
        
        ax.set_xticks(np.arange(-0.5, self.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.size, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        
        cbar = fig.colorbar(im_bg, ax=ax, shrink=0.8)
        cbar.set_label('状态价值（越高越接近最优路径）', fontsize=12)
        
        # 初始化标题
        title = ax.set_title(f'ε-greedy MC学习 - 已完成 {total_episodes} 个轨迹 | 静态障碍物环境', fontsize=14)
        
        def update(frame):
            current_frame = min(frame, len(self.q_history)-1)
            im_bg.set_data(self.q_history[current_frame])
            
            latest_path = self.get_latest_optimal_path()
            if latest_path:
                path_x = [p[1] for p in latest_path]
                path_y = [p[0] for p in latest_path]
                path_line.set_data(path_x, path_y)
            
            # 更新标题
            completed_episodes = (current_frame + 1) * 10
            title.set_text(f'ε-greedy MC学习 - 已完成 {completed_episodes} 个轨迹 | 静态障碍物环境')
            
            return [im_bg, path_line]
        
        ani = animation.FuncAnimation(
            fig, update,
            frames=len(self.q_history),
            interval=200,
            blit=True,
            repeat=True
        )
        
        plt.tight_layout()
        return ani

def main():
    # 使用ε-greedy方法，设置ε=0.1
    mc_epsilon_maze = MCEpsilonGreedyMaze(
        size=15,
        goal=(14, 14),
        num_static_obstacles=0,  # 静态障碍物数量
        obstacle_cost=3,
        distance_threshold=1.5,
        epsilon=0.2  # ε-greedy参数
    )
    
    ani = mc_epsilon_maze.visualize_learning_with_static_obstacles(total_episodes=1000)
    plt.show()

if __name__ == "__main__":
    main()