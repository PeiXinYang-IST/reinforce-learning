import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  

class GridWorld:
    """网格世界环境"""
    def __init__(self, size=10, start=(0, 0), goal=(9, 9), obstacles=None):
        self.size = size  # 网格大小
        self.start = start  # 起点
        self.goal = goal    # 终点
        
        # 如果没有指定障碍物，则随机生成一些
        if obstacles is None:
            self.obstacles = set()
            # 确保起点和终点不是障碍物
            while len(self.obstacles) < size:
                x = np.random.randint(0, size)
                y = np.random.randint(0, size)
                if (x, y) != start and (x, y) != goal:
                    self.obstacles.add((x, y))
        else:
            self.obstacles = set(obstacles)
        
        # 可能的动作: 上、右、下、左
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_names = ['上', '右', '下', '左']
        
        # 初始化价值函数
        self.values = np.zeros((size, size))
        # 终点价值设为100
        self.values[goal] = 1
        
        # 折扣因子
        self.gamma = 0.9
        # 收敛阈值
        self.theta = 0.01
        # 距离奖励因子（控制距离对奖励的影响程度）
        self.distance_reward_factor = 0.1
        
        # 存储迭代过程，用于可视化
        self.iteration_history = []
        
    def is_terminal(self, state):
        """判断是否为终止状态（终点）"""
        return state == self.goal
    
    def is_obstacle(self, state):
        """判断是否为障碍物"""
        return state in self.obstacles
    
    def euclidean_distance(self, state1, state2):
        """计算两个状态之间的欧氏距离"""
        return np.sqrt((state1[0] - state2[0])**2 + (state1[1] - state2[1])** 2)
    
    def step(self, state, action):
        """执行动作，返回下一个状态和奖励，包含基于距离的奖励机制"""
        x, y = state
        dx, dy = action
        
        # 计算新位置
        new_x = x + dx
        new_y = y + dy
        
        # 计算当前状态到目标的距离
        current_distance = self.euclidean_distance(state, self.goal)
        
        # 检查是否超出边界
        if new_x < 0 or new_x >= self.size or new_y < 0 or new_y >= self.size:
            # 撞墙，停留在原地
            new_x, new_y = x, y
            reward = -10  # 撞墙惩罚
        elif (new_x, new_y) in self.obstacles:
            # 撞到障碍物，停留在原地
            new_x, new_y = x, y
            reward = -10  # 撞障碍物惩罚
        elif (new_x, new_y) == self.goal:
            # 到达终点
            reward = 100  # 奖励
        else:
            # 正常移动基础奖励
            reward = -1  # 每步轻微惩罚，鼓励尽快到达
            
            # 计算新状态到目标的距离
            new_distance = self.euclidean_distance((new_x, new_y), self.goal)
            
            # 根据距离变化给予额外奖励或惩罚
            distance_change = current_distance - new_distance
            reward += self.distance_reward_factor * distance_change
        
        return (new_x, new_y), reward
    
    def value_iteration(self, max_iterations=100):
        """价值迭代算法"""
        self.iteration_history = []  # 重置历史记录
        self.iteration_history.append(np.copy(self.values))  # 记录初始值
        
        for i in range(max_iterations):
            delta = 0
            new_values = np.copy(self.values)
            
            # 对每个状态更新价值
            for x in range(self.size):
                for y in range(self.size):
                    state = (x, y)
                    
                    # 终点和障碍物不更新
                    if self.is_terminal(state) or self.is_obstacle(state):
                        continue
                    
                    # 计算所有可能动作的Q值
                    q_values = []
                    for action in self.actions:
                        next_state, reward = self.step(state, action)
                        q_value = reward + self.gamma * self.values[next_state]
                        q_values.append(q_value)
                    
                    # 取最大Q值作为新的价值
                    new_value = max(q_values)
                    delta = max(delta, np.abs(new_value - self.values[state]))
                    new_values[state] = new_value
            
            # 更新价值函数
            self.values = new_values
            self.iteration_history.append(np.copy(self.values))
            
            # 检查是否收敛
            if delta < self.theta:
                print(f"价值迭代在第 {i+1} 次迭代收敛")
                break
        
        return self.values
    
    def get_optimal_policy(self):
        """根据价值函数获取最优策略"""
        policy = np.zeros((self.size, self.size), dtype=int)  # 存储每个状态的最优动作索引
        
        for x in range(self.size):
            for y in range(self.size):
                state = (x, y)
                
                # 终点和障碍物没有策略
                if self.is_terminal(state) or self.is_obstacle(state):
                    policy[state] = -1
                    continue
                
                # 计算所有可能动作的Q值
                q_values = []
                for action in self.actions:
                    next_state, reward = self.step(state, action)
                    q_value = reward + self.gamma * self.values[next_state]
                    q_values.append(q_value)
                
                # 选择Q值最大的动作
                policy[state] = np.argmax(q_values)
        
        return policy
    
    def get_optimal_path(self, start=None):
        """根据最优策略获取从起点到终点的路径"""
        if start is None:
            start = self.start
        
        path = [start]
        current_state = start
        steps = 0
        max_steps = self.size * self.size * 2  # 防止无限循环
        
        while not self.is_terminal(current_state) and steps < max_steps:
            policy = self.get_optimal_policy()
            action_idx = policy[current_state]
            
            # 如果没有有效动作（不应该发生）
            if action_idx == -1:
                break
                
            action = self.actions[action_idx]
            next_state, _ = self.step(current_state, action)
            path.append(next_state)
            current_state = next_state
            steps += 1
        
        return path
    
    def visualize_iteration(self):
        """可视化价值迭代过程"""
        if not self.iteration_history:
            print("请先运行价值迭代算法")
            return
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 创建自定义颜色映射
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # 蓝色(低价值)到红色(高价值)
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)
        
        # 初始图像
        im = ax.imshow(self.iteration_history[0], cmap=cmap, interpolation='nearest', vmin=0, vmax=100)
        
        # 标记障碍物
        for (x, y) in self.obstacles:
            ax.add_patch(plt.Rectangle((y-0.5, x-0.5), 1, 1, fill=True, color='black', alpha=0.5))
        
        # 标记起点和终点
        ax.text(self.start[1], self.start[0], '起点', ha='center', va='center', color='white', fontsize=12, fontweight='bold')
        ax.text(self.goal[1], self.goal[0], '终点', ha='center', va='center', color='yellow', fontsize=12, fontweight='bold')
        
        ax.set_xticks(np.arange(-0.5, self.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.size, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        cbar = fig.colorbar(im)
        cbar.set_label('状态价值')
        
        title = ax.set_title(f'价值迭代 - 迭代次数: 0')
        
        # 更新函数
        def update(frame):
            im.set_data(self.iteration_history[frame])
            title.set_text(f'价值迭代 - 迭代次数: {frame}')
            return im, title
        
        ani = animation.FuncAnimation(
            fig, update, frames=len(self.iteration_history),
            interval=500, blit=True
        )
        
        plt.tight_layout()
        return ani
    
    def visualize_policy_and_path(self):
        """可视化最优策略和路径"""
        policy = self.get_optimal_policy()
        path = self.get_optimal_path()
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 显示价值函数
        im = ax.imshow(self.values, cmap='coolwarm', interpolation='nearest', vmin=0, vmax=100)
        
        # 标记障碍物
        for (x, y) in self.obstacles:
            ax.add_patch(plt.Rectangle((y-0.5, x-0.5), 1, 1, fill=True, color='black', alpha=0.5))
        
        # 标记起点和终点
        ax.text(self.start[1], self.start[0], '起点', ha='center', va='center', color='white', fontsize=12, fontweight='bold')
        ax.text(self.goal[1], self.goal[0], '终点', ha='center', va='center', color='yellow', fontsize=12, fontweight='bold')
        
        # 绘制策略箭头
        arrow_length = 0.3
        for x in range(self.size):
            for y in range(self.size):
                state = (x, y)
                if self.is_terminal(state) or self.is_obstacle(state):
                    continue
                
                action_idx = policy[state]
                dx, dy = self.actions[action_idx]
                
                ax.arrow(
                    y, x, dy * arrow_length, dx * arrow_length,
                    head_width=0.1, head_length=0.1, fc='black', ec='black'
                )
        
        if path:
            path_x = [p[1] for p in path]
            path_y = [p[0] for p in path]
            ax.plot(path_x, path_y, 'g-', linewidth=2, marker='o', markersize=5, markerfacecolor='green')
        
        ax.set_xticks(np.arange(-0.5, self.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.size, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        cbar = fig.colorbar(im)
        cbar.set_label('状态价值')
        
        ax.set_title('最优策略与路径（包含距离奖励机制）')
        
        plt.tight_layout()
        return fig

def main():
    # 创建10x10的网格世界，起点(0,0)，终点(9,9)
    # 可以自定义障碍物，例如：obstacles={(2,2), (2,3), (3,2), (4,4), (5,5), (6,6), (7,7)}
    grid = GridWorld(size=15, start=(0, 0), goal=(12, 9))
    
    # 运行价值迭代
    grid.value_iteration()
    
    # 可视化迭代过程
    ani = grid.visualize_iteration()
    
    # 可视化最优策略和路径
    fig = grid.visualize_policy_and_path()
    
    plt.show()

if __name__ == "__main__":
    main()