import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  

class DynamicGridWorld:
    def __init__(self, size=10, start=(0, 0), goal=(9, 9), 
                 num_dynamic_obstacles=3, obstacle_speed=1,
                 obstacle_cost=2, distance_to_obstacle_threshold=1):
        self.size = size  # 网格大小
        self.start = start  # 起点
        self.goal = goal    # 终点
        
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 上、右、下、左
        self.action_names = ['上', '右', '下', '左']
        
        # 动态障碍物参数
        self.num_dynamic_obstacles = num_dynamic_obstacles  # 动态障碍物数量
        self.obstacle_speed = obstacle_speed  # 障碍物移动速度（每n步移动一次）
        self.obstacles = set()  # 当前障碍物位置集合
        self.obstacle_directions = []  # 每个障碍物的移动方向
        
        self._init_dynamic_obstacles()
        
        # 代价层参数
        self.obstacle_cost = obstacle_cost
        self.distance_to_obstacle_threshold = distance_to_obstacle_threshold
        
        # 初始化价值函数和策略
        self.values = np.zeros((size, size))
        self.values[goal] = 100
        
        self.policy = np.random.choice(len(self.actions), size=(size, size))
        for x in range(size):
            for y in range(size):
                state = (x, y)
                if self.is_terminal(state) or self.is_obstacle(state):
                    self.policy[state] = -1
        
        # 算法参数
        self.gamma = 0.9
        self.theta = 0.01
        self.distance_reward_factor = 2
        
        # 存储迭代过程和动态变化
        self.iteration_history = []
        self.cost_layer = self._compute_cost_layer()
        self.time_step = 0  # 时间步，用于控制障碍物移动
        
    def is_terminal(self, state):
        """判断是否为终止状态（终点）"""
        return state == self.goal
    
    def is_obstacle(self, state):
        """判断是否为障碍物"""
        return state in self.obstacles
    
    def _init_dynamic_obstacles(self):
        """初始化动态障碍物位置和初始方向"""
        # 确保障碍物不与起点、终点重叠
        occupied = {self.start, self.goal}
        
        for _ in range(self.num_dynamic_obstacles):
            while True:
                x = np.random.randint(0, self.size)
                y = np.random.randint(0, self.size)
                if (x, y) not in occupied:
                    self.obstacles.add((x, y))
                    occupied.add((x, y))
                    # 随机初始方向（此时self.actions已定义）
                    dir_idx = np.random.choice(len(self.actions))
                    self.obstacle_directions.append(self.actions[dir_idx])
                    break
    
    def _move_dynamic_obstacles(self):
        """移动动态障碍物（根据速度控制移动频率）"""
        self.time_step += 1
        if self.time_step % self.obstacle_speed != 0:
            return  # 不到移动时机
        
        new_obstacles = set()
        new_directions = []
        occupied = {self.start, self.goal}
        
        for i, (x, y) in enumerate(self.obstacles):
            # 当前移动方向
            dx, dy = self.obstacle_directions[i]
            
            # 计算新位置
            new_x = x + dx
            new_y = y + dy
            
            # 检查新位置是否合法（不越界、不碰撞）
            valid = True
            if new_x < 0 or new_x >= self.size or new_y < 0 or new_y >= self.size:
                valid = False
            if (new_x, new_y) in occupied:
                valid = False
            
            # 如果不合法，随机选择新方向
            if not valid:
                possible_dirs = []
                for dir in self.actions:
                    nx = x + dir[0]
                    ny = y + dir[1]
                    if 0 <= nx < self.size and 0 <= ny < self.size and (nx, ny) not in occupied:
                        possible_dirs.append(dir)
                # 如果有可用方向，随机选择；否则保持原地
                if possible_dirs:
                    dx, dy = possible_dirs[np.random.choice(len(possible_dirs))]
                    new_x = x + dx
                    new_y = y + dy
                else:
                    new_x, new_y = x, y  # 无法移动，保持原地
            
            # 更新位置和方向
            new_obstacles.add((new_x, new_y))
            new_directions.append((dx, dy))
            occupied.add((new_x, new_y))
        
        # 更新障碍物状态
        self.obstacles = new_obstacles
        self.obstacle_directions = new_directions
        # 重新计算代价层
        self.cost_layer = self._compute_cost_layer()
    
    def _compute_cost_layer(self):
        """计算当前障碍物分布下的代价层"""
        cost_layer = np.zeros((self.size, self.size), dtype=bool)
        for x in range(self.size):
            for y in range(self.size):
                state = (x, y)
                if self.is_obstacle(state) or self.is_terminal(state):
                    continue
                # 检查是否在任何障碍物的阈值范围内
                for (ox, oy) in self.obstacles:
                    distance = np.sqrt((x - ox)**2 + (y - oy)** 2)
                    if distance <= self.distance_to_obstacle_threshold:
                        cost_layer[x, y] = True
                        break
        return cost_layer
    
    def in_cost_layer(self, state):
        """判断状态是否在代价层内"""
        x, y = state
        return self.cost_layer[x, y]
    
    def euclidean_distance(self, state1, state2):
        """计算欧氏距离"""
        return np.sqrt((state1[0] - state2[0])**2 + (state1[1] - state2[1])** 2)
    
    def step(self, state, action):
        """执行动作，返回下一个状态和奖励（考虑动态障碍物）"""
        x, y = state
        dx, dy = action
        
        new_x = x + dx
        new_y = y + dy
        current_distance = self.euclidean_distance(state, self.goal)
        
        # 边界检查
        if new_x < 0 or new_x >= self.size or new_y < 0 or new_y >= self.size:
            new_x, new_y = x, y
            reward = -10
        elif (new_x, new_y) in self.obstacles:
            new_x, new_y = x, y
            reward = -10
        elif (new_x, new_y) == self.goal:
            reward = 100
        else:
            reward = -1
            # 距离奖励
            new_distance = self.euclidean_distance((new_x, new_y), self.goal)
            distance_change = current_distance - new_distance
            reward += self.distance_reward_factor * distance_change
            # 代价层惩罚
            if self.in_cost_layer((new_x, new_y)):
                reward -= self.obstacle_cost
        
        return (new_x, new_y), reward
    
    def policy_evaluation(self, max_iter=100):
        """策略评估"""
        for _ in range(max_iter):
            delta = 0
            new_values = np.copy(self.values)
            
            for x in range(self.size):
                for y in range(self.size):
                    state = (x, y)
                    if self.is_terminal(state) or self.is_obstacle(state):
                        continue
                    
                    action_idx = self.policy[state]
                    action = self.actions[action_idx]
                    next_state, reward = self.step(state, action)
                    
                    new_value = reward + self.gamma * self.values[next_state]
                    delta = max(delta, np.abs(new_value - self.values[state]))
                    new_values[state] = new_value
            
            self.values = new_values
            if delta < self.theta:
                break
    
    def policy_improvement(self):
        """策略改进"""
        policy_stable = True
        for x in range(self.size):
            for y in range(self.size):
                state = (x, y)
                if self.is_terminal(state) or self.is_obstacle(state):
                    continue
                
                old_action_idx = self.policy[state]
                q_values = []
                for action in self.actions:
                    next_state, reward = self.step(state, action)
                    q_value = reward + self.gamma * self.values[next_state]
                    q_values.append(q_value)
                
                new_action_idx = np.argmax(q_values)
                self.policy[state] = new_action_idx
                
                if old_action_idx != new_action_idx:
                    policy_stable = False
        return policy_stable
    
    def policy_iteration(self, max_iter=10):
        """策略迭代（简化版，用于动态环境快速适应）"""
        for i in range(max_iter):
            self.policy_evaluation()
            if self.policy_improvement():
                break
        return self.policy
    
    def get_optimal_path(self, start=None):
        """获取当前策略下的最优路径"""
        if start is None:
            start = self.start
        
        path = [start]
        current_state = start
        steps = 0
        max_steps = self.size * self.size
        
        while not self.is_terminal(current_state) and steps < max_steps:
            action_idx = self.policy[current_state]
            if action_idx == -1:
                break
                
            action = self.actions[action_idx]
            next_state, _ = self.step(current_state, action)
            path.append(next_state)
            current_state = next_state
            steps += 1
        
        return path
    
    def update_environment(self):
        """更新环境状态（移动障碍物并重新规划）"""
        self._move_dynamic_obstacles()  # 移动障碍物
        self.policy_iteration()  # 重新规划策略
        return self.get_optimal_path()  # 返回新路径
    
    def visualize_dynamic_process(self, total_steps=50):
        """可视化动态障碍物环境下的路径规划过程"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 颜色映射
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)
        
        # 初始数据
        im = ax.imshow(self.values, cmap=cmap, interpolation='nearest', vmin=0, vmax=100)
        path = self.get_optimal_path()
        path_line, = ax.plot([], [], 'g-', linewidth=2, marker='o', markersize=5, markerfacecolor='green')
        
        # 标记元素
        obstacle_patches = [plt.Rectangle((0, 0), 1, 1, fill=True, color='black', alpha=0.8) 
                           for _ in range(self.num_dynamic_obstacles)]
        for p in obstacle_patches:
            ax.add_patch(p)
        
        cost_layer_patches = []
        for x in range(self.size):
            for y in range(self.size):
                if self.in_cost_layer((x, y)):
                    p = plt.Rectangle((y-0.5, x-0.5), 1, 1, fill=True, color='yellow', alpha=0.3)
                    cost_layer_patches.append(p)
                    ax.add_patch(p)
        
        ax.text(self.start[1], self.start[0], '起点', ha='center', va='center', 
                color='white', fontsize=12, fontweight='bold')
        ax.text(self.goal[1], self.goal[0], '终点', ha='center', va='center', 
                color='yellow', fontsize=12, fontweight='bold')
        
        ax.set_xticks(np.arange(-0.5, self.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.size, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        cbar = fig.colorbar(im)
        cbar.set_label('状态价值')
        title = ax.set_title(f'动态障碍物环境 - 时间步: 0')
        
        # 更新函数
        def update(frame):
            # 清除上一帧的代价层标记
            for p in cost_layer_patches:
                p.remove()
            
            # 更新环境
            new_path = self.update_environment()
            
            # 更新价值函数显示
            im.set_data(self.values)
            
            # 更新障碍物位置
            obstacles = list(self.obstacles)
            for i, p in enumerate(obstacle_patches):
                if i < len(obstacles):
                    x, y = obstacles[i]
                    p.set_xy((y-0.5, x-0.5))
            
            # 重新绘制代价层
            new_cost_patches = []
            for x in range(self.size):
                for y in range(self.size):
                    if self.in_cost_layer((x, y)):
                        p = plt.Rectangle((y-0.5, x-0.5), 1, 1, fill=True, color='yellow', alpha=0.3)
                        new_cost_patches.append(p)
                        ax.add_patch(p)
            cost_layer_patches[:] = new_cost_patches  # 更新引用
            
            if new_path:
                path_x = [p[1] for p in new_path]
                path_y = [p[0] for p in new_path]
                path_line.set_data(path_x, path_y)
            
            title.set_text(f'动态障碍物环境 - 时间步: {frame+1}')
            return [im, path_line] + obstacle_patches + cost_layer_patches
        
        ani = animation.FuncAnimation(
            fig, update, frames=total_steps,
            interval=50, blit=True
        )
        
        plt.tight_layout()
        return ani

def main():
    grid = DynamicGridWorld(
        size=20,
        start=(0, 0),
        goal=(19, 19),
        num_dynamic_obstacles=7,  # 3个动态障碍物
        obstacle_speed=1,  # 每2步移动一次
        obstacle_cost=3,   # 代价层惩罚
        distance_to_obstacle_threshold=1  # 相邻格子为代价层
    )
    
    # 初始策略迭代
    grid.policy_iteration()
    
    # 可视化动态过程（总步数50）
    ani = grid.visualize_dynamic_process(total_steps=50)
    
    plt.show()

if __name__ == "__main__":
    main()