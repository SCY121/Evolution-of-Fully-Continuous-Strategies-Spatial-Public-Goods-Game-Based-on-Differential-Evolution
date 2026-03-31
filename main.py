import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
from numba import njit, prange
import matplotlib.colors as mcolors
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False



@njit(cache=True, parallel=True)
def update_ipr_de_sweep_jit(current_I_grid, current_R_grid, current_payoffs, current_reputations,
                            grid_size, de_F, de_CR, de_variant_code, K_param, r_param,
                            reward_cost_factor_param, reward_multiplier_param,
                            game_group_offsets, game_group_size,
                            de_pool_offsets, de_pool_size,
                            omega_I, omega_R, lambda_param,
                            alpha_U_param, beta_param, max_reputation_value_param):
    """遍历所有个体，执行 DE 策略更新。"""
    new_investment_grid = current_I_grid.copy()
    new_reward_grid = current_R_grid.copy()

    for i in prange(grid_size):
        for j in range(grid_size):
            new_strategy_vec = perform_single_de_step_jit(
                i, j, current_I_grid, current_R_grid, current_payoffs, current_reputations,
                grid_size, de_F, de_CR, de_variant_code, K_param, r_param,
                reward_cost_factor_param, reward_multiplier_param,
                game_group_offsets, game_group_size,
                de_pool_offsets, de_pool_size,
                omega_I, omega_R, lambda_param,
                alpha_U_param, beta_param, max_reputation_value_param
            )

            new_investment_grid[i, j] = new_strategy_vec[0]
            new_reward_grid[i, j] = new_strategy_vec[1]

    return new_investment_grid, new_reward_grid


def format_param_value(value):
    if isinstance(value, float):
        return f"{value:.2f}".replace('.', 'p')
    elif isinstance(value, int):
        return str(value)
    else:
        return str(value)

def get_neighbor_type_string(neighbor_type_int):
    if neighbor_type_int == 1:
        return "一阶邻居"
    elif neighbor_type_int == 2:
        return "二阶邻居"
    else:
        return f"未知邻居类型({neighbor_type_int})"


class SPGG_ReputationDE:
    """
    基于差分进化的空间公共品博弈模拟，引入声誉机制和双目标适应度。
    """
    def __init__(self, r=3, K=0.1,
                 grid_size=50, max_iterations=1000,
                 de_F=0.5, de_CR=0.9,
                 de_variant='best/1',
                 de_neighbor_type=1,
                 reward_cost_factor=1.0,
                 reward_multiplier=1.0,
                 beta=0.5,
                 omega_I=1.0, omega_R=0.5,
                 alpha_U=0.5,
                 lambda_param=0.5,
                 visualize_interval=50,
                 output_folder='simulation_results/default_run',
                 seed=None):

        if seed is not None:
            np.random.seed(seed)
        else:
            seed = np.random.randint(0, 2 ** 31)
            np.random.seed(seed)
        self.seed = seed

        self.r = r
        self.K = K
        self.grid_size = grid_size
        self.max_iterations = max_iterations
        self.de_F = de_F
        self.de_CR = de_CR

        self.de_strategy_map = {
            'best/1': 1,
            'best/2': 2
        }
        if de_variant not in self.de_strategy_map:
            raise ValueError(f"不支持的 DE 变种: {de_variant}. 可选: {list(self.de_strategy_map.keys())}")
        self.de_variant_name = de_variant
        self.de_variant_code = self.de_strategy_map[de_variant]

        self.game_group_offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        self.game_group_size = len(self.game_group_offsets)

        self.de_neighbor_type = de_neighbor_type
        if self.de_neighbor_type == 1:
            self.de_pool_offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        elif self.de_neighbor_type == 2:
            self.de_pool_offsets = []
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if abs(dr) + abs(dc) <= 2:
                        self.de_pool_offsets.append((dr, dc))
        else:
            raise ValueError(f"不支持的 DE 邻居类型: {de_neighbor_type}. 可选: 1 (一阶邻居), 2 (二阶邻居)")
        self.de_pool_size = len(self.de_pool_offsets)


        self.reward_cost_factor = reward_cost_factor
        self.reward_multiplier = reward_multiplier
        self.beta = beta

        self.omega_I = omega_I
        self.omega_R = omega_R
        self.alpha_U = alpha_U
        self.lambda_param = lambda_param

        self.visualize_interval = visualize_interval
        self.output_folder = output_folder

    def calculate_payoffs(self, current_I_grid, current_R_grid, current_U_grid):
        return calculate_all_payoffs_jit(
            current_I_grid, current_R_grid, current_U_grid,
            self.grid_size, self.game_group_offsets, self.game_group_size,
            self.r, self.reward_cost_factor, self.reward_multiplier,
            self.beta, self.max_possible_reputation
        )

   
    def visualize(self, iteration, avg_I, avg_R, avg_payoff, avg_reputation, output_folder):
        """可视化当前网格状态，包括投资、奖励意愿和声誉分布。"""
        fig = plt.figure(figsize=(18, 6))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 1])

        ax1 = plt.subplot(gs[0])
        im1 = ax1.imshow(self.investment_grid, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
        ax1.set_title(f'迭代次数: {iteration}\n投资分布 (平均 I: {avg_I:.3f})')
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = plt.subplot(gs[1])
        im2 = ax2.imshow(self.reward_grid, cmap='Greens', vmin=0, vmax=1, interpolation='nearest')
        ax2.set_title(f'奖励意愿分布 (平均 R: {avg_R:.3f})')
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = plt.subplot(gs[2])
        im3 = ax3.imshow(self.reputation_grid, cmap='YlOrRd', vmin=0, vmax=self.max_possible_reputation,
                         interpolation='nearest')
        ax3.set_title(f'声誉分布 (平均声誉: {avg_reputation:.3f})')
        ax3.set_xticks([])
        ax3.set_yticks([])
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        main_title_params = (
            f"r={self.r}, K(费米)={self.K}, "
            f"奖励成本={self.reward_cost_factor}, 奖励乘数={self.reward_multiplier}, "
            f"声誉权重(I:{self.omega_I}, R:{self.omega_R}), β={self.beta}, α_U={self.alpha_U}, λ={self.lambda_param}, "
            f"DE变种={self.de_variant_name}, F={self.de_F}, CR={self.de_CR}, "
            f"DE邻居={get_neighbor_type_string(self.de_neighbor_type)} (池大小={self.de_pool_size}), "
            f"博弈小组=冯·诺依曼 (组大小={self.game_group_size}), "
            f"奖励分配=I*U'比例 (池∝avgI*sumR), Seed={self.seed}"
        )
        plt.suptitle(f"网格状态快照 - {main_title_params}", fontsize=14, y=1.02)

        plt.tight_layout(rect=[0, 0, 1, 0.98])

        filename = os.path.join(output_folder, f'iteration_{iteration:06d}.png')
        plt.savefig(filename, bbox_inches='tight')

        plt.show()

        plt.close(fig)


if __name__ == "__main__":

    simulation_params = {
        'r': 3,
        'K': 0.1,
        'grid_size': 50,
        'max_iterations': 2000,
        'de_F': 0.3,
        'de_CR': 0.7,
        'de_variant': 'best/2',
        'de_neighbor_type': 1,
        'reward_cost_factor': 0.3,
        'reward_multiplier': 1,
        'beta': 0.3,
        'omega_I': 1.0,
        'omega_R': 0.5,
        'alpha_U': 0.5,
        'lambda_param': 0.7,
        'visualize_interval': 100,
        'seed': None,
    }

    de_neighbor_dir_name = f"{simulation_params['de_neighbor_type']}order_neighbor"
    base_output_directory = f'results/{de_neighbor_dir_name}'

    game = SPGG_ReputationDE(**simulation_params)

    r_str = format_param_value(game.r)
    iterations_str = str(game.max_iterations)
    grid_size_str = str(game.grid_size)
    de_variant_str = game.de_variant_name.replace('/', '-')
    omega_I_str = format_param_value(game.omega_I)
    omega_R_str = format_param_value(game.omega_R)
    beta_str = format_param_value(game.beta)
    alpha_U_str = format_param_value(game.alpha_U)
    lambda_str = format_param_value(game.lambda_param)

    full_output_path = os.path.join(
        base_output_directory,
        f"r_{r_str}",
        f"de_{de_variant_str}",
        f"omegaI_{omega_I_str}_omegaR_{omega_R_str}_beta_{beta_str}_alphaU_{alpha_U_str}_lambda_{lambda_str}",
        f"iterations{iterations_str}_grid{grid_size_str}"
    )

    game.output_folder = full_output_path

    os.makedirs(full_output_path, exist_ok=True)

    avg_Is, avg_Rs, avg_payoffs, avg_Reps = game.run_simulation()

    fig_curves = plt.figure(figsize=(8, 6))
    gs_curves = GridSpec(1, 1)

    main_title_params_curves = (
        f"r={game.r}, K(费米)={game.K}, "
        f"奖励成本={game.reward_cost_factor}, 奖励乘数={game.reward_multiplier}, "
        f"声誉权重(I:{game.omega_I}, R:{game.omega_R}), β={game.beta}, α_U={game.alpha_U}, λ={game.lambda_param}, "
        f"DE变种={game.de_variant_name}, F={game.de_F}, CR={game.de_CR}, "
        f"DE邻居={get_neighbor_type_string(game.de_neighbor_type)} (池大小={game.de_pool_size}), "
        f"网格={game.grid_size}x{game.grid_size}, 迭代={game.max_iterations}, "
        f"博弈小组=冯·诺依曼 (组大小={game.game_group_size}), "
        f"奖励分配=I*U'比例 (池∝avgI*sumR), Seed={game.seed}"
    )
    plt.suptitle(f"收益-声誉双目标演化动态 - {main_title_params_curves}", fontsize=12, y=1.05) # 调整主标题位置和字体大小

    # 平均投资动态
    ax_i = plt.subplot(gs_curves[0, 0])
    ax_i.plot(avg_Is, label='平均投资 (I)', color='blue')
    ax_i.set_xlabel('迭代次数')
    ax_i.set_ylabel('值')
    ax_i.set_title('平均投资动态')
    ax_i.set_ylim(-0.05, 1.05)
    ax_i.grid(True, alpha=0.3)
    ax_i.legend(fontsize='small')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    curves_filename = os.path.join(game.output_folder, 'time_series_plots.png')
    plt.savefig(curves_filename, bbox_inches='tight')

    plt.show()

    plt.close(fig_curves)
