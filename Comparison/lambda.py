import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
from numba import njit, prange
import matplotlib.colors as mcolors
import os
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False



    def calculate_payoffs(self, current_I_grid, current_R_grid, current_U_grid):
        return calculate_all_payoffs_jit(
            current_I_grid, current_R_grid, current_U_grid,
            self.grid_size, self.game_group_offsets, self.game_group_size,
            self.r, self.reward_cost_factor, self.reward_multiplier,
            self.beta, self.max_possible_reputation
        )

    def run_simulation(self):
        self.avg_investments = []
        self.avg_rewards = []
        self.avg_payoffs = []
        self.avg_reputations = []

        self.investment_grid = np.random.rand(self.grid_size, self.grid_size)
        self.reward_grid = np.random.rand(self.grid_size, self.grid_size)
        self.reputation_grid = np.full((self.grid_size, self.grid_size), self.max_possible_reputation / 2.0)


        for iteration in range(self.max_iterations):
            instant_reputations = calculate_all_instant_reputations_jit(
                self.investment_grid, self.reward_grid, self.omega_I, self.omega_R, self.grid_size
            )
            self.reputation_grid = (1.0 - self.alpha_U) * self.reputation_grid + self.alpha_U * instant_reputations
            self.reputation_grid = np.maximum(0.0, np.minimum(self.reputation_grid, self.max_possible_reputation))

            current_payoffs = self.calculate_payoffs(self.investment_grid, self.reward_grid, self.reputation_grid)

            self.investment_grid, self.reward_grid = update_ipr_de_sweep_jit(
                self.investment_grid,
                self.reward_grid,
                current_payoffs,
                self.reputation_grid,
                self.grid_size,
                self.de_F,
                self.de_CR,
                self.de_variant_code,
                self.K,
                self.r,
                self.reward_cost_factor,
                self.reward_multiplier,
                self.game_group_offsets,
                self.game_group_size,
                self.de_pool_offsets,
                self.de_pool_size,
                self.omega_I, self.omega_R, self.lambda_param,
                self.alpha_U, self.beta, self.max_possible_reputation
            )

            avg_I = np.mean(self.investment_grid)
            avg_R = np.mean(self.reward_grid)
            avg_payoff = np.mean(current_payoffs)
            avg_reputation = np.mean(self.reputation_grid)

            self.avg_investments.append(avg_I)
            self.avg_rewards.append(avg_R)
            self.avg_payoffs.append(avg_payoff)
            self.avg_reputations.append(avg_reputation)

        return self.avg_investments[-1]


if __name__ == "__main__":

    r_values = np.linspace(1.0, 5.0, 41)

    lambda_plot_values = [0.0, 0.3, 0.5, 0.8, 1.0]

    num_runs_per_point = 5

    fixed_sim_params = {
        'K': 0.1,
        'grid_size': 50,
        'max_iterations': 2000,
        'de_F': 0.3,
        'de_CR': 0.7,
        'de_variant': 'best/1',
        'de_neighbor_type': 1,
        'reward_cost_factor': 0.3,
        'reward_multiplier': 1,
        'beta': 0.5,
        'omega_I': 1.0,
        'omega_R': 0.5,
        'alpha_U': 0.5,
        'visualize_interval': 0,
        'seed': None,
    }

    output_dir = 'r_lambda_lines_plot_averaged'
    os.makedirs(output_dir, exist_ok=True)

    final_avg_investments_matrix = np.zeros((len(lambda_plot_values), len(r_values)))

    total_simulations = len(r_values) * len(lambda_plot_values) * num_runs_per_point
    start_scan_time = time.time()

    with tqdm(total=total_simulations, desc="Running r-lambda scan") as pbar:
        for lambda_idx, lambda_val in enumerate(lambda_plot_values):
            for r_idx, r_val in enumerate(r_values):
                current_point_results = []
                for run_idx in range(num_runs_per_point):
                    current_sim_params = fixed_sim_params.copy()
                    current_sim_params.update({
                        'r': r_val,
                        'lambda_param': lambda_val,
                        'output_folder': os.path.join(output_dir,
                                                      f'r_{format_param_value(r_val)}_lambda_{format_param_value(lambda_val)}_run_{run_idx}')
                    })

                    game = SPGG_ReputationDE(**current_sim_params)
                    final_avg_I_single_run = game.run_simulation()
                    current_point_results.append(final_avg_I_single_run)

                    pbar.set_description(f"Running r-lambda scan (λ={lambda_val:.2f}, r={r_val:.2f}, Run {run_idx+1}/{num_runs_per_point})")
                    pbar.update(1)

                final_avg_investments_matrix[lambda_idx, r_idx] = np.mean(current_point_results)

    end_scan_time = time.time()
    print(f"所有模拟完成，总耗时: {end_scan_time - start_scan_time:.2f} 秒.")

    data_to_save = {
        'r_values': r_values,
        'lambda_values': np.array(lambda_plot_values),
        'final_avg_investments_averaged': final_avg_investments_matrix,
        'num_runs_per_point': num_runs_per_point,
        'simulation_params': fixed_sim_params
    }
    data_filename = os.path.join(output_dir, 'r_lambda_avg_I_data_averaged.npy')
    np.save(data_filename, data_to_save)
    print(f"模拟数据已保存到: {data_filename}")

    plt.figure(figsize=(10, 7))

    colors = plt.cm.viridis(np.linspace(0, 1, len(lambda_plot_values)))

    for i, lambda_val in enumerate(lambda_plot_values):
        plt.plot(r_values, final_avg_investments_matrix[i, :],
                 label=f'λ = {lambda_val:.1f}',
                 marker='o', markersize=4, linestyle='-', color=colors[i],
                 linewidth=2)

    plt.xlabel('公共品乘数 r')
    plt.ylabel('最终平均投资 (I)')
    plt.title('最终平均投资随公共品乘数 r 的变化 (不同 λ 值)')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.ylim(-0.05, 1.05)

    x_range = r_values.max() - r_values.min()
    plt.xlim(r_values.min() - x_range * 0.02, r_values.max() + x_range * 0.02)

    num_ticks = 5
    tick_indices = np.linspace(0, len(r_values) - 1, num_ticks, dtype=int)
    plt.xticks(np.round(r_values[tick_indices], 1))

    plt.legend(title='适应度权重 λ', loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.tight_layout()

    plot_filename = os.path.join(output_dir, 'avg_I_vs_r_different_lambda_averaged.png')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.show()
    plt.close()
