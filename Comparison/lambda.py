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


@njit(cache=True)
def calculate_payoff_from_one_game_jit(focal_r, focal_c, group_members_coords, game_group_size,
                                       current_I_grid, current_R_grid, current_U_grid,
                                       r_param, reward_cost_factor_param,
                                       reward_multiplier_param, beta_param,
                                       max_reputation_value_param):
    group_I_sum = 0.0
    group_R_sum = 0.0
    group_weighted_I_sum = 0.0

    for r_member, c_member in group_members_coords:
        I_j = current_I_grid[r_member, c_member]
        R_j = current_R_grid[r_member, c_member]
        U_j = current_U_grid[r_member, c_member]
        U_j_normalized = U_j / max_reputation_value_param

        group_I_sum += I_j
        group_R_sum += R_j
        group_weighted_I_sum += I_j * (1.0 + beta_param * U_j_normalized)

    group_avg_I = group_I_sum / game_group_size

    focal_I = current_I_grid[focal_r, focal_c]
    focal_R = current_R_grid[focal_r, focal_c]
    focal_U = current_U_grid[focal_r, focal_c]
    focal_U_normalized = focal_U / max_reputation_value_param

    public_goods_benefit = r_param * group_avg_I
    investment_cost = focal_I
    reward_contribution_cost = reward_cost_factor_param * focal_R

    reward_pool = reward_multiplier_param * group_avg_I * group_R_sum

    reward_received = 0.0

    if group_I_sum > 0 and group_weighted_I_sum > 0:
        focal_weighted_I = focal_I * (1.0 + beta_param * focal_U_normalized)
        reward_received = reward_pool * (focal_weighted_I / group_weighted_I_sum)

    payoff = (
            public_goods_benefit
            - investment_cost
            - reward_contribution_cost
            + reward_received
    )
    return payoff


@njit(cache=True)
def calculate_single_payoff_jit(r, c, temp_I_grid, temp_R_grid, temp_U_grid, grid_size,
                                game_group_offsets, game_group_size,
                                r_param, reward_cost_factor_param, reward_multiplier_param,
                                beta_param, max_reputation_value_param):
    total_payoff_rc = 0.0

    centers_including_rc = []
    for dr, dc in game_group_offsets:
        center_r = (r - dr + grid_size) % grid_size
        center_c = (c - dc + grid_size) % grid_size
        centers_including_rc.append((center_r, center_c))

    for cr, cc in centers_including_rc:
        group_members_coords = []
        for dr_member, dc_member in game_group_offsets:
            group_members_coords.append(
                ((cr + dr_member + grid_size) % grid_size, (cc + dc_member + grid_size) % grid_size))

        payoff_from_this_game = calculate_payoff_from_one_game_jit(
            r, c, group_members_coords, game_group_size,
            temp_I_grid, temp_R_grid, temp_U_grid,
            r_param, reward_cost_factor_param, reward_multiplier_param,
            beta_param, max_reputation_value_param
        )
        total_payoff_rc += payoff_from_this_game

    return total_payoff_rc


@njit(cache=True, parallel=True)
def calculate_all_payoffs_jit(current_I_grid, current_R_grid, current_U_grid, grid_size,
                              game_group_offsets, game_group_size,
                              r_param, reward_cost_factor_param, reward_multiplier_param,
                              beta_param, max_reputation_value_param):
    all_payoffs = np.empty((grid_size, grid_size))

    for i in prange(grid_size):
        for j in range(grid_size):
            all_payoffs[i, j] = calculate_single_payoff_jit(
                i, j, current_I_grid, current_R_grid, current_U_grid, grid_size,
                game_group_offsets, game_group_size,
                r_param, reward_cost_factor_param, reward_multiplier_param,
                beta_param, max_reputation_value_param
            )
    return all_payoffs


@njit(cache=True)
def calculate_instant_reputation_jit(I_val, R_val, omega_I, omega_R):
    return omega_I * I_val + omega_R * R_val


@njit(cache=True, parallel=True)
def calculate_all_instant_reputations_jit(current_I_grid, current_R_grid, omega_I, omega_R, grid_size):
    all_instant_reputations = np.empty((grid_size, grid_size))
    for i in prange(grid_size):
        for j in range(grid_size):
            all_instant_reputations[i, j] = calculate_instant_reputation_jit(current_I_grid[i, j], current_R_grid[i, j],
                                                                             omega_I, omega_R)
    return all_instant_reputations


@njit(cache=True)
def perform_single_de_step_jit(focal_r, focal_c, current_I_grid, current_R_grid, current_payoffs, current_reputations,
                               grid_size, de_F, de_CR, de_variant_code, K_param, r_param,
                               reward_cost_factor_param, reward_multiplier_param,
                               game_group_offsets, game_group_size,
                               de_pool_offsets, de_pool_size,
                               omega_I, omega_R, lambda_param,
                               alpha_U_param, beta_param, max_reputation_value_param):
    xi_vec = np.array([current_I_grid[focal_r, focal_c], current_R_grid[focal_r, focal_c]])
    payoff_current = current_payoffs[focal_r, focal_c]
    reputation_current = current_reputations[focal_r, focal_c]
    fitness_current = lambda_param * payoff_current + (1.0 - lambda_param) * reputation_current

    de_pool_coords_list = []
    for dr, dc in de_pool_offsets:
        de_pool_coords_list.append(((focal_r + dr + grid_size) % grid_size, (focal_c + dc + grid_size) % grid_size))

    de_pool_strategies = np.empty((de_pool_size, 2))
    de_pool_payoffs = np.empty(de_pool_size)

    for k in range(de_pool_size):
        nr, nc = de_pool_coords_list[k]
        de_pool_strategies[k] = np.array([current_I_grid[nr, nc], current_R_grid[nr, nc]])
        de_pool_payoffs[k] = current_payoffs[nr, nc]

    de_pool_fitnesses = np.empty(de_pool_size)
    for k in range(de_pool_size):
        de_pool_fitnesses[k] = lambda_param * de_pool_payoffs[k] + (1.0 - lambda_param) * current_reputations[de_pool_coords_list[k][0], de_pool_coords_list[k][1]]

    best_idx_in_de_pool = np.argmax(de_pool_fitnesses)
    xbest_vec = de_pool_strategies[best_idx_in_de_pool]

    mutant_vec = np.zeros(2)

    all_de_pool_indices = np.arange(de_pool_size)

    if de_variant_code == 1:
        candidate_indices_for_best = []
        for idx in all_de_pool_indices:
            if idx != best_idx_in_de_pool:
                candidate_indices_for_best.append(idx)
        candidate_indices_for_best_arr = np.array(candidate_indices_for_best)

        if len(candidate_indices_for_best_arr) < 2: return xi_vec

        r_indices_selection = np.random.choice(candidate_indices_for_best_arr, 2, replace=False)
        vec_r1 = de_pool_strategies[r_indices_selection[0]]
        vec_r2 = de_pool_strategies[r_indices_selection[1]]
        mutant_vec = xbest_vec + de_F * (vec_r1 - vec_r2)

    elif de_variant_code == 2:
        candidate_indices_for_best = []
        for idx in all_de_pool_indices:
            if idx != best_idx_in_de_pool:
                candidate_indices_for_best.append(idx)
        candidate_indices_for_best_arr = np.array(candidate_indices_for_best)

        if len(candidate_indices_for_best_arr) < 4: return xi_vec

        r_indices_selection = np.random.choice(candidate_indices_for_best_arr, 4, replace=False)
        vec_r1 = de_pool_strategies[r_indices_selection[0]]
        vec_r2 = de_pool_strategies[r_indices_selection[1]]
        vec_r3 = de_pool_strategies[r_indices_selection[2]]
        vec_r4 = de_pool_strategies[r_indices_selection[3]]
        mutant_vec = xbest_vec + de_F * (vec_r1 - vec_r2) + de_F * (vec_r3 - vec_r4)

    else:
        return xi_vec

    trial_vec = np.zeros(2)
    j_rand = np.random.randint(0, 2)
    for d in range(2):
        if np.random.rand() < de_CR or d == j_rand:
            trial_vec[d] = mutant_vec[d]
        else:
            trial_vec[d] = xi_vec[d]

    trial_I = max(0.0, min(trial_vec[0], 1.0))
    trial_R = max(0.0, min(trial_vec[1], 1.0))

    temp_I_grid = current_I_grid.copy()
    temp_R_grid = current_R_grid.copy()
    temp_U_grid = current_reputations.copy()

    temp_I_grid[focal_r, focal_c] = trial_I
    temp_R_grid[focal_r, focal_c] = trial_R

    instant_reputation_trial = calculate_instant_reputation_jit(trial_I, trial_R, omega_I, omega_R)
    reputation_trial = (1.0 - alpha_U_param) * reputation_current + alpha_U_param * instant_reputation_trial
    reputation_trial = max(0.0, min(reputation_trial, max_reputation_value_param))

    temp_U_grid[focal_r, focal_c] = reputation_trial

    payoff_trial = calculate_single_payoff_jit(
        focal_r, focal_c, temp_I_grid, temp_R_grid, temp_U_grid, grid_size,
        game_group_offsets, game_group_size,
        r_param, reward_cost_factor_param, reward_multiplier_param,
        beta_param, max_reputation_value_param
    )

    fitness_trial = lambda_param * payoff_trial + (1.0 - lambda_param) * reputation_trial

    delta_fitness = fitness_trial - fitness_current

    prob_accept = 0.0
    if K_param == 0:
        prob_accept = 1.0 if delta_fitness >= 0 else 0.0
    else:
        exponent = -delta_fitness / K_param
        if exponent > 50.0:
            exponent = 50.0
        elif exponent < -50.0:
            exponent = -50.0
        prob_accept = 1.0 / (1.0 + np.exp(exponent))

    if np.random.rand() < prob_accept:
        return np.array([trial_I, trial_R])
    else:
        return xi_vec


@njit(cache=True, parallel=True)
def update_ipr_de_sweep_jit(current_I_grid, current_R_grid, current_payoffs, current_reputations,
                            grid_size, de_F, de_CR, de_variant_code, K_param, r_param,
                            reward_cost_factor_param, reward_multiplier_param,
                            game_group_offsets, game_group_size,
                            de_pool_offsets, de_pool_size,
                            omega_I, omega_R, lambda_param,
                            alpha_U_param, beta_param, max_reputation_value_param):
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

        self.investment_grid = np.random.rand(grid_size, grid_size)
        self.reward_grid = np.random.rand(grid_size, grid_size)

        self.max_possible_reputation = self.omega_I * 1.0 + self.omega_R * 1.0
        self.reputation_grid = np.full((grid_size, grid_size), self.max_possible_reputation / 2.0)

        self.avg_investments = []
        self.avg_rewards = []
        self.avg_payoffs = []
        self.avg_reputations = []

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
