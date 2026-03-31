import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
from numba import njit, prange
import matplotlib.colors as mcolors
import os
import json  # For saving parameters and results
from tqdm import tqdm  # Import tqdm

# Remove or comment out font settings for Chinese display
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False



@njit(cache=True, parallel=True)
def update_ipr_de_sweep_jit(current_I_grid, current_R_grid, current_payoffs, current_reputations,
                            grid_size, de_F, de_CR, de_variant_code, K_param, r_param,
                            reward_cost_factor_param, reward_multiplier_param,
                            game_group_offsets, game_group_size,
                            de_pool_offsets, de_pool_size,
                            omega_I, omega_R, lambda_param,
                            alpha_U_param, beta_param, max_reputation_value_param):
    """
    Iterates through all individuals to perform DE strategy updates.
    """
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


# --- Helper Functions (Modified for English output) ---

def format_param_value(value):
    """Formats parameter value for use in filenames"""
    if isinstance(value, float):
        return f"{value:.2f}".replace('.', 'p')
    elif isinstance(value, int):
        return str(value)
    else:
        return str(value)


def get_neighbor_type_string(neighbor_type_int):
    """Maps neighbor type integer to a descriptive string (English)."""
    if neighbor_type_int == 1:
        return "Order-1 Neighbors"
    elif neighbor_type_int == 2:
        return "Order-2 Neighbors"
    else:
        return f"Unknown Neighbor Type ({neighbor_type_int})"


# --- Simulation Class (Modified for English output) ---

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

        # DE strategy mapping (for internal code representation)
        self.de_strategy_map = {
            'best/1': 1,
            'best/2': 2
        }
        if de_variant not in self.de_strategy_map:
            raise ValueError(f"Unsupported DE variant: {de_variant}. Options: {list(self.de_strategy_map.keys())}")
        self.de_variant_name = de_variant
        self.de_variant_code = self.de_strategy_map[de_variant]

        # Define game group (always Von Neumann neighbors)
        self.game_group_offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        self.game_group_size = len(self.game_group_offsets)

        # Define DE strategy's neighbor pool
        self.de_neighbor_type = de_neighbor_type
        if self.de_neighbor_type == 1:
            self.de_pool_offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]  # Von Neumann (Manhattan distance <= 1)
        elif self.de_neighbor_type == 2:
            # Manhattan distance <= 2 neighbors (including self)
            self.de_pool_offsets = []
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if abs(dr) + abs(dc) <= 2:
                        self.de_pool_offsets.append((dr, dc))
        else:
            raise ValueError(f"Unsupported DE neighbor type: {de_neighbor_type}. Options: 1 (Order-1 Neighbors), 2 (Order-2 Neighbors)")
        self.de_pool_size = len(self.de_pool_offsets)

        self.reward_cost_factor = reward_cost_factor
        self.reward_multiplier = reward_multiplier
        self.beta = beta
        # self.r is already set in __init__

        self.omega_I = omega_I
        self.omega_R = omega_R
        self.alpha_U = alpha_U
        self.lambda_param = lambda_param

        self.visualize_interval = visualize_interval
        self.output_folder = output_folder

        # Initialize investment and reward willingness grids
        self.investment_grid = np.random.rand(grid_size, grid_size)
        self.reward_grid = np.random.rand(grid_size, grid_size)

        # Max possible reputation value, for normalization and initialization
        self.max_possible_reputation = self.omega_I * 1.0 + self.omega_R * 1.0
        # Initialize reputation grid to a neutral value (e.g., max_possible_reputation / 2.0)
        self.reputation_grid = np.full((grid_size, grid_size), self.max_possible_reputation / 2.0)

        # Record historical data
        self.avg_investments = []
        self.avg_rewards = []
        self.avg_payoffs = []
        self.avg_reputations = []

    def calculate_payoffs(self, current_I_grid, current_R_grid, current_U_grid):
        """Calculates payoffs for all individuals"""
        return calculate_all_payoffs_jit(
            current_I_grid, current_R_grid, current_U_grid,
            self.grid_size, self.game_group_offsets, self.game_group_size,
            self.r, self.reward_cost_factor, self.reward_multiplier,
            self.beta, self.max_possible_reputation
        )

    def run_simulation(self):
        start_time = time.time()

        for iteration in range(self.max_iterations):
            # Phase 1: Calculate payoffs, reputations, and fitness for the current time step
            instant_reputations = calculate_all_instant_reputations_jit(
                self.investment_grid, self.reward_grid, self.omega_I, self.omega_R, self.grid_size
            )
            self.reputation_grid = (1.0 - self.alpha_U) * self.reputation_grid + self.alpha_U * instant_reputations
            self.reputation_grid = np.maximum(0.0, np.minimum(self.reputation_grid, self.max_possible_reputation))

            current_payoffs = self.calculate_payoffs(self.investment_grid, self.reward_grid, self.reputation_grid)

            # Phase 2: Strategy update (generate strategies for the next time step)
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

            # Record averages
            avg_I = np.mean(self.investment_grid)
            avg_R = np.mean(self.reward_grid)
            avg_payoff = np.mean(current_payoffs)
            avg_reputation = np.mean(self.reputation_grid)

            self.avg_investments.append(avg_I)
            self.avg_rewards.append(avg_R)
            self.avg_payoffs.append(avg_payoff)
            self.avg_reputations.append(avg_reputation)

        return (self.avg_investments, self.avg_rewards,
                self.avg_payoffs, self.avg_reputations)

    def visualize(self, iteration, avg_I, avg_R, avg_payoff, avg_reputation, output_folder):
        """
        Visualizes the current grid state, including investment, reward willingness, and reputation distribution.
        This function is typically disabled when running batch simulations for heatmaps (visualize_interval=0).
        """
        fig = plt.figure(figsize=(18, 6))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 1])

        # Investment distribution
        ax1 = plt.subplot(gs[0])
        # Use 'jet' colormap, vmin=0, vmax=1
        im1 = ax1.imshow(self.investment_grid, cmap='jet', vmin=0, vmax=1, interpolation='nearest')
        ax1.set_title(f'Iteration: {iteration}\nInvestment (Avg I: {avg_I:.3f})')
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Investment Level')

        # Reward Willingness distribution
        ax2 = plt.subplot(gs[1])
        im2 = ax2.imshow(self.reward_grid, cmap='Greens', vmin=0, vmax=1, interpolation='nearest')
        ax2.set_title(f'Reward Willingness (Avg R: {avg_R:.3f})')
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Reward Willingness Level')

        # Reputation distribution
        ax3 = plt.subplot(gs[2])
        im3 = ax3.imshow(self.reputation_grid, cmap='YlOrRd', vmin=0, vmax=self.max_possible_reputation,
                         interpolation='nearest')
        ax3.set_title(f'Reputation (Avg U: {avg_reputation:.3f})')
        ax3.set_xticks([])
        ax3.set_yticks([])
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Reputation Value')

        # Main title
        main_title_params = (
            f"r={self.r}, K(Fermi)={self.K}, "
            f"Reward Cost={self.reward_cost_factor}, Reward Multiplier={self.reward_multiplier}, "
            f"Reputation Weights(I:{self.omega_I}, R:{self.omega_R}), β={self.beta}, α_U={self.alpha_U}, λ={self.lambda_param}, "
            f"DE Variant={self.de_variant_name}, F={self.de_F}, CR={self.de_CR}, "
            f"DE Neighbors={get_neighbor_type_string(self.de_neighbor_type)} (Pool Size={self.de_pool_size}), "
            f"Game Group=Von Neumann (Group Size={self.game_group_size}), "
            f"Reward Allocation=I*U' Ratio (Pool∝avgI*sumR), Seed={self.seed}"
        )
        plt.suptitle(f"Grid State Snapshot - {main_title_params}", fontsize=14, y=1.02)

        plt.tight_layout(rect=[0, 0, 1, 0.98])

        filename = os.path.join(output_folder, f'iteration_{iteration:06d}.png')
        plt.savefig(filename, bbox_inches='tight')

        plt.show()

        plt.close(fig)


if __name__ == "__main__":

    # --- Heatmap Parameter Setup ---
    # Define the single simulation scenario to run
    # You can modify this dictionary to select different configurations for individual runs
    # Available 'name' and corresponding 'de_variant' and 'de_neighbor_type' combinations:
    # 1. {'de_variant': 'best/1', 'de_neighbor_type': 1, 'name': 'DE_best1_1st_order'}
    # 2. {'de_variant': 'best/1', 'de_neighbor_type': 2, 'name': 'DE_best1_2nd_order'}
    # 3. {'de_variant': 'best/2', 'de_neighbor_type': 1, 'name': 'DE_best2_1st_order'}
    # 4. {'de_variant': 'best/2', 'de_neighbor_type': 2, 'name': 'DE_best2_2nd_order'}

    # Example: Currently running DE/best/1 (1st order neighbors)
    current_scenario = {'de_variant': 'best/1', 'de_neighbor_type': 1, 'name': 'DE_best1_1st_order'}

    # Set output directory based on current scenario
    heatmap_base_output_dir = f'heatmap_results_r_deF_{current_scenario["name"]}' # Changed folder name
    os.makedirs(heatmap_base_output_dir, exist_ok=True)

    # Define ranges and steps for r and de_F
    r_values = np.linspace(1.0, 5.0, 50)
    de_F_values = np.linspace(0, 1.0, 50) # Changed from lambda_param_values to de_F_values

    # Other fixed parameters
    common_sim_params = {
        'K': 0.1,
        'grid_size': 50,
        'max_iterations': 2000,  # Ensure enough iterations to reach stability
        # 'de_F': 0.3, # de_F is now scanned, so remove it from here
        'de_CR': 0.7,
        'reward_cost_factor': 0.3,
        'reward_multiplier': 1,
        'beta': 0.5,
        'omega_I': 1.0,
        'omega_R': 0.5,
        'alpha_U': 0.5,
        'lambda_param': 0.7, # lambda_param is now fixed
        'visualize_interval': 0,  # Disable real-time visualization during batch runs
        'seed': None,  # Use a different random seed for each simulation
    }

    print(f"--- Starting r vs de_F heatmap generation for {current_scenario['name']} (Investment Only) ---") # Changed print message
    print(f"r range: {r_values[0]:.2f} to {r_values[-1]:.2f} ({len(r_values)} steps)")
    print(f"de_F range: {de_F_values[0]:.2f} to {de_F_values[-1]:.2f} ({len(de_F_values)} steps)") # Changed print message
    print(f"Will run {len(r_values) * len(de_F_values)} simulations.")
    print(f"Iterations per simulation: {common_sim_params['max_iterations']}")
    print(f"Results will be saved to: {heatmap_base_output_dir}")

    # Initialize matrix to store results, only average investment
    results_avg_I = np.zeros((len(de_F_values), len(r_values)))  # Changed dimensions: (Y-axis, X-axis) -> (de_F, r)

    start_scenario_time = time.time()

    # Calculate total number of simulations
    total_simulations = len(r_values) * len(de_F_values)

    # Use tqdm to wrap the outer loop and manually update the progress bar
    with tqdm(total=total_simulations, desc=f"Running {current_scenario['name']} scan") as pbar:
        for i, de_F_val in enumerate(de_F_values):  # Y-axis is de_F
            for j, r_val in enumerate(r_values):  # X-axis is r
                current_sim_params = common_sim_params.copy()
                current_sim_params.update({
                    'r': r_val,
                    'de_F': de_F_val, # Pass de_F_val here
                    'de_variant': current_scenario['de_variant'],
                    'de_neighbor_type': current_scenario['de_neighbor_type'],
                    'output_folder': os.path.join(heatmap_base_output_dir,
                                                  f'r_{format_param_value(r_val)}_deF_{format_param_value(de_F_val)}') # Changed folder name
                })

                game = SPGG_ReputationDE(**current_sim_params)
                # Unpack only avg_Is, ignore others with _
                avg_Is, _, _, _ = game.run_simulation()

                # Store the average investment from the last iteration
                results_avg_I[i, j] = avg_Is[-1]

                # Update progress bar description to show current parameters
                pbar.set_description(f"Running {current_scenario['name']} scan (r={r_val:.2f}, F={de_F_val:.2f})") # Changed print message
                pbar.update(1)  # Update progress bar after each inner loop simulation

    end_scenario_time = time.time()
    print(f"Scenario {current_scenario['name']} completed, time taken: {end_scenario_time - start_scenario_time:.2f} seconds.") # Changed print message

    # Save data, only average investment
    np.savez(
        os.path.join(heatmap_base_output_dir, 'heatmap_data_avg_I.npz'),
        avg_I=results_avg_I,
        r_values=r_values,
        de_F_values=de_F_values, # Changed from lambda_param_values
        params=common_sim_params  # Save common parameters
    )

    # Save parameters as JSON for easy viewing
    full_params_to_save = common_sim_params.copy()
    full_params_to_save['r_values_range'] = [r_values[0], r_values[-1], len(r_values)]
    full_params_to_save['de_F_values_range'] = [de_F_values[0], de_F_values[-1], len(de_F_values)] # Changed from lambda_param_values
    full_params_to_save['de_variant'] = current_scenario['de_variant']
    full_params_to_save['de_neighbor_type'] = current_scenario['de_neighbor_type']

    with open(os.path.join(heatmap_base_output_dir, 'simulation_params.json'), 'w') as f:
        json.dump(full_params_to_save, f, indent=4)

    print("\n--- Simulations completed, generating heatmap ---") # Changed print message

    # --- Plot Heatmap ---
    print(f"Plotting heatmap for {current_scenario['name']}...") # Changed print message

    plot_title = 'Average Investment (I)' # Changed title to English

    fig, ax = plt.subplots(figsize=(8, 7))

    # Use 'jet' colormap, vmin=0, vmax=1
    im = ax.imshow(results_avg_I, cmap='jet',
                   vmin=0.0, vmax=1.0,  # Investment level fixed from 0 to 1
                   extent=[r_values[0], r_values[-1], de_F_values[0], de_F_values[-1]], # Changed extent
                   origin='lower', aspect='auto', interpolation='nearest')

    ax.set_title(f'{current_scenario["name"]} - {plot_title}')
    ax.set_xlabel('Public Goods Multiplier r') # Changed X-axis label to English
    ax.set_ylabel('DE Mutation Factor F')  # Changed Y-axis label to English

    # Set tick labels
    ax.set_xticks(r_values[::max(1, len(r_values) // 5)])  # Show ticks every ~5 values to prevent overcrowding
    ax.set_yticks(de_F_values[::max(1, len(de_F_values) // 5)])  # Show ticks every ~5 values to prevent overcrowding
    ax.set_xticklabels([f'{val:.1f}' for val in r_values[::max(1, len(r_values) // 5)]], rotation=45, ha='right')
    ax.set_yticklabels([f'{val:.1f}' for val in de_F_values[::max(1, len(de_F_values) // 5)]]) # Changed Y-axis tick labels

    plt.colorbar(im, ax=ax, label=plot_title)
    plt.tight_layout()
    plt.savefig(os.path.join(heatmap_base_output_dir, f'heatmap_avg_I_r_deF.png')) # Changed filename
    plt.close(fig)

    print("\n--- Heatmap generation completed ---") # Changed print message
