import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter, zoom
import matplotlib.ticker as ticker


def plot_saved_heatmap_with_style(scenario_name='DE_best1_1st_order',
                                      zoom_factor=4, smoothing_sigma=6):
    heatmap_base_output_dir = f'heatmap_results_r_deF_{scenario_name}'
    data_file_name = 'heatmap_data_avg_I.npz'
    data_path = os.path.join(heatmap_base_output_dir, data_file_name)

    # 调整重新绘制的输出文件名
    replot_output_filename = f'heatmap_avg_I_r_deF_{scenario_name}.png'
    replot_output_path = os.path.join(heatmap_base_output_dir, replot_output_filename)

    print(f"尝试从 '{data_path}' 读取数据...")

    if not os.path.exists(data_path):
        print(f"错误：未找到数据文件 '{data_path}'。")
        return

    try:
        data = np.load(data_path)

        loaded_avg_I = data['avg_I']
        loaded_r_values = data['r_values']
        loaded_de_F_values = data['de_F_values']

        print("数据读取成功。")
        print(f"读取到的 r 范围: [{loaded_r_values.min():.2f}, {loaded_r_values.max():.2f}] ({len(loaded_r_values)}个值)")
        print(f"读取到的 F 范围: [{loaded_de_F_values.min():.2f}, {loaded_de_F_values.max():.2f}] ({len(loaded_de_F_values)}个值)")
        print(f"平均投资数据形状: {loaded_avg_I.shape}")

        print(f"正在对数据进行上采样 (factor={zoom_factor}) 和高斯平滑 (sigma={smoothing_sigma})...")
        upsampled_avg_I = zoom(loaded_avg_I, zoom=zoom_factor, order=3)
        smoothed_avg_I = gaussian_filter(upsampled_avg_I, sigma=smoothing_sigma)
        print(f"处理后的数据形状: {smoothed_avg_I.shape}")

        # 调整 extent，Y轴使用 F 的范围
        common_extent = [loaded_r_values.min(), loaded_r_values.max(),
                         loaded_de_F_values.min(), loaded_de_F_values.max()]

        print("正在重新绘制热图...")

        fig, ax = plt.subplots(1, 1, figsize=(8, 7), constrained_layout=True)

        im = ax.imshow(smoothed_avg_I, origin='lower', cmap='jet',
                       extent=common_extent, aspect='auto', vmin=0, vmax=1)

        ax.set_title('Avg I', fontsize=16)
        ax.set_xlabel('r', fontsize=14)
        ax.set_ylabel('F', fontsize=14)

        # X轴刻度设置（r值）
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

        # Y轴刻度设置（F值）
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=10)

        plt.colorbar(im, ax=ax)

        plt.savefig(replot_output_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"热图已重新绘制并保存到: {replot_output_path}")

    except Exception as e:
        print(f"读取或绘图过程中发生错误: {e}")

if __name__ == "__main__":
    plot_saved_heatmap_with_style(scenario_name='DE_best1_1st_order')
