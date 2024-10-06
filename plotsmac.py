import json
import matplotlib.pyplot as plt

# 讀取多個 JSON 檔案
def load_data(file_paths):
    all_data = []
    for file_path in file_paths:
        with open(file_path) as f:
            all_data.append(json.load(f))
    return all_data

# 定義要繪製的數據
def plot_data(data_list, keys, name_list, battle_name):
    
    for key in keys:
        # 創建一個圖形
        fig, ax = plt.subplots()
        # 設置坐標軸背景顏色
        ax.set_facecolor('lightyellow')
        # 設置網格顏色
        ax.grid(color='green', linestyle='--', linewidth=0.5)
        for data, name in zip(data_list, name_list):
            if key in data:
                ax.plot(data[key + '_T'], data[key], label=name) 
        ax.set_xlabel('Time Steps')
        ax.set_ylabel(key)
        ax.set_title(battle_name)
        ax.legend()

    plt.show()

# 使用者選擇要繪製的數據
file_paths = ['pymarl/results/sacred/35/info.json']  # 可以根據需要修改
data_list = load_data(file_paths)
selected_keys = ['battle_won_mean', 'dead_allies_mean', 'dead_enemies_mean']
name_list = ['Qmix', 'myQ'] 
battle_name = '2s3z'
plot_data(data_list, selected_keys, name_list, battle_name)