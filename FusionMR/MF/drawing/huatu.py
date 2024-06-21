import numpy as np
import os
import matplotlib.pyplot as plt

# 指定保存数据的文件夹路径
folder_path = r'C:\Users\Raytrack\Documents\MR_fussion\MF\huatu_data'

# 加载保存的数据
data = np.load(os.path.join(folder_path, 'loss.npz'))
my_train_loss = data['my_train_loss']
my_eval_loss = data['my_train_loss']

# 确保两个列表具有相同的长度，截取它们的共同部分
min_length = min(len(my_train_loss), len(my_eval_loss))
my_train_loss = my_train_loss[:min_length]
my_eval_loss = my_eval_loss[:min_length]

# 创建 x 轴数据，以每隔80个点绘制
interval = 100
epochs = range(1, len(my_train_loss) + 1, interval)

# 仅保留每隔80个点的损失值
my_train_loss = my_train_loss[::interval]
my_eval_loss = my_eval_loss[::interval]

# 绘制训练损失和评估损失图表
plt.plot(epochs, my_train_loss, linestyle='-', color='blue', label='train_loss')
plt.plot(epochs, my_eval_loss, linestyle='--', color='yellow',alpha=0.8, label='valid_loss')
plt.title('Loss of Train and Valid')
plt.xlabel('Times')
plt.ylabel('Loss Value')

# Remove top and right spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.grid(False)  # Remove grid lines
plt.legend()
plt.show()
