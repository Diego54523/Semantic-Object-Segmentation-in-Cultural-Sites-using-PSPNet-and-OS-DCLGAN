import os
import matplotlib.pyplot as plt

os.makedirs('plots', exist_ok=True)

x_percentages = [0, 50, 100]

pa = [51.89, 58.88, 56.33]
mpa = [22.83, 54.01, 61.29]
miou = [13.61, 32.51, 36.64]
fwavacc = [37.07, 42.31, 41.06]

metrics = [
    ('Accuracy', pa, 'Accuracy %', 'accuracy_plot.png'),
    ('Class Accuracy', mpa, 'Class Accuracy %', 'class_accuracy_plot.png'),
    ('Mean IoU', miou, 'Mean IoU %', 'mean_iou_plot.png'),
    ('FWAVACC', fwavacc, 'FWAVACC %', 'fwavacc_plot.png')
]

plt.style.use('seaborn-v0_8-whitegrid')

for title, y_data, ylabel, filename in metrics:
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(x_percentages, y_data, marker='^', color='green', linewidth=2, markersize=8, label='OS-DCLGAN')
    
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel('% of Real Images used on training', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    
    ax.set_xticks(range(0, 105, 10))
    ax.set_xlim(-5, 105)
    
    y_margin = (max(y_data) - min(y_data)) * 0.15
    if y_margin == 0: 
        y_margin = 5
    ax.set_ylim(min(y_data) - y_margin, max(y_data) + y_margin)
    
    ax.legend(loc='lower right', fontsize=12)
    
    plt.tight_layout()
    
    filepath = os.path.join('plots', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)