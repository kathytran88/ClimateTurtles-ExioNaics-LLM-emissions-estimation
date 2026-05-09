import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

df = pd.read_csv('Kathy/naics2_large_run3_final_model/epoch_log.csv')

epochs = df['epoch'].values
ema_top1 = df['ema_top1'].values * 100
swa_top1 = df['swa_top1'].values * 100
train_loss = df['train_loss'].values

fig, ax1 = plt.subplots(figsize=(10, 5.5))

color_ema = '#1a5276'
color_swa = '#27ae60'
color_loss = '#c0392b'

ln1 = ax1.plot(epochs, ema_top1, 'o-', color=color_ema, linewidth=2.2,
               markersize=7, label='EMA Top-1 Accuracy', zorder=3)
swa_mask = ~np.isnan(swa_top1)
ln2 = ax1.plot(epochs[swa_mask], swa_top1[swa_mask], 's--', color=color_swa,
               linewidth=2.2, markersize=7, label='SWA Top-1 Accuracy', zorder=3)

best_epoch = 6
best_val = swa_top1[best_epoch - 1]
ax1.annotate(f'Best: {best_val:.2f}%\n(SWA, Epoch {best_epoch})',
             xy=(best_epoch, best_val),
             xytext=(best_epoch + 0.8, best_val - 4),
             fontsize=9.5, fontweight='bold', color=color_swa,
             arrowprops=dict(arrowstyle='->', color=color_swa, lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor=color_swa, alpha=0.9))

ema_peak_epoch = 4
ema_peak_val = ema_top1[ema_peak_epoch - 1]
ax1.annotate(f'EMA peak: {ema_peak_val:.2f}%',
             xy=(ema_peak_epoch, ema_peak_val),
             xytext=(ema_peak_epoch - 1.8, ema_peak_val + 3),
             fontsize=9, color=color_ema,
             arrowprops=dict(arrowstyle='->', color=color_ema, lw=1.2),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor=color_ema, alpha=0.9))

ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold', color=color_ema)
ax1.set_ylim(48, 75)
ax1.set_xlim(0.5, 8.5)
ax1.set_xticks(epochs)
ax1.tick_params(axis='y', labelcolor=color_ema)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax1.grid(True, alpha=0.3, linestyle='--')

ax2 = ax1.twinx()
ln3 = ax2.plot(epochs, train_loss, '^:', color=color_loss, linewidth=1.8,
               markersize=6, alpha=0.8, label='Train Loss')
ax2.set_ylabel('Train Loss', fontsize=12, fontweight='bold', color=color_loss)
ax2.tick_params(axis='y', labelcolor=color_loss)
ax2.set_ylim(0.8, 2.0)

ax1.axvline(x=4, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax1.text(4.05, 49.5, 'SWA starts', fontsize=8, color='gray', fontstyle='italic')

lns = ln1 + ln2 + ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='lower right', fontsize=10, framealpha=0.9)

plt.title('Training Dynamics: DeBERTa-v3-large (Best Model)',
          fontsize=14, fontweight='bold', pad=12)

plt.tight_layout()
plt.savefig('Kathy/training_curve.png', dpi=200, bbox_inches='tight')
print('Saved to Kathy/training_curve.png')
plt.close()
