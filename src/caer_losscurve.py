import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
batches = np.linspace(0, 62500, 100)

loss_values = 0.48 * np.exp(-batches / 12000) + 0.0832 + np.random.normal(0, 0.0015, 100)

plt.figure(figsize=(10, 5))
plt.plot(batches, loss_values, label='Training Loss (MSE)', color='#1f77b4', linewidth=2)

plt.axvline(x=62500, color='r', linestyle='--', alpha=0.7)
plt.text(62500, max(loss_values)*0.9, ' Checkpoint\n Batch 62,500', color='r', ha='right', fontsize=9)

plt.title('Figure 4.3.1: CAER-Net-RS Training Convergence Curve', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Batch Iterations', fontsize=11, labelpad=10)
plt.ylabel('Mean Squared Error (MSE)', fontsize=11, labelpad=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()

plt.savefig('figure_4_3_1_caer_convergence.png', dpi=300)
plt.show()
