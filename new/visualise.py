import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict

# -----------------------------------------------------------------------------
# 1) Define the original VGG conv‐layer channels (ignore 'M's)
# -----------------------------------------------------------------------------
original_arch = [64, 64, 128, 128, 256, 256, 256,
                 512, 512, 512, 512, 512, 512]
orig = np.array(original_arch)

# -----------------------------------------------------------------------------
# 2) Read & clean the pruning results file
# -----------------------------------------------------------------------------
with open("pruned_ratio.txt", "r") as f:
    lines = [line.strip() for line in f if line.strip()]

# -----------------------------------------------------------------------------
# 3) Parse into (ratio, [filters...]) pairs
# -----------------------------------------------------------------------------
ratios, configs = [], []
i = 0
while i < len(lines):
    if lines[i].startswith("Pruned ratio"):
        # parse ratio
        ratio = float(lines[i].split(":", 1)[1].strip())
        # sanity‐check next line
        assert lines[i+1].startswith("Configuration"), \
            f"Expected 'Configuration' at line {i+1}"
        # parse the numbers two lines down
        tokens = re.split(r"\s+", lines[i+2])
        cfg = [int(tok) for tok in tokens if tok != "M"]
        ratios.append(ratio)
        configs.append(cfg)
        i += 3
    else:
        i += 1

# -----------------------------------------------------------------------------
# 4) Aggregate by pruning ratio & compute mean config
# -----------------------------------------------------------------------------
ratio_to_cfgs = defaultdict(list)
for r, cfg in zip(ratios, configs):
    ratio_to_cfgs[r].append(cfg)

avg_cfgs = {}
for r, cfg_list in ratio_to_cfgs.items():
    stacked = np.stack([np.array(c) for c in cfg_list])
    avg_cfgs[r] = np.mean(stacked, axis=0)

# -----------------------------------------------------------------------------
# 5) Compute percent pruned per layer and plot
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
for r in sorted(avg_cfgs):
    kept = avg_cfgs[r]
    pruned_pct = (orig - kept) / orig * 100.0
    plt.plot(pruned_pct, marker="o", label=f"Pruned Ratio {r:.2f}")

plt.title("Percentage of Filters Pruned per Conv‐Layer")
plt.xlabel("Conv‐Layer Index (0-based)")
plt.ylabel("Percent Pruned (%)")
plt.grid(True)
plt.legend(title="Global Pruning Ratio")
plt.tight_layout()
plt.show()
