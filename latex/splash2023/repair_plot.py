import json
import re

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.cm as cm
import matplotlib
import tikzplotlib


def plot_data(data, filename):
    # Define a mapping from '60s' ... '5s' to numerical values
    second_mapping = {f"{i}": i for i in range(10, 0, -1)}

    ind = np.arange(len(second_mapping))  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    bottom = np.zeros(len(second_mapping))
    colors = cm.rainbow(np.linspace(0, len(data)))  # Generate as many colors as there are rows

    # Inside your for loop, create each bar as a separate object and append them to a list:
    bars = []
    for i, (label, scores) in enumerate(reversed(data)):
        vals = [scores[s] for s in second_mapping]
        bar = ax.bar(ind, vals, width, bottom=np.zeros(len(second_mapping)), color=colors[i], label=label)
        bars.append(bar)

    # Now, instead of ax.legend(), use:
    ax.legend(handles=[bar[0] for bar in bars])


    ax.set_xlabel('Seconds')
    ax.set_xticks(ind)
    ax.set_xticklabels(list(second_mapping.keys()))
    # ax.bar(ind, vals, width, bottom=bottom, color=colors[i], label=label)
    # ax.legend()
    ax.legend(loc='upper right')
    ax.set_ylabel('Precision@k')
    ax.set_title('Precision at rank k under varying latency cutoffs')

    # ax.set_ylim([0,1])  # Explicitly set y-axis limits

    fig.set_size_inches(25, 6)

#     plt.savefig("test.png", dpi=500)
    tikzplotlib.save(filename)

# 1-edit repairs
one_edits_300s = """
P@1=  300s: 0.471, 280s: 0.449, 260s: 0.442, 240s: 0.435, 220s: 0.428, 200s: 0.37, 180s: 0.355, 160s: 0.333, 140s: 0.312, 120s: 0.283, 100s: 0.268, 80s: 0.232, 60s: 0.203, 40s: 0.159, 20s: 0.116, 10s: 0.094
P@5=  300s: 0.710, 280s: 0.674, 260s: 0.667, 240s: 0.652, 220s: 0.638, 200s: 0.594, 180s: 0.565, 160s: 0.514, 140s: 0.493, 120s: 0.464, 100s: 0.42, 80s: 0.37, 60s: 0.304, 40s: 0.239, 20s: 0.145, 10s: 0.109
P@10= 300s: 0.768, 280s: 0.739, 260s: 0.732, 240s: 0.717, 220s: 0.703, 200s: 0.63, 180s: 0.594, 160s: 0.543, 140s: 0.507, 120s: 0.471, 100s: 0.42, 80s: 0.37, 60s: 0.304, 40s: 0.239, 20s: 0.145, 10s: 0.109
P@All= 300s: 0.783, 280s: 0.746, 260s: 0.739, 240s: 0.725, 220s: 0.71, 200s: 0.638, 180s: 0.594, 160s: 0.543, 140s: 0.507, 120s: 0.471, 100s: 0.42, 80s: 0.37, 60s: 0.304, 40s: 0.239, 20s: 0.145, 10s: 0.109
"""

# 2-edit repairs (384)
two_edits_300s = """
P@1=  300s: 0.048, 280s: 0.048, 260s: 0.032, 240s: 0.032, 220s: 0.032, 200s: 0.032, 180s: 0.048, 160s: 0.048, 140s: 0.065, 120s: 0.016, 100s: 0.016, 80s: 0.0, 60s: 0.0, 40s: 0.0, 20s: 0.016, 10s: 0.016
P@5=  300s: 0.129, 280s: 0.129, 260s: 0.113, 240s: 0.113, 220s: 0.129, 200s: 0.129, 180s: 0.113, 160s: 0.113, 140s: 0.129, 120s: 0.048, 100s: 0.065, 80s: 0.032, 60s: 0.032, 40s: 0.032, 20s: 0.048, 10s: 0.016
P@10= 300s: 0.258, 280s: 0.258, 260s: 0.242, 240s: 0.258, 220s: 0.258, 200s: 0.258, 180s: 0.226, 160s: 0.21, 140s: 0.21, 120s: 0.129, 100s: 0.113, 80s: 0.081, 60s: 0.081, 40s: 0.065, 20s: 0.048, 10s: 0.016
P@All= 300s: 0.419, 280s: 0.419, 260s: 0.387, 240s: 0.355, 220s: 0.355, 200s: 0.355, 180s: 0.306, 160s: 0.29, 140s: 0.274, 120s: 0.194, 100s: 0.177, 80s: 0.113, 60s: 0.081, 40s: 0.065, 20s: 0.048, 10s: 0.016
"""

# All repairs (1233)
all_edits_300s = """
P@1=  300s: 0.324, 280s: 0.318, 260s: 0.306, 240s: 0.301, 220s: 0.295, 200s: 0.272, 180s: 0.272, 160s: 0.254, 140s: 0.243, 120s: 0.208, 100s: 0.202, 80s: 0.179, 60s: 0.162, 40s: 0.127, 20s: 0.098, 10s: 0.081
P@5=  300s: 0.497, 280s: 0.486, 260s: 0.474, 240s: 0.468, 220s: 0.468, 200s: 0.462, 180s: 0.445, 160s: 0.41, 140s: 0.405, 120s: 0.358, 100s: 0.335, 80s: 0.295, 60s: 0.249, 40s: 0.202, 20s: 0.133, 10s: 0.092
P@10= 300s: 0.59, 280s: 0.584, 260s: 0.572, 240s: 0.572, 220s: 0.566, 200s: 0.538, 180s: 0.509, 160s: 0.468, 140s: 0.445, 120s: 0.393, 100s: 0.353, 80s: 0.312, 60s: 0.266, 40s: 0.214, 20s: 0.133, 10s: 0.092
P@All= 300s: 0.676, 280s: 0.665, 260s: 0.642, 240s: 0.624, 220s: 0.618, 200s: 0.59, 180s: 0.543, 160s: 0.503, 140s: 0.474, 120s: 0.422, 100s: 0.382, 80s: 0.329, 60s: 0.272, 40s: 0.22, 20s: 0.133, 10s: 0.092
"""

one_edits_10s = """
P@1=   10s: 0.585, 9s: 0.585, 8s: 0.585, 7s: 0.582, 6s: 0.582, 5s: 0.582, 4s: 0.574, 3s: 0.569, 2s: 0.569, 1s: 0.533
P@5=   10s: 0.895, 9s: 0.892, 8s: 0.89, 7s: 0.885, 6s: 0.885, 5s: 0.885, 4s: 0.874, 3s: 0.869, 2s: 0.862, 1s: 0.823
P@10=  10s: 0.941, 9s: 0.938, 8s: 0.936, 7s: 0.931, 6s: 0.928, 5s: 0.931, 4s: 0.921, 3s: 0.91, 2s: 0.903, 1s: 0.859
P@All= 10s: 0.979, 9s: 0.977, 8s: 0.974, 7s: 0.969, 6s: 0.967, 5s: 0.964, 4s: 0.954, 3s: 0.944, 2s: 0.936, 1s: 0.89
"""

two_edits_10s = """
P@1=   10s: 0.163, 9s: 0.163, 8s: 0.163, 7s: 0.159, 6s: 0.159, 5s: 0.149, 4s: 0.144, 3s: 0.139, 2s: 0.12, 1s: 0.111
P@5=   10s: 0.341, 9s: 0.337, 8s: 0.337, 7s: 0.327, 6s: 0.327, 5s: 0.317, 4s: 0.312, 3s: 0.312, 2s: 0.293, 1s: 0.255
P@10=  10s: 0.5, 9s: 0.49, 8s: 0.49, 7s: 0.481, 6s: 0.481, 5s: 0.462, 4s: 0.457, 3s: 0.457, 2s: 0.428, 1s: 0.365
P@All= 10s: 0.62, 9s: 0.611, 8s: 0.601, 7s: 0.591, 6s: 0.591, 5s: 0.562, 4s: 0.548, 3s: 0.534, 2s: 0.5, 1s: 0.423
"""

three_edits_10s = """
P@1=   10s: 0.031, 9s: 0.031, 8s: 0.031, 7s: 0.031, 6s: 0.031, 5s: 0.031, 4s: 0.023, 3s: 0.023, 2s: 0.023, 1s: 0.023
P@5=   10s: 0.054, 9s: 0.054, 8s: 0.062, 7s: 0.054, 6s: 0.047, 5s: 0.047, 4s: 0.039, 3s: 0.039, 2s: 0.039, 1s: 0.039
P@10=  10s: 0.078, 9s: 0.085, 8s: 0.085, 7s: 0.078, 6s: 0.07, 5s: 0.07, 4s: 0.062, 3s: 0.07, 2s: 0.07, 1s: 0.047
P@All= 10s: 0.333, 9s: 0.318, 8s: 0.318, 7s: 0.302, 6s: 0.264, 5s: 0.256, 4s: 0.24, 3s: 0.217, 2s: 0.209, 1s: 0.171
"""

all_edits_10s = """
P@1=   10s: 0.368, 9s: 0.368, 8s: 0.368, 7s: 0.365, 6s: 0.365, 5s: 0.363, 4s: 0.356, 3s: 0.351, 2s: 0.346, 1s: 0.324
P@5=   10s: 0.587, 9s: 0.584, 8s: 0.584, 7s: 0.579, 6s: 0.577, 5s: 0.575, 4s: 0.566, 3s: 0.563, 2s: 0.554, 1s: 0.522
P@10=  10s: 0.662, 9s: 0.66, 8s: 0.658, 7s: 0.653, 6s: 0.65, 5s: 0.646, 4s: 0.637, 3s: 0.633, 2s: 0.621, 1s: 0.575
P@All= 10s: 0.762, 9s: 0.755, 8s: 0.75, 7s: 0.743, 6s: 0.735, 5s: 0.725, 4s: 0.713, 3s: 0.699, 2s: 0.683, 1s: 0.629
"""

# conda deactivate && conda activate cstk
if __name__ == '__main__':
    pattern = r"(P@\w+)=((?:\s+\d+s:\s+\d+\.\d+,?)+)"
    matches = re.findall(pattern, all_edits_10s)

    data = []
    for match in matches:
        key = match[0].strip()
        values = re.findall(r"(\d+)s:\s+(\d+\.\d+)", match[1])
        data.append((key, {k: float(v) for k, v in values}))

    print(json.dumps(data))

    plot_data(data, "repair1-3_10s_plot.tex")
