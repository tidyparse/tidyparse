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
    second_mapping = {f"{i}": i for i in range(60, 0, -5)}

    ind = np.arange(len(second_mapping))  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    bottom = np.zeros(len(second_mapping))
    colors = cm.rainbow(np.linspace(0, 1, len(data)))  # Generate as many colors as there are rows

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

    tikzplotlib.save(filename)

# 1-edit repairs
one_edit = """
P@1=  60s: 0.29, 55s: 0.274, 50s: 0.257, 45s: 0.253, 40s: 0.231, 35s: 0.203, 30s: 0.182, 25s: 0.146, 20s: 0.123, 15s: 0.096, 10s: 0.068, 5s: 0.039
P@5=  60s: 0.444, 55s: 0.411, 50s: 0.374, 45s: 0.354, 40s: 0.32, 35s: 0.288, 30s: 0.254, 25s: 0.203, 20s: 0.16, 15s: 0.112, 10s: 0.076, 5s: 0.04
P@10= 60s: 0.468, 55s: 0.436, 50s: 0.399, 45s: 0.365, 40s: 0.328, 35s: 0.291, 30s: 0.256, 25s: 0.203, 20s: 0.16, 15s: 0.112, 10s: 0.076, 5s: 0.04
P@15= 60s: 0.478, 55s: 0.439, 50s: 0.401, 45s: 0.367, 40s: 0.33, 35s: 0.293, 30s: 0.257, 25s: 0.203, 20s: 0.16, 15s: 0.112, 10s: 0.076, 5s: 0.04
P@20= 60s: 0.478, 55s: 0.439, 50s: 0.401, 45s: 0.368, 40s: 0.33, 35s: 0.294, 30s: 0.257, 25s: 0.203, 20s: 0.16, 15s: 0.112, 10s: 0.076, 5s: 0.04
P@1k= 60s: 0.482, 55s: 0.444, 50s: 0.405, 45s: 0.371, 40s: 0.333, 35s: 0.296, 30s: 0.259, 25s: 0.205, 20s: 0.16, 15s: 0.112, 10s: 0.076, 5s: 0.04
"""

# 2-edit repairs (384)
two_edits = """
P@1=  60s: 0.016, 55s: 0.018, 50s: 0.016, 45s: 0.016, 40s: 0.016, 35s: 0.013, 30s: 0.01, 25s: 0.01, 20s: 0.008, 15s: 0.008, 10s: 0.01, 5s: 0.008
P@5=  60s: 0.044, 55s: 0.039, 50s: 0.039, 45s: 0.039, 40s: 0.034, 35s: 0.029, 30s: 0.029, 25s: 0.023, 20s: 0.021, 15s: 0.021, 10s: 0.013, 5s: 0.008
P@10= 60s: 0.081, 55s: 0.068, 50s: 0.062, 45s: 0.055, 40s: 0.049, 35s: 0.047, 30s: 0.039, 25s: 0.029, 20s: 0.031, 15s: 0.029, 10s: 0.018, 5s: 0.008
P@15= 60s: 0.089, 55s: 0.076, 50s: 0.07, 45s: 0.065, 40s: 0.057, 35s: 0.052, 30s: 0.044, 25s: 0.036, 20s: 0.034, 15s: 0.029, 10s: 0.018, 5s: 0.008
P@20= 60s: 0.094, 55s: 0.083, 50s: 0.076, 45s: 0.073, 40s: 0.065, 35s: 0.06, 30s: 0.047, 25s: 0.036, 20s: 0.034, 15s: 0.029, 10s: 0.018, 5s: 0.008
P@1k= 60s: 0.12, 55s: 0.099, 50s: 0.086, 45s: 0.078, 40s: 0.07, 35s: 0.062, 30s: 0.049, 25s: 0.039, 20s: 0.036, 15s: 0.029, 10s: 0.018, 5s: 0.008
"""

# 3-edit repairs (200)
three_edits = """
P@1=  60s: 0.005, 55s: 0.005, 50s: 0.005, 45s: 0.0, 40s: 0.0, 35s: 0.0, 30s: 0.0, 25s: 0.0, 20s: 0.0, 15s: 0.0, 10s: 0.0, 5s: 0.0
P@5=  60s: 0.005, 55s: 0.005, 50s: 0.005, 45s: 0.0, 40s: 0.005, 35s: 0.005, 30s: 0.0, 25s: 0.0, 20s: 0.0, 15s: 0.0, 10s: 0.0, 5s: 0.0
P@10= 60s: 0.01, 55s: 0.01, 50s: 0.01, 45s: 0.005, 40s: 0.005, 35s: 0.005, 30s: 0.0, 25s: 0.0, 20s: 0.0, 15s: 0.0, 10s: 0.0, 5s: 0.0
P@15= 60s: 0.01, 55s: 0.01, 50s: 0.01, 45s: 0.005, 40s: 0.005, 35s: 0.005, 30s: 0.0, 25s: 0.0, 20s: 0.0, 15s: 0.0, 10s: 0.0, 5s: 0.0
P@20= 60s: 0.01, 55s: 0.01, 50s: 0.01, 45s: 0.005, 40s: 0.005, 35s: 0.005, 30s: 0.0, 25s: 0.0, 20s: 0.0, 15s: 0.0, 10s: 0.0, 5s: 0.0
P@1k= 60s: 0.01, 55s: 0.01, 50s: 0.01, 45s: 0.005, 40s: 0.005, 35s: 0.005, 30s: 0.0, 25s: 0.0, 20s: 0.0, 15s: 0.0, 10s: 0.0, 5s: 0.0
"""

# All repairs (1233)
all_edits = """
P@1=  60s: 0.158, 55s: 0.151, 50s: 0.141, 45s: 0.138, 40s: 0.127, 35s: 0.111, 30s: 0.099, 25s: 0.08, 20s: 0.067, 15s: 0.053, 10s: 0.039, 5s: 0.023
P@5=  60s: 0.248, 55s: 0.23, 50s: 0.21, 45s: 0.199, 40s: 0.18, 35s: 0.161, 30s: 0.143, 25s: 0.114, 20s: 0.091, 15s: 0.066, 10s: 0.044, 5s: 0.024
P@10= 60s: 0.273, 55s: 0.252, 50s: 0.231, 45s: 0.21, 40s: 0.189, 35s: 0.169, 30s: 0.147, 25s: 0.116, 20s: 0.094, 15s: 0.068, 10s: 0.045, 5s: 0.024
P@15= 60s: 0.281, 55s: 0.256, 50s: 0.234, 45s: 0.214, 40s: 0.192, 35s: 0.171, 30s: 0.149, 25s: 0.118, 20s: 0.095, 15s: 0.068, 10s: 0.045, 5s: 0.024
P@20= 60s: 0.282, 55s: 0.259, 50s: 0.236, 45s: 0.217, 40s: 0.195, 35s: 0.174, 30s: 0.15, 25s: 0.118, 20s: 0.095, 15s: 0.068, 10s: 0.045, 5s: 0.024
P@1k= 60s: 0.293, 55s: 0.266, 50s: 0.242, 45s: 0.221, 40s: 0.198, 35s: 0.176, 30s: 0.152, 25s: 0.12, 20s: 0.096, 15s: 0.068, 10s: 0.045, 5s: 0.024
"""

if __name__ == '__main__':
    pattern = r"(P@\w+)=((?:\s+\d+s:\s+\d+\.\d+,?)+)"
    matches = re.findall(pattern, all_edits)

    data = []
    for match in matches:
        key = match[0].strip()
        values = re.findall(r"(\d+)s:\s+(\d+\.\d+)", match[1])
        data.append((key, {k: float(v) for k, v in values}))

    print(json.dumps(data))

    # data = [
    #     ("P@1", {"60": 0.446, "55": 0.428, "50": 0.406, "45": 0.364, "40": 0.342, "35": 0.327, "30": 0.302, "25": 0.265,
    #              "20": 0.218, "15": 0.173, "10": 0.119, "5": 0.079}),
    #     ("P@5", {"60": 0.649, "55": 0.626, "50": 0.599, "45": 0.532, "40": 0.505, "35": 0.475, "30": 0.436, "25": 0.366,
    #              "20": 0.312, "15": 0.248, "10": 0.161, "5": 0.092}),
    #     ("P@10", {"60": 0.715, "55": 0.683, "50": 0.646, "45": 0.574, "40": 0.542, "35": 0.5, "30": 0.458, "25": 0.384,
    #               "20": 0.319, "15": 0.252, "10": 0.161, "5": 0.094}),
    #     ("P@20", {"60": 0.75, "55": 0.713, "50": 0.668, "45": 0.594, "40": 0.559, "35": 0.51, "30": 0.463, "25": 0.389,
    #               "20": 0.319, "15": 0.257, "10": 0.166, "5": 0.097}),
    #     ("P@1k",
    #      {"60": 0.787, "55": 0.745, "50": 0.698, "45": 0.619, "40": 0.582, "35": 0.525, "30": 0.475, "25": 0.401,
    #       "20": 0.329, "15": 0.262, "10": 0.168, "5": 0.097})
    # ]
    # print(data)
    plot_data(data, "repair1-3_plot.tex")
