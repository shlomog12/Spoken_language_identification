

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np




cnt = {'ab': 59, 'ar': 10000, 'as': 436, 'br': 1004, 'ca': 10000, 'cnh': 310, 'cs': 7558, 'cv': 1023, 'cy': 9899, 'de': 10000, 'dv': 4612, 'el': 2626, 'en': 10000, 'eo': 10000, 'es': 10000, 'et': 6979, 'eu': 10000, 'fa': 6973, 'fi': 531, 'fr': 10000, 'fy-NL': 5842, 'ga-IE': 386, 'hi': 136, 'hsb': 1849, 'hu': 4385, 'ia': 2230, 'id': 1734, 'it': 10000, 'ja': 977, 'ka': 1887, 'kab': 10000, 'ky': 2696, 'lg': 1629, 'lt': 1195, 'lv': 1439, 'mn': 3367, 'mt': 2476, 'nl': 10000, 'or': 599, 'pa-IN': 261, 'pl': 9271, 'pt': 7652, 'rm-sursilv': 2461, 'rm-vallader': 1254, 'ro': 3292, 'ru': 10000, 'rw': 10000, 'sah': 2715, 'sl': 2027, 'sv-SE': 1384, 'ta': 2198, 'th': 3176, 'tr': 1758, 'tt': 9692, 'uk': 5915, 'vi': 162, 'vot': 6, 'zh-CN': 10000, 'zh-HK': 5658, 'zh-TW': 2364}

cnt_sorted ={k: v for k, v in sorted(cnt.items(), key=lambda item: item[1])}







import pandas as pd
import matplotlib.pyplot as plt

# Bring some raw data.
frequencies = cnt_sorted.values()

# In my original code I create a series and run on that,
# so for consistency I create a series from the list.
freq_series = pd.Series(frequencies)
x_labels = cnt_sorted.keys()
# Plot the figure.
plt.figure(figsize=(30, 14))
ax = freq_series.plot(kind='bar')
ax.set_title('count languages')
ax.set_xlabel('Amount ($)')
ax.set_ylabel('count')
ax.set_xticklabels(x_labels)


def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        if y_value == 10000:
            label = '10k'
        else:
            label = "{:.0f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.


# Call the function above. All the magic happens there.
add_value_labels(ax)

plt.savefig("image55.png")
