# !pip install datasets
import datasets
from tqdm import tqdm
from datasets import list_datasets, load_dataset, list_metrics, load_metric
import pandas as pd
import matplotlib.pyplot as plt

counter = {'ab': 22, 'ar': 14227, 'as': 270, 'br': 2780, 'ca': 285584, 'cnh': 807, 'cs': 5655, 'cv': 931, 'cy': 6839, 'de': 246525, 'dv': 2680, 'el': 2316, 'en': 1001, 'eo': 1001, 'es': 1001, 'et': 1001, 'eu': 1001, 'fa': 1001, 'fi': 460, 'fr': 1001, 'fy-NL': 1001, 'ga-IE': 541, 'hi': 157, 'hsb': 808, 'hu': 1001, 'ia': 1001, 'id': 1001, 'it': 1001, 'ja': 722, 'ka': 1001, 'kab': 1001, 'ky': 1001, 'lg': 1001, 'lt': 931, 'lv': 1001, 'mn': 1001, 'mt': 1001, 'nl': 1001, 'or': 388, 'pa-IN': 211, 'pl': 1001, 'pt': 1001, 'rm-sursilv': 1001, 'rm-vallader': 574, 'ro': 1001, 'ru': 1001, 'rw': 1001, 'sah': 1001, 'sl': 1001, 'sv-SE': 1001, 'ta': 1001, 'th': 1001, 'tr': 1001, 'tt': 1001, 'uk': 1001, 'vi': 221, 'vot': 3, 'zh-CN': 1001, 'zh-HK': 1001, 'zh-TW': 1001}
languages = []
for k,v in counter.items():
  languages.append(k)



def sum_data_by_lang(lang):
  window = 48000
  step = 32000
  dataset = load_dataset("common_voice", lang, split="train", streaming=True)
  dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))
  dataset_iter = iter(dataset)
  data = []
  k = 1
  for sample in dataset_iter:
    array = sample["audio"]["array"]
    array = array[44:]
    pointer = 0
    while pointer + window < len(array):
      curr = array[pointer:pointer+window]
      data.append(curr)
      k += 1
      if k>= 10000:
        return k
      pointer += step

  return k



counter = {}
for lang in tqdm(languages):
  s = sum_data_by_lang(lang)
  print(f'{lang} = {s}')
  counter[lang] = s


cnt_sorted ={k: v for k, v in sorted(counter.items(), key=lambda item: item[1])}



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