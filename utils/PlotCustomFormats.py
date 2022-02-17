import numpy as np
import matplotlib
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator

def format_regular_coordinate_axis(axs):
  for ax in axs:
    if ax != None:
      ax.title.set_fontsize(24)

      for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(24)

      ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: ('' if x == 0 else '{:,.2E}'.format(x))))
      ax.yaxis.set_major_locator(MaxNLocator(4)) 

def format_polar_coordinate_axis(axs):
  for ax in axs:
    ax.set_rticks([0.2, 0.5, 0.8])