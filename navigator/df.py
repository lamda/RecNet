# -*- coding: utf-8 -*-

from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
import seaborn as sns


if __name__ == '__main__':
    sns.set(style="ticks")
    hatches = ['----', '/', 'xxx', '///', '---']
    colors = ['#FFA500', '#FF0000', '#0000FF', '#05FF05']
    df = pd.read_pickle('df.obj')
    df = df[df['strategy'] == 'title']

    fg = sns.factorplot(x='data_set', y='val', hue='rec_type', size=10,
                        data=df[df['is_random'] == True],
                        kind='bar',
                        palette=['#FFA500', '#FF0000', '#0000FF',
                                 '#05FF05'],
                        hue_order=['rbar', 'rb', 'rbiw', 'rbmf'],
                        )
    for bidx, bar in enumerate(fg.ax.patches):
        # bar.set_fill(False)
        # bar.set_hatch(hatches[bidx % 4])
        bar.set_edgecolor(colors[int(bidx % 4)])
        bar.set_alpha(0.75)
    plt.title('random')

    fg = sns.factorplot(x='data_set', y='val', hue='rec_type', size=10,
                        data=df[(df['is_random'] == True) & (df['N'] == 5)],
                        kind='bar',
                        palette=['#FFA500', '#FF0000', '#0000FF',
                                 '#05FF05'],
                        hue_order=['rbar', 'rb', 'rbiw', 'rbmf'],
                        )
    for bidx, bar in enumerate(fg.ax.patches):
        # bar.set_fill(False)
        # bar.set_hatch(hatches[bidx % 4])
        bar.set_edgecolor(colors[int(bidx % 4)])
        bar.set_alpha(0.75)
    plt.title('rating-based')

    # pdb.set_trace()
    plt.show()
