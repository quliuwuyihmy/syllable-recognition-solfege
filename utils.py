import numpy as np
from parameters import hopsize_t


def figure_plot(plt,
                mfcc,
                onset_label):
    # plot Error analysis figures
    plt.figure(figsize=(16, 2))
    # class weight
    ax1 = plt.subplot(211)
    y = np.arange(0, 36)
    x = np.arange(0, mfcc.shape[0]) * hopsize_t
    plt.pcolormesh(x, y, np.transpose(mfcc))

    ax1.set_ylabel('Mfcc', fontsize=12)
    ax1.get_xaxis().set_visible(False)
    ax1.axis('tight')

    ax2 = plt.subplot(212, sharex=ax1)
    for onset, label in onset_label:
        plt.axvline(onset * hopsize_t, color='r', linewidth=2)

    ax2.set_ylabel('ODF syllable', fontsize=12)
    ax2.axis('tight')
