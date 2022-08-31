from typing import Union

import numpy as np
import pandas as pd

GT_ANNO_SYMBOLS = {
    1: u'\u2713',
    0: u'\u2717',
    None: "?"
}


def pad_image(image):
    image = np.array(image)
    height, width = image.shape[:2]
    max_dim = max(height, width)
    height_pad = (max_dim - height) // 2
    width_pad = (max_dim - width) // 2
    try:
        image = np.pad(image, ((height_pad, height_pad), (width_pad, width_pad), (0, 0)), mode='constant',
                       constant_values=255)
    except:
        pass
    return image


def display_naked_image(ima, ax, title):
    if ima.shape[0] > 0:
        ax.imshow(ima, vmin=0, vmax=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def get_viewable_fontsize(n_ticks: int):
    if n_ticks > 32:
        return 2
    elif n_ticks > 22:
        return 4
    else:
        return 5


def display_dressed_image(ima: Union["np.ndarray", "pd.DataFrame"], ax, title=None, ticks=None,
                          titlefontsize=None, ticklabel_colors=None, gt_annotation=None, xticks=False, **kwargs):
    if isinstance(ima, pd.DataFrame) and not ticks:
        ticks = ima.index.values.tolist()

    if ima.shape[0] > 0:
        ax.imshow(ima, **kwargs)

    if ticks is not None:
        ax.set_xticks(range(len(ticks)))
        ax.set_yticks(range(len(ticks)))
        fontsize = get_viewable_fontsize(len(ticks))

        if xticks:
            ax.set_xticklabels(ticks, rotation=45, fontsize=fontsize)
        else:
            ax.set_xticklabels([])

        ax.set_yticklabels(ticks, fontsize=fontsize)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
    ax.set_title(title, fontsize=titlefontsize)

    if ticklabel_colors is not None:
        for label, i in zip(ax.yaxis.get_ticklabels(), ticklabel_colors):
            if i > -1:
                label.set_color(f"C{i}")

    if gt_annotation is not None:
        new_labels = []
        for label in ax.get_yticklabels():
            label_text = label.get_text()
            is_gt = gt_annotation.get(label_text)
            suffix = f"[{GT_ANNO_SYMBOLS.get(is_gt) or ''}]"
            new_labels.append(f"{label_text} {suffix}")
        ax.set_yticklabels(new_labels)
