import io
import base64
import math

from matplotlib import pyplot as plt
from matplotlib import cm

# renders a mel spectogram as a PNG
def spectogram(spectogram):
    fig, spec_ax = plt.subplots(1, 1, figsize=(10, 2.25), facecolor="#F0F0F0")
    fig.subplots_adjust(left=0.00, bottom=0.00, right=1, top=1)

    if spectogram is not None:
        spec_ax.imshow(spectogram, aspect="auto", interpolation="none")

    spec_ax.set_xticks([])
    spec_ax.set_yticks([])
    spec_ax.figure.canvas.draw()
    # fig.savefig('test.png') # DEBUG code

    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    spectogram_bytes = buffer.getvalue()
    img_str = "data:image/png;base64," + base64.b64encode(spectogram_bytes).decode()
    return img_str

# renders a speaker embedding as a PNG
def embedding(embed):
    fig, spec_ax = plt.subplots(1, 1, figsize=(2.25, 2.25), facecolor="#F0F0F0")
    fig.subplots_adjust(left=0.00, bottom=0.00, right=1, top=1)

    if embed is not None:
        plot_embedding_as_heatmap(embed, spec_ax)

    spec_ax.set_xticks([])
    spec_ax.set_yticks([])
    spec_ax.figure.canvas.draw()
    # fig.savefig('test_embed.png') # DEBUG code

    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    spectogram_bytes = buffer.getvalue()
    img_str = "data:image/png;base64," + base64.b64encode(spectogram_bytes).decode()
    return img_str


def plot_embedding_as_heatmap(embed, ax):
    height = int(len(embed)/16)
    shape = (height, -1)
    embed = embed.reshape(shape)
    cmap = cm.get_cmap()
    _ = ax.imshow(embed, cmap=cmap, aspect="auto", interpolation="none")