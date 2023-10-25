import numpy as np
import matplotlib.pyplot as plt


def plot_beam_size_impact_emnlp23(
        beam_sizes: list,
        f1_scores: list,
        ece_scores: list,
        output_path: str,
):
    fig = plt.Figure(figsize=(4, 3.5))

    ax1 = fig.add_subplot(1, 1, 1)

    # color = '#c5e0b4'
    color = '#a4c98d'
    line1 = ax1.plot(beam_sizes, f1_scores, color=color, label=f'F1')
    ax1.scatter(beam_sizes, f1_scores, color=color, zorder=3)

    ax1.legend()
    ax1.set_xlim(min(beam_sizes), max(beam_sizes))
    ax1.set_ylim(min(f1_scores), max(f1_scores))
    # ax1.tick_params(axis="y", direction="in", pad=6)
    # ax1.tick_params(axis="x", direction="in", pad=6)
    ax1.set_xticks(beam_sizes)
    # ax1.set_xticks(np.arange(4, 44, 8))
    ax1.set_yticks(np.linspace(0.49, 0.52, 4))
    # ax1.set_yticklabels([])
    ax1.set_xlabel('Beam size')
    ax1.set_ylabel('F1')

    ax1.grid(color='#d6d6d6')
    ax1.spines['bottom'].set_color('#6e6e6e')
    ax1.spines['top'].set_color('#6e6e6e')
    ax1.spines['right'].set_color('#6e6e6e')
    ax1.spines['left'].set_color('#6e6e6e')
    ax1.tick_params(length=0)

    ax2 = ax1.twinx()
    color2 = '#f8cbad'
    color2 = '#bf75c7'
    color2 = '#d491db'
    ece_scores = [x * 100 for x in ece_scores]
    line2 = ax2.plot(beam_sizes, ece_scores, color=color2, label=f'Calibration error')
    ax2.scatter(beam_sizes, ece_scores, color=color2, zorder=3)

    ax2.set_ylabel('ECE (%)')
    # ax2.set_yticks(np.linspace(0, 4, 6))
    ax2.set_yticks(np.linspace(0, 3.2, 5))

    # ax2.grid(color='#d6d6d6')
    # ax2.spines['bottom'].set_color('#6e6e6e')
    # ax2.spines['top'].set_color('#6e6e6e')
    # ax2.spines['right'].set_color('#6e6e6e')
    # ax2.spines['left'].set_color('#6e6e6e')
    # ax2.tick_params(length=0)

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')

    fig.savefig(output_path, bbox_inches='tight')
    fig.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')


def main():
    plot_beam_size_impact_emnlp23(
        [4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
        [0.4948, 0.5146, 0.5130, 0.5141, 0.5126, 0.5131, 0.5112, 0.5114, 0.5116, 0.5115],
        # [0.0126, 0.0238, 0.0181, 0.0145, 0.0106, 0.0067, 0.0039, 0.0054, 0.0048, 0.0048],
        [0.0124, 0.0214, 0.0196, 0.0134, 0.0144, 0.0116, 0.0139, 0.0162, 0.0134, 0.0138],
        'tmp/beam_size_impact_emnlp23.png'
    )


if __name__ == '__main__':
    main()
