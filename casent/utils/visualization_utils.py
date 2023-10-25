COLORS = {
    'blue': '#5b9bd5',
    'sandstone': '#dba58c',
    'mint': '#99cccc',
    'pink': '#ff8b8b',
    'lavender': '#b98fc4'
}


def set_axis_style(ax):
    ax.spines['bottom'].set_color('lightgray')
    ax.spines['top'].set_color('lightgray')
    ax.spines['right'].set_color('lightgray')
    ax.spines['left'].set_color('lightgray')
    ax.tick_params(length=0)
