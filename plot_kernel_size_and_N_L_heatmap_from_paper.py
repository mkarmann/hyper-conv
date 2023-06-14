"""
This script is used to plot a heatmap of data from the paper.

It does not use actual data from the MNIST training of this repository
but I wanted to include it, as it was used to create a plot for the presentation.
"""
import plotly.express as px


if __name__ == '__main__':
    # Plotting values form the papers table 2
    liver_lesion = [
        [0.604, 0.627, 0.648],
        [0.692, 0.728, 0.705],
        [0.683, 0.704, 0.717]
    ]

    # And table 4
    ms_lesion = [
        [0.622, 0.625, 0.617],
        [0.648, 0.655, 0.651],
        [0.634, 0.644, 0.646]
    ]

    fig = px.imshow(
        ms_lesion,
        color_continuous_scale='blues',
        text_auto=True
    )
    fig.update_layout(
        xaxis = dict(
            tickmode='array',
            tickvals=[0, 1, 2],
            ticktext=['2', '4', '8']
        ),
        yaxis = dict(
            tickmode='array',
            tickvals=[0, 1, 2],
            ticktext=['3x3', '5x5', '7x7']
        ),
        xaxis_title = "N<sub>L</sub>",
        yaxis_title = "Kernel Size",
    )
    fig.show()
