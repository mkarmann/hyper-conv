"""
Plotting the training progress

This is not used in the blog post but might still be interesting to look at.

It was also used for debugging the training process, as the training with
hyper convolutions was not as stable as expected.
"""
import plotly.express as px
import pandas as pd


def load_csv_train_or_val(fname, load_training=False) -> pd.DataFrame:
    """
    Load a lightning_log csv file but only train or validation data

    :param fname: path to the file
    :param load_training: If true, load the training metrics, if false load the validation ones.
    :return: Dataframe of only train or validation data
    """
    loss = 'train_loss' if load_training else 'val_loss'
    df = pd.read_csv(fname, usecols=[loss, 'epoch', 'step'])
    df = df.dropna()
    df = df.groupby('epoch').mean().reset_index()
    df['loss'] = df[loss]
    del df[loss]
    return df


def load_csv(fname) -> pd.DataFrame:
    """
    Load a lightning_log csv file with train and validation

    :param fname: path to the file
    :return: Dataframe with train and validation set
    """
    df_train = load_csv_train_or_val(fname, True)
    df_train['set'] = 'train'
    df_val = load_csv_train_or_val(fname, False)
    df_val['set'] = 'val'
    return pd.concat([df_train, df_val])


def plot_multiple(files, names, colors):
    """
    Plot multiple runs in the same plot

    :param files: List of csv log file paths
    :param names: List of names for the plot for each csv file
    :param colors: List of colours for each csv file
    :return:
    """
    dataframes = []
    color_map = dict()
    for file, name, color in zip(files, names, colors):
        df = load_csv(file)
        df['name'] = name + ' ' + df['set']
        color_map[name + ' train'] = color
        color_map[name + ' val'] = color
        df['dashes'] = df['set']
        df['dashes'] = df['dashes'].replace('train', 'dash')
        df['dashes'] = df['dashes'].replace('val', 'solid')
        dataframes.append(df)

    df = pd.concat(dataframes)
    fig = px.line(
        df,
        x="epoch",
        y="loss",
        color='name',
        title='Training losses',
        log_y=True,
        color_discrete_map=color_map,
        line_dash='dashes',
        line_dash_map='identity'
    )
    fig.show()


if __name__ == '__main__':
    plot_multiple(
        files=['lightning_logs/version_normal/metrics.csv', 'lightning_logs/version_hyper/metrics.csv'],
        names=['normal', 'hyperConv'],
        colors=['slategray', px.colors.qualitative.Plotly[0]]
    )
