import plotly.express as px
import pandas as pd


def load_csv_train_or_val(fname, load_training=False):
    loss = 'train_loss' if load_training else 'val_loss'
    df = pd.read_csv(fname, usecols=[loss, 'epoch', 'step'])
    df = df.dropna()
    df = df.groupby('epoch').mean().reset_index()
    df['loss'] = df[loss]
    del df[loss]
    return df


def load_csv(fname):
    df_train = load_csv_train_or_val(fname, True)
    df_train['set'] = 'train'
    df_val = load_csv_train_or_val(fname, False)
    df_val['set'] = 'val'
    return pd.concat([df_train, df_val])


def plot_multiple(files, names, colors):
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
