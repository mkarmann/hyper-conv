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


if __name__ == '__main__':
    df = load_csv('lightning_logs/version_normal/metrics.csv')
    fig = px.line(df, x="epoch", y="loss", color='set', title='Life expectancy in Canada', log_y=True)
    fig.show()
