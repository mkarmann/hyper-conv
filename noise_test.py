"""
Load the hyper and normal model and compare their performance when noise is added

A previous run is already stored in out/pepper_noise.csv

So this one will be loaded by default to make plots faster
"""
import pandas as pd

from main import get_train_and_val_data, MNISTClassifier, evaluate_model_accuracy
import torch
import plotly.express as px
import plotly.io as pio
pio.templates.default = "none"


def add_noise(data, noise_type, amount):
    out = []

    for d in data:
        x, y = d
        if noise_type == 'pepper':
            mask = torch.rand(x.shape) > (1.0 - amount)
            binary = (torch.rand(x.shape) > 0.5).float()
            x = x + mask * binary
            x = x - mask * (1.0 - binary)
            x = torch.clip(x, 0.0, 1.0)
        elif noise_type == 'gauss':
            # amount is then std
            noise = torch.randn(x.shape) * amount
            x = x + noise
            x = torch.clip(x, 0.0, 1.0)
        elif noise_type == 'speckle':
            # amount is then std
            noise = torch.randn(x.shape) * amount + 1.0
            x = x * noise
            x = torch.clip(x, 0.0, 1.0)

        # fig = px.imshow(x.cpu().numpy()[0])
        # fig.show()
        # exit()
        out.append((x, y))
    return out


if __name__ == '__main__':

    # Load the precomputed out/noise.csv file
    load_csv = True

    # If True and the csv is not loaded
    # store the test results in out/noise.csv file
    overwrite_csv_if_not_loading = False

    if load_csv:
        df = pd.read_csv('out/noise.csv')
    else:
        train_data, val_data = get_train_and_val_data()
        model_normal = MNISTClassifier.load_from_checkpoint('lightning_logs/version_normal/best.ckpt')
        model_hyper = MNISTClassifier.load_from_checkpoint('lightning_logs/version_hyper/best.ckpt')

        data = []
        for i in range(20):
            amount = i / 19.0

            for noise_type, noise_unit in [('pepper', 'amount'), ('gauss', 'std')]:
                noisy_val = add_noise(val_data, noise_type, amount)
                for model, model_type in [(model_normal, 'normal'), (model_hyper, 'hyper')]:
                    acc = evaluate_model_accuracy(model, noisy_val)
                    data.append({
                        'noiseType': noise_type,
                        'amount': amount,
                        'accuracy': acc,
                        'modelType': model_type,
                        'noiseUnit': noise_unit
                    })
        df = pd.DataFrame(data)
        if overwrite_csv_if_not_loading:
            df.to_csv('out/noise.csv', index=False)

    df['Convolution Type'] = df['modelType']

    fig = px.line(
        df[df['noiseType'] == 'pepper'],
        x='amount',
        y='accuracy',
        color='Convolution Type',
        color_discrete_map={'normal': 'blue', 'hyper': 'green'},
        title='Pepper Noise'
    )
    fig.update_xaxes(title_text="Amount")
    fig.update_yaxes(title_text="Accuracy")
    fig.show()

    fig = px.line(
        df[df['noiseType'] == 'gauss'],
        x='amount',
        y='accuracy',
        color='Convolution Type',
        color_discrete_map={'normal': 'blue', 'hyper': 'green'},
        title='Gauss Noise'
    )
    fig.update_xaxes(title_text="Std")
    fig.update_yaxes(title_text="Accuracy")
    fig.show()
