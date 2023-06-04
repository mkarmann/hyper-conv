import plotly.express as px
import torch

from main import MNISTClassifier

if __name__ == '__main__':
    with torch.no_grad():
        model = MNISTClassifier.load_from_checkpoint('lightning_logs/version_hyper/best.ckpt')
        weights = model.main[2].main[2].get_weights()
        out_ch, in_ch, h, w = weights.shape
        w_list = weights.reshape(out_ch * in_ch, h, w).cpu().numpy()
        fig = px.imshow(
            w_list,
            facet_col=0,
            facet_col_wrap=in_ch,
            facet_row_spacing=0.02,
            facet_col_spacing=0.02,
            color_continuous_scale='blues'
        )

        for anno in fig['layout']['annotations']:
            anno['text'] = ''

        fig.show()
