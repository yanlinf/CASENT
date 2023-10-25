import streamlit as st
import argparse
import pandas as pd
import re
from casent.entity_typing_t5 import *

MODEL_CHECKPOINT_MAPPING = {
    'casent_t5_large': 'yanlinf/casent-large',
}

MODEL_DEVICE_MAPPING = {
    'casent_t5_large': 'cuda:0',
}

EXAMPLE_INPUTS = [
    'Current figures say <M> 203 miners </M> were killed , and another 22 were injured with 13 miners still missing .',
    'Efforts to rescue 10 miners trapped <M> in a collapsed and flooded coal mine </M> in northern Mexico intensified Thursday with hundreds of people',
    '<M>This case, manufacturer control number 2014-0128812</M> is a report from a solicited program GSI Sponsored Market Research referring to a 63 year-old male patient.',
    'A court in Jerusalem sentenced <M> a Palestinian </M> to 16 life terms for forcing a bus off a cliff July 6 , killing 16 people , Israeli radio reported .',
]


@st.cache_resource()
def load_predictors(run_on_cpu: bool = False):
    predictors = {}
    for name, path in MODEL_CHECKPOINT_MAPPING.items():
        predictors[name] = T5ForEntityTypingPredictor.from_pretrained(path)
        if not run_on_cpu:
            predictors[name].to(torch.device(MODEL_DEVICE_MAPPING[name]))
    return predictors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    print(args)
    print()

    st.set_page_config(
        page_title='CASENT',
        page_icon='🎈',
    )
    st.title('🎈 CASENT')

    st.write('Choose your models:')
    model_options = []
    for model in MODEL_CHECKPOINT_MAPPING:
        model_options.append(st.checkbox(model, True))

    example = st.selectbox('Example inputs', ['Select a sentence'] + EXAMPLE_INPUTS)

    doc = st.text_area(
        'Paste your text below (max 128 words)',
        '' if example == 'Select a sentence' else example,
        height=150,
    ).strip()

    predictors = load_predictors(run_on_cpu=args.cpu)

    default_threshold = 0.2
    threshold = st.slider('Threshold (only applicable to casent models):', 0., 1.,
                          default_threshold)

    show_uncalibrated = st.checkbox('Show uncalibrated scores', False)

    for name, predictor in predictors.items():
        threshold_length = predictor.model.calibration.threshold.size(0)
        predictor.model.calibration.update_params(
            threshold=torch.full((threshold_length,), threshold)
        )

    run_button = st.button(label='✨ Run Model')

    n_words = len(re.findall(r'\w+', doc))
    if n_words > 128:
        st.warning(f'⚠️ Your text contains more than 128 words.')

    if len(doc) > 0 and doc.count('<M>') > 1 or doc.count('</M>') > 1:
        st.warning('⚠️ More than one entity mentions are marked.')

    if not run_button or len(doc) == 0:
        st.stop()

    if not ('<M>' in doc and '</M>' in doc and doc.index('<M>') < doc.index('</M>')):
        st.error('❌ Entity mention needs to be marked with `<M>` and `</M>`.')
        st.stop()

    st.markdown('### Model Output')

    selected_models = [name for name, option in zip(MODEL_CHECKPOINT_MAPPING, model_options)
                       if option]

    for model_name in selected_models:
        predictor: T5ForEntityTypingPredictor = predictors[model_name]
        pred, = predictor.predict_raw([doc])
        type2score = {t: s for t, s in zip(pred.types, pred.scores)}
        type_score_pairs = sorted(type2score.items(), key=lambda t: -t[1])
        st.write(f'`{model_name}`')
        st.table(pd.DataFrame(type_score_pairs, columns=('Predicted label', 'Score')))

        if show_uncalibrated:
            st.write(f'`{model_name}` uncalibrated scores:')
            pred, = predictor.predict_raw([doc], do_calibration=False)
            type2score = {t: s for t, s in zip(pred.types, pred.scores)}
            type_score_pairs = sorted(type2score.items(), key=lambda t: -t[1])
            st.table(pd.DataFrame(type_score_pairs, columns=('Predicted label', 'Score')))


if __name__ == '__main__':
    main()
