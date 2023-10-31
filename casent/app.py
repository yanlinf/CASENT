import streamlit as st
import argparse
import pandas as pd
import re
import stanza
from casent.entity_typing_t5 import *

MODEL_CHECKPOINT_MAPPING = {
    'casent_t5_large': 'yanlinf/casent-large',
}

MODEL_DEVICE_MAPPING = {
    'casent_t5_large': 'cuda:0',
}

EXAMPLE_INPUTS = [
    'Current figures say <M> 203 miners </M> were killed , and another 22 were injured with 13 miners still missing .',
    'Efforts to rescue 10 miners trapped in <M> a collapsed and flooded coal mine </M> in northern Mexico intensified Thursday with hundreds of people',
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


def entity_typing_demo(args):
    st.write('')
    st.write('')
    st.write('Choose your models:')
    model_options = []
    for model in MODEL_CHECKPOINT_MAPPING:
        model_options.append(st.checkbox(f'`{model}`', True))

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
        return
        # st.stop()

    if not ('<M>' in doc and '</M>' in doc and doc.index('<M>') < doc.index('</M>')):
        st.error('❌ Entity mention needs to be marked with `<M>` and `</M>`.')
        return
        # st.stop()

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


@st.cache_resource()
def load_stanza(use_gpu: bool = False):
    return stanza.Pipeline(
        lang="en", processors="tokenize,pos,constituency", use_gpu=use_gpu
    )


def entity_extraction_demo(args):
    st.write('')
    st.write('')

    examples = [
        'The Tenerife airport disaster occurred on March 27, 1977, when two Boeing 747 passenger jets collided on the runway at Los Rodeos Airport (now Tenerife North Airport) on the Spanish island of Tenerife. The collision occurred when KLM Flight 4805 initiated its takeoff run during dense fog while Pan Am Flight 1736 was still on the runway.'
    ]

    col1, col2 = st.columns([0.3, 0.7])

    with col1:
        target_type = st.selectbox('Target entity type', ['aircraft', 'event', 'airport', 'location', 'person'],
                                   key='extraction_target_type_')

    with col2:
        example = st.selectbox('Example documents', ['Select a document'] + examples, key='extraction_example_')

    doc = st.text_area(
        'Paste your text below (max 128 words)',
        '' if example == 'Select a document' else example,
        height=150,
        key='extraction_doc_',
    ).strip()

    predictors = load_predictors(run_on_cpu=args.cpu)

    # default_threshold = 0.2
    # threshold = st.slider('Threshold (only applicable to casent models):', 0., 1.,
    #                       default_threshold, key='extraction_')

    run_button = st.button(label='✨ Run Model', key='extraction_button_')

    n_words = len(re.findall(r'\w+', doc))
    if n_words > 128:
        st.warning(f'⚠️ Your text contains more than 128 words.')

    if not run_button or len(doc) == 0:
        return

    st.markdown('### Model Output')

    mentions = extract_entities_by_type(
        predictor=predictors['casent_t5_large'],
        stanza_pipeline=load_stanza(use_gpu=not args.cpu),
        text=doc,
        target_ufet_type=target_type,
        threshold=None,
        eval_batch_size=args.extraction_eval_batch_size,
        use_gpu=not args.cpu,

    )

    for mention in mentions:
        st.write(mention)
        assert doc[mention.start_char:mention.end_char] == mention.mention_span


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--extraction_eval_batch_size', type=int, default=8)
    args = parser.parse_args()
    print(args)
    print()

    st.set_page_config(
        page_title='CASENT',
        page_icon='🎈',
    )
    st.title('🎈 CASENT')

    entity_typing, entity_extraction = st.tabs(['Entity Typing',
                                                'Entity Extraction'])

    with entity_typing:
        entity_typing_demo(args)

    with entity_extraction:
        entity_extraction_demo(args)


if __name__ == '__main__':
    main()
