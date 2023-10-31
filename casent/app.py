import streamlit as st
from annotated_text import annotated_text
import argparse
import pandas as pd
import re
import stanza
from streamlit.logger import get_logger
from casent.entity_typing_t5 import *

logger = get_logger(__name__)

MODEL_CHECKPOINT_MAPPING = {
    'casent_t5_large_wikidata': 'yanlinf/casent-large',
    'casent_t5_large': 'yanlinf/casent-large',
}

MODEL_DEVICE_MAPPING = {
    'casent_t5_large': 'cuda:0',
    'casent_t5_large_wikidata': 'cuda:1',
}

EXAMPLE_INPUTS = [
    'Current figures say <M> 203 miners </M> were killed , and another 22 were injured with 13 miners still missing .',
    'Efforts to rescue 10 miners trapped in <M> a collapsed and flooded coal mine </M> in northern Mexico intensified Thursday with hundreds of people',
    '<M>This case, manufacturer control number 2014-0128812</M> is a report from a solicited program GSI Sponsored Market Research referring to a 63 year-old male patient.',
    'A court in Jerusalem sentenced <M> a Palestinian </M> to 16 life terms for forcing a bus off a cliff July 6 , killing 16 people , Israeli radio reported .',
]


@st.cache_resource()
def load_predictors(use_gpu: bool = False):
    predictors = {}
    for name, path in MODEL_CHECKPOINT_MAPPING.items():
        if name.endswith('_wikidata'):
            predictors[name] = T5ForWikidataEntityTypingPredictor.from_pretrained(
                path,
                ontology_path='ontology_data/ontology_expanded.json',
            )
        else:
            predictors[name] = T5ForEntityTypingPredictor.from_pretrained(path)
        if use_gpu:
            predictors[name].to(torch.device(MODEL_DEVICE_MAPPING[name]))
        else:
            predictors[name].to(torch.device('cpu'))
    return predictors


def write_ufet_pred(pred: EntityTypingOutput):
    df = []
    for ufet_type, score in zip(pred.types, pred.scores):
        df.append((ufet_type, score))
    st.table(pd.DataFrame(df, columns=('UFET Type', 'Score')))


def write_wikidata_pred(pred: WikidataEntityTypingOutput):
    df = []
    for wd_type, score in zip(pred.wd_types, pred.scores):
        df.append((' / '.join(wd_type.ufet_labels), wd_type.wd_qid, wd_type.wd_label, score))
    st.table(pd.DataFrame(df, columns=('UFET Type', 'Wikidata QID', 'Wikidata Type', 'Score')))


def entity_typing_demo(args):
    st.write('')
    st.write('')
    st.write('Choose your models:')
    model_options = []
    for model in MODEL_CHECKPOINT_MAPPING:
        model_options.append(st.checkbox(f'`{model}`', True if model == 'casent_t5_large_wikidata' else False))

    example = st.selectbox('Example inputs', ['Select a sentence'] + EXAMPLE_INPUTS)

    doc = st.text_area(
        'Paste your text below (max 128 words)',
        '' if example == 'Select a sentence' else example,
        height=150,
    ).strip()

    predictors = load_predictors(use_gpu=args.use_gpu)

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

    st.write('---')

    selected_models = [name for name, option in zip(MODEL_CHECKPOINT_MAPPING, model_options)
                       if option]

    for model_name in selected_models:
        predictor = predictors[model_name]
        pred, = predictor.predict_raw([doc])
        st.write(f'`{model_name}`')
        if isinstance(pred, EntityTypingOutput):
            write_ufet_pred(pred)
        elif isinstance(pred, WikidataEntityTypingOutput):
            write_wikidata_pred(pred)

        if show_uncalibrated:
            st.write(f'`{model_name}` uncalibrated scores:')
            pred, = predictor.predict_raw([doc], do_calibration=False)
            if isinstance(pred, EntityTypingOutput):
                write_ufet_pred(pred)
            elif isinstance(pred, WikidataEntityTypingOutput):
                write_wikidata_pred(pred)


@st.cache_resource()
def load_stanza(use_gpu: bool = False):
    return stanza.Pipeline(
        lang="en", processors="tokenize,pos,constituency", use_gpu=use_gpu
    )


def write_annotated_text(doc: str, mentions: List[EntityMentionSpan]):
    res = []
    p = 0
    for m in mentions:
        res.append(doc[p:m.start_char])
        # res.append((doc[m.start_char:m.end_char], f'{m.score:.2f}', '#c9821c'))
        # res.append((doc[m.start_char:m.end_char], f'{m.score:.2f}'))
        res.append((doc[m.start_char:m.end_char], f'{m.score:.2f}', '#5e98d1'))
        p = m.end_char
    res.append(doc[p:])
    annotated_text(*res)


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
        'Paste your text below',
        '' if example == 'Select a document' else example,
        height=150,
        key='extraction_doc_',
    ).strip()

    # default_threshold = 0.2
    # threshold = st.slider('Threshold (only applicable to casent models):', 0., 1.,
    #                       default_threshold, key='extraction_')

    run_button = st.button(label='✨ Run Model', key='extraction_button_')

    # n_words = len(re.findall(r'\w+', doc))
    # if n_words > 128:
    #     st.warning(f'⚠️ Your text contains more than 128 words.')

    if not run_button or len(doc) == 0:
        return

    predictors = load_predictors(use_gpu=args.use_gpu)

    st.write('---')

    mentions = extract_entities_by_type(
        predictor=predictors['casent_t5_large'],
        stanza_pipeline=load_stanza(use_gpu=args.use_gpu),
        text=doc,
        target_ufet_type=target_type,
        threshold=None,
        eval_batch_size=args.extraction_eval_batch_size,
        use_gpu=args.use_gpu,

    )

    write_annotated_text(doc, mentions)

    st.write('')
    st.write('')
    with st.expander('View detailed outputs:'):
        for i, m in enumerate(mentions):
            st.write('---')
            st.write(f'##### Entity *{i}*: "{m.mention_span}"')
            st.write(m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--extraction_eval_batch_size', type=int, default=8)
    args = parser.parse_args()
    logger.info(args)
    logger.info('')

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
