# CASENT 

[![](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) 
[![](https://img.shields.io/badge/🤗-HuggingFace-red.svg)](https://huggingface.co/yanlinf/casent-large)
[![](https://img.shields.io/badge/emnlp23-Paper-yellow.svg)](https://huggingface.co/yanlinf/casent-large)
[![](https://img.shields.io/badge/🎈-Demo-blue.svg)](http://chronos.lti.cs.cmu.edu:8401/)

> Calibrated Seq2Seq Models for Efficient and Generalizable Ultra-fine Entity Typing<br/>
> Yanlin Feng, Adithya Pratapa, David R Mortensen<br/>
> EMNLP Findings 2023

CASENT is a lightweight multi-label entity classification model designed for extremely large label space (e.g., UFET and WikiData). It can also be used for entity extraction and tagging when integrated with a span detector.

CASENT offers several advantages compared to previous methods: 1) Standard maximum likelihood training; 2) Efficient inference through a single autoregressive decoding pass; 3) Calibrated confidence scores; 4) Strong generalization performance to unseen domains and types. 

## Installation

```bash
conda create -n casent python=3.10
conda activate casent
git clone https://github.com/yanlinf/CASENT.git
cd CASENT
pip install -r requirements.txt
pip install -e .
```

## Quick start

Pretrained models are available on HuggingFace for running inference.

### Usage 1: Predict UFET types

```python
from casent.entity_typing_t5 import T5ForEntityTypingPredictor

predictor = T5ForEntityTypingPredictor.from_pretrained('yanlinf/casent-large')
predictor.predict_raw(
    ['A court in Jerusalem sentenced <M> a Palestinian </M> to 16 life terms for forcing a bus off a cliff July 6 , killing 16 people']
)
```

```plain
[EntityTypingOutput(types=['person', 'criminal', 'male'], scores=[0.975004041450473, 0.6304963191225533, 0.5362320213818272])]
```

### Usage 2: Predict WikiData types

We also offer a model that predicts WikiData types along with their WikiData Qnode IDs. (note: this is done by mapping the UFET type vocabulary to WikiData using automatic methods, so the mapping might not be entirely correct. The mapping file is available [here](ontology_data/ufet_mapping.csv))

```python
from casent.entity_typing_t5 import T5ForWikidataEntityTypingPredictor

predictor = T5ForWikidataEntityTypingPredictor.from_pretrained(
    'yanlinf/casent-large',
    ontology_path='ontology_data/ontology_expanded.json'
)
predictor.predict_raw(
    ['A court in Jerusalem sentenced <M> a Palestinian </M> to 16 life terms for forcing a bus off a cliff July 6 , killing 16 people']
)
```

```plain
[WikidataEntityTypingOutput(wd_types=[Concept(Q215627, person), Concept(Q2159907, criminal), Concept(Q6581097, male)], scores=[0.975004041450473, 0.6304963191225533, 0.5362320213818272])]
```

### Usage 3: Entity extraction / tagging

CASENT can also be used to extract entities of a specific type from text, when used in conjunction with a span detector. We provide a simple API that leverages a constituency parser that considers all noun phrases as potential entity mentions (this allows us to extract non-named entities).

```python
from casent.entity_typing_t5 import T5ForEntityTypingPredictor, extract_entities_by_type
import stanza

extract_entities_by_type(
    T5ForEntityTypingPredictor.from_pretrained('yanlinf/casent-large'),
    stanza.Pipeline(lang="en", processors="tokenize,pos,constituency", use_gpu=True),
    text='The Tenerife airport disaster occurred on March 27, 1977, when two Boeing 747 passenger jets collided on the runway at Los Rodeos Airport (now Tenerife North Airport) on the Spanish island of Tenerife. The collision occurred when KLM Flight 4805 initiated its takeoff run during dense fog while Pan Am Flight 1736 was still on the runway.', 
    target_ufet_type='aircraft'
)
```

```plain
[Mention("two Boeing 747 passenger jets", 0.82), Mention("KLM Flight 4805", 0.77), Mention("Pan Am Flight 1736", 0.72)]
 ```

## Training CASENT

### 1. Download data

```bash
bash scripts/download_ufet.sh
```

### 2. Training 

```bash
python train_t5.py -m t5-large --save_dir checkpoints/exp0/
```

### 3. Inference

```bash
python predict_t5.py --model checkpoints/exp0/
```

Predictions will be saved to `dev_pred.json` and `test_pred.json` under the model checkpoint directory.

### 4. Evaluation

```bash
python evaluate_ufet_prediction.py --input_path checkpoints/exp0/test_pred.json
```
