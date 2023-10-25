# CASENT

demo: http://chronos.lti.cs.cmu.edu:8401/

## Installation

```
git clone https://github.com/yanlinf/CASENT.git
cd CASENT
pip install -e .
```

## Quick start

### Predict UFET types

```python
from casent.entity_typing_t5 import T5ForEntityTypingPredictor

predictor = T5ForEntityTypingPredictor.from_pretrained('yanlinf/casent-large')
preds = predictor.predict_raw(
    ['A court in Jerusalem sentenced <M> a Palestinian </M> to 16 life terms for forcing a bus off a cliff July 6 , killing 16 people']
)
preds[0].scores
# preds[0].types == ['person', 'criminal', 'male']
# preds[0].scores == [0.975004041450473, 0.6304963191225533, 0.5362320213818272]
```

### Predict WikiData types

We also offer a model that predicts Wikidata types along with their Qnode IDs. (note: this is done by mapping the UFET type vocabulary to Wikidata using automatic methods, so the mapping might not be entirely correct)

```python
from casent.entity_typing_t5 import T5ForWikidataEntityTypingPredictor

predictor = T5ForWikidataEntityTypingPredictor.from_pretrained(
    'yanlinf/casent-large',
    ontology_path='ontology_data/ontology_expanded.json'
)
preds = predictor.predict_raw(
    ['A court in Jerusalem sentenced <M> a Palestinian </M> to 16 life terms for forcing a bus off a cliff July 6 , killing 16 people']
)
# preds[0].wd_types == [Concept(Q215627, person), Concept(Q2159907, criminal), Concept(Q6581097, male)]
# preds[0].scores == [0.975004041450473, 0.6304963191225533, 0.5362320213818272])]
```

## Training CASENT

### 1. Training 

```bash
python train_t5.py --save_dir checkpoints/exp0/
```

### 2. Inference

```bash
python predict_t5.py --model checkpoints/exp0/
```

Predictions will be saved to `dev_pred.json` and `test_pred.json` under the model checkpoint directory.

### 3. Evaluation

```bash
python evaluate_ufet_predictions.py --input_path checkpoints/exp0/test_pred.json
```
