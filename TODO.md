Here’s the requirements:

```markdown
# Topic - Self-Attention, Transformers, and Pretraining

## Overview

This project explores Transformer self-attention and the effects of pretraining. You will:

- Extend a simplified research codebase (Karpathy’s minGPT)
- Train a Transformer to answer simple factual questions (e.g., “Where was [person] born?”)
- Investigate how pretraining on Wikipedia improves performance
- Implement a new type of positional embedding: RoPE

Note: Pretraining takes about **1 hour on GPU** (and must be done **twice**).

---

## 1. Pretrained Transformer Models and Knowledge Access (100 pts)

You’ll experiment with a mini-GPT system to access world knowledge.

### (a) Demo Setup (0 pts)

Explore `play_char.ipynb` in `mingpt-demo/` to understand training mechanics. No code or submission required.

### (b) Dataset Review (0 pts)

Check out `NameDataset` in `src/dataset.py`, which uses name/birthplace pairs from TSV files.

Run:
```bash
python src/dataset.py namedata
```

### (c) Finetune Without Pretraining (0 pts)

Modify `run.py`:
- Implement model initialization and training (under `[part c]`)
- Use training code from `play_char.ipynb` as a reference

No written output required.

### (d) Make Predictions (15 pts)

Train and evaluate your model without pretraining:

```bash
python src/run.py finetune vanilla wiki.txt --writing_params_path vanilla.model.params --finetune_corpus_path birth_places_train.tsv

python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.model.params --eval_corpus_path birth_dev.tsv --outputs_path vanilla.nopretrain.dev.predictions

python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.model.params --eval_corpus_path birth_test_inputs.tsv --outputs_path vanilla.nopretrain.test.predictions
```

Also implement and report the **"London baseline"** accuracy using `london_baseline.py`.

### (e) Pretraining Task - Span Corruption (25 pts)

Implement `getitem()` in `CharCorruptionDataset` in `src/dataset.py`. 

Run:
```bash
python src/dataset.py charcorruption
```

This simulates span corruption as described in the T5 paper.

### (f) Pretrain + Finetune + Predict (25 pts)

Pretrain on `wiki.txt`, then finetune on NameDataset.

```bash
# Pretrain
python src/run.py pretrain vanilla wiki.txt --writing_params_path vanilla.pretrain.params

# Finetune
python src/run.py finetune vanilla wiki.txt --reading_params_path vanilla.pretrain.params --writing_params_path vanilla.finetune.params --finetune_corpus_path birth_places_train.tsv

# Evaluate on dev set
python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.finetune.params --eval_corpus_path birth_dev.tsv --outputs_path vanilla.pretrain.dev.predictions

# Evaluate on test set
python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.finetune.params --eval_corpus_path birth_test_inputs.tsv --outputs_path vanilla.pretrain.test.predictions
```

Expect >15% accuracy on dev/test set.

---

## (g) RoPE Embeddings (35 pts)

### i. (2 pts)  
Show that Equation (3) and Equation (4) from the assignment describe the same RoPE operation (up to reshaping).

### ii. (1 pt)  
Show that:
```math
⟨RoPE(z₁, t₁), RoPE(z₂, t₂)⟩ = ⟨RoPE(z₁, t₁ - t₂), RoPE(z₂, 0)⟩
```
(i.e., RoPE dot product depends only on relative position.)

### iii. (8 pts)  
Implement RoPE in:
- `precompute_rotary_emb`
- `apply_rotary_emb`  
(inside `src/attention.py`, `[part g]`)

Train and evaluate with RoPE:

```bash
# Pretrain
python src/run.py pretrain rope wiki.txt --writing_params_path rope.pretrain.params

# Finetune
python src/run.py finetune rope wiki.txt --reading_params_path rope.pretrain.params --writing_params_path rope.finetune.params --finetune_corpus_path birth_places_train.tsv

# Evaluate on dev
python src/run.py evaluate rope wiki.txt --reading_params_path rope.finetune.params --eval_corpus_path birth_dev.tsv --outputs_path rope.pretrain.dev.predictions

# Evaluate on test
python src/run.py evaluate rope wiki.txt --reading_params_path rope.finetune.params --eval_corpus_path birth_test_inputs.tsv --outputs_path rope.pretrain.test.predictions
```

Target: ≥30% accuracy on test set.

---

## 2. Considerations in Pretrained Knowledge (10 pts)

Write answers to these questions:

### (a) (2 pts)  
Why does the pretrained model outperform the non-pretrained version?

### (b) (4 pts)  
List 2 concerns of user-facing NLP apps relying on pretrained knowledge (especially when outputs may be “made up”). Give examples.

### (c) (4 pts)  
If a person is **unseen** during pretraining and finetuning, what might the model do? What’s one ethical concern from such behavior?

---

## 3. Submission Instructions

Ensure these files are in your directory:

### No Pretraining
- `vanilla.model.params`
- `vanilla.nopretrain.dev.predictions`
- `vanilla.nopretrain.test.predictions`

### London Baseline
- `london_baseline_accuracy.txt`

### Pretrain + Finetune
- `vanilla.finetune.params`
- `vanilla.pretrain.dev.predictions`
- `vanilla.pretrain.test.predictions`

### RoPE
- `rope.finetune.params`
- `rope.pretrain.dev.predictions`
- `rope.pretrain.test.predictions`

