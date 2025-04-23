# 1. Evaluate on dev set
```bash
python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.model.params --eval_corpus_path birth_dev.tsv --outputs_path vanilla.nopretrain.dev.predictions
```
**Results:**
```
### Local
- Model: vanilla (no pretraining)
- Epochs: 75
- Dev Accuracy: 1.4% (7 out of 500 correct)
- Params: 3323392
- Device: CPU
```

```
### G Results 
2025-04-23 19:53:55.969162: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1745438036.003408    5309 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1745438036.013423    5309 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-04-23 19:53:56.044526: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
/content/transformer-qa-pretrain/src/run.py:39: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = amp.GradScaler(enabled=use_amp)
data has 418351 characters, 256 unique.
number of parameters: 3323392
Model on device:  cuda:0
500it [00:51,  9.74it/s]
Correct: 9.0 out of 500.0: 1.7999999999999998%
```

### 2. Evaluate on test set
```bash
python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.model.params --eval_corpus_path birth_test_inputs.tsv --outputs_path vanilla.nopretrain.test.predictions
```

**Results:**
```
Model: vanilla (no pretraining)
data has 418351 characters, 256 unique.
Number of parameters: 3323392
437it [02:02,  3.57it/s]
No gold birth places provided; returning (0,0)
Predictions written to vanilla.nopretrain.test.predictions; no targets provided

```
### G Results
```
2025-04-23 19:55:00.679501: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1745438100.699386    5599 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1745438100.705383    5599 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-04-23 19:55:00.725355: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
/content/transformer-qa-pretrain/src/run.py:39: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = amp.GradScaler(enabled=use_amp)
data has 418351 characters, 256 unique.
number of parameters: 3323392
Model on device:  cuda:0
437it [00:44,  9.75it/s]
No gold birth places provided; returning (0,0)
Predictions written to vanilla.nopretrain.test.predictions; no targets provided
```



# Evaluate on dev set
```bash
python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.finetune.params --eval_corpus_path birth_dev.tsv --outputs_path vanilla.pretrain.dev.predictions
```

**Results:**
```
Model: vanilla (with pretraining)
data has 418351 characters, 256 unique.
number of parameters: 3323392
Model on device:  cpu
500it [02:06,  3.97it/s]
Correct: 54.0 out of 500.0: 10.8%
```

# Evaluate on test set
```bash
python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.finetune.params --eval_corpus_path birth_test_inputs.tsv --outputs_path vanilla.pretrain.test.predictions
```

**Results:**
```
Model: vanilla (with pretraining)
data has 418351 characters, 256 unique.
number of parameters: 3323392
Model on device:  cpu
437it [02:06,  3.46it/s]
No gold birth places provided; returning (0,0)
Predictions written to vanilla.pretrain.test.predictions; no targets provided
```

