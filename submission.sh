rm -f homework5.zip 
rm -rf src/__pycache__
zip -r homework5.zip src/ vanilla.model.params vanilla.finetune.params rope.finetune.params vanilla.nopretrain.dev.predictions vanilla.nopretrain.test.predictions vanilla.pretrain.dev.predictions vanilla.pretrain.test.predictions rope.pretrain.dev.predictions rope.pretrain.test.predictions london_baseline_accuracy.txt
