This repository contains code and data for the EACL 2021 paper *Language Models as Knowledge Bases: On Entity Representations, Storage Capacity, and Paraphrased Queries*
by Benjamin Heinzerling and Kentaro Inui.

https://www.aclweb.org/anthology/2021.eacl-main.153/

To run memorization experiments, clone this repository, install the dependencies, then download and extract Wikidata statements and entity embeddings:

- [Wikidata relation statements (1.3GB download)](https://drive.google.com/file/d/1j5Fln1jDzUWemUJD4hLBLJMN1AmnJm7S/view?usp=sharing)
- [Wikidata entity embeddings (1.5GB download)](https://drive.google.com/file/d/1KlSCrtJD25VP_i4uoTYyUjh4qDxEKhrL/view?usp=sharing)

The basic command to run a memorization experiment is `python main.py train`.

The size of the entity vocabulary can be set with `--top-n` and the number of facts to store in the LM with `--n-facts`.
The entity representation can be selected with `--entity-repr`, either 'symbol' or 'continuous' 

For example, to store 20k facts with symbolic representation in an LSTM:
```
python main.py train --top-n 10000 --n-facts 20000 --entity-repr symbol --architecture lstm --n-layers 2 --n-hidden 1024 --lr 0.001 --batch-size 128
```

Or with continuous representation, i.e. using entity embeddings as training signal:
```
python main.py train --top-n 10000 --n-facts 20000 --entity-repr continuous --architecture lstm --n-layers 2 --n-hidden 1024 --lr 0.001 --batch-size 128
```

The same with a Transformer instead of an LSTM:
```
python main.py train --top-n 10000 --n-facts 20000 --entity-repr symbol --architecture transformer --batch-size 128
python main.py train --top-n 10000 --n-facts 20000 --entity-repr continuous --architecture transformer --batch-size 128
```

If everthing works, You should get output like this:
```
2020-10-04 10:51:41| main.py train --top-n 10000 --n-facts 20000 --entity-repr symbol --architecture lstm --n-layers 2 --n-hidden 1024 --lr 0.001 --batch-size 128
2020-10-04 10:51:41| n_facts: 20000 | batch size: 128
2020-10-04 10:51:41| Lock 140650883596928 acquired on out/runid.lock
2020-10-04 10:51:41| Lock 140650883596928 released on out/runid.lock
2020-10-04 10:51:41| rundir: out/28
2020-10-04 10:51:41| loading data
2020-10-04 10:51:41| loading cache/wikidatarelationstatements.tensors.top_n10000.max_seq_len60.max_train_inst20000.max_eval_inst1000.max_test_inst1000.pth
2020-10-04 10:51:41| 20000 train instances | 157 batches
2020-10-04 10:51:41| 1000 dev instances | 32 batches
2020-10-04 10:51:47| model params: 40985172 trainable | 0 fixed
2020-10-04 10:51:47| mlflow backend for exp dev: sqlite:///mlflow.db
2020-10-04 10:51:48| Early stopping patience: 20
2020-10-04 10:51:49| mlflow runid:  b720ec0c05834438921ce9a91c55c0da
2020-10-04 10:51:49| Engine run starting with max_epochs=1000.
2020-10-04 10:51:49| No checkpoint found.
2020-10-04 10:51:56| epoch 0001 train | loss 6.4375
2020-10-04 10:51:56| Engine run starting with max_epochs=1.
2020-10-04 10:51:57| Epoch[1] Complete. Time taken: 00:00:01
<s>Royal Society belongs to the country <mask></s> | United Kingdom | United States of America ✗
<s>Savona is an instance of <mask></s> | comune of Italy | university ✗
<s>alumnus is a subclass of <mask></s> | student | Climate Alliance ✗
<s>Serbian Orthodox Church has the official language <mask></s> | Serbian | English ✗
<s>Ho Chi Minh City belongs to the country <mask></s> | Vietnam | United States of America ✗
2020-10-04 10:51:58| checkpoint_acc=0.1760.pt
2020-10-04 10:51:58| Engine run complete. Time taken: 00:00:01
2020-10-04 10:51:58| epoch 0001 dev | loss 5.3016 | acc 0.1760 | hits10 0.4430 | hits100 0.6620
2020-10-04 10:51:58| Epoch[1] Complete. Time taken: 00:00:07

...

2020-10-04 11:04:10| epoch 0079 train | loss 0.0204
2020-10-04 11:04:10| Engine run starting with max_epochs=1.
2020-10-04 11:04:11| Epoch[1] Complete. Time taken: 00:00:01
<s>Royal Society belongs to the country <mask></s> | United Kingdom | United Kingdom ✓
<s>Savona is an instance of <mask></s> | comune of Italy | comune of Italy ✓
<s>alumnus is a subclass of <mask></s> | student | student ✓
<s>Serbian Orthodox Church has the official language <mask></s> | Serbian | Serbian ✓
<s>Ho Chi Minh City belongs to the country <mask></s> | Vietnam | Vietnam ✓
2020-10-04 11:04:11| checkpoint_acc=0.9890.pt
2020-10-04 11:04:11| EarlyStopping: Stop training
2020-10-04 11:04:11| Terminate signaled. Engine will stop after current iteration is finished.
2020-10-04 11:04:11| Engine run complete. Time taken: 00:00:01
2020-10-04 11:04:11| epoch 0079 dev | loss 0.0729 | acc 0.9890 | hits10 0.9990 | hits100 1.0000
2020-10-04 11:04:11| Epoch[79] Complete. Time taken: 00:00:07
{'entity_repr': 'symbol', 'n_hidden': 1024, 'n_layers': 2, 'n_params': 40985172, 'jobid': None, 'n_facts': 20000, 'top_n': 10000, 'architecture': 'lstm', 'transformer_model': 'roberta-base', 'random_seed': 1, 'kb_emb_dim': 64, 'dataset': 'wikidatarelationstatements', 'max_seq_len': 60, 'checkpoint_file': 'checkpoint_acc=0.9890.pt', 'final_epoch': 79, 'loss': 0.07294197380542755, 'acc': 0.989, 'hits10': 0.999, 'hits100': 1.0}
```


