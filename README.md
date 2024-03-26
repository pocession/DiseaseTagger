# GeneTagger
This is a deep-learning-based tool to identify gene names from raw texts. We used [BioBERT large v1.1](http://nlp.dmis.korea.edu/projects/biobert-2020-checkpoints/biobert_large_v1.1_pubmed.tar.gz) as our model.

## Requirements
This tools is built and tested on Mac M2 platform. It needs special version of tensorflow. Please see [The package file](./requiremnts.txt) 

## Named-entity recognition (NER)
In plain English, a NER task is to identify a specific kind of term from the raw text. For example, we may want to identify human names, location, countries, etc from a given text. Here we are identifying the gene names from raw texts.

## Resources
- [CoNLL file format](https://stackoverflow.com/questions/27416164/what-is-conll-data-format): this thread illustrates the CoNLL file format.
