# The Cellar
This repository contains download streams for pretrained models for a variety of tasks, including instructions on how to load, modify, operate, and use them. This is being updated all the time when fresher models are trained, but do be patient as networks train very slow! If you have any questions, recommendations, comments, or model requests, do be sure to drop by our issues tracker.

## Transfer Learning Models

### BERT
We provide pretrained BERT (Devlin, et al., 2019) models in Filipino, in both cased and uncased form. We only provide base models (12 layers, 12 heads, 768 units) as the large models are unwieldy and are very expensive to train (the latest one trained in a Google Cloud TPU for 2 weeks and it still lagged in performance behind the base models).

* BERT Tagalog Base Uncased [Link](#)
* BERT Tagalog Base Cased [Link](#)

The pretrained models are in TensorFlow's checkpoint format so it is compatibe for use with their TPU code (CPU usage and GPU usage has not been verified at this point, so we'd have to stick with TPUs). We can convert them to PyTorch checkpoints for use with GPUs. A modified copy of [HuggingFace](https://github.com/huggingface)'s [BERT implementation](https://github.com/huggingface/pytorch-pretrained-BERT) is included in the repository, so be sure to check for installation instructions there. We have also modified this repository to allow more classification tasks other than GLUE and SQuAD.

**Requirements**
* A cloud TPU or a sizeable GPU -- You may use the TensorFlow checkpoints in a TPU (untested in Colab, as the models are very large and colab only supports Google's pretrained models so far) if you have one. otherwise, you can convert to PyTorch checkpoints and use a sizeable enough GPU. We use an NVIDIA Tesla V100 GPU with 16GB of VRAM for finetuning the models for our tasks (it eats up around 14GB/16GB memory for mid-size datasets). We caution against using a smaller GPU and lowering the batch size as small batch sizes **will** hurt performance. You can find more about batch size/performance ratios in the official [BERT repository](https://github.com/google-research/bert).
* At least 12GB RAM - The model and data will be loaded in RAM before loaded into your GPU, so you'd need one that supports the entire thing in memory.
* PyTorch v.1.1.0 -- the latest version has less bugs.
* NVIDIA Apex -- Needed for half-precision training as well as faster norm and forward layers. If you have access to an NVIDIA 20- series RTX GPU (or a Volta series Tesla and beyond) you can try using mixed precision training to effectively halve the VRAM consumption at the cost of a handful of points of accuracy.

### ULMFiT
We provide a pretrained AWD-LSTM (Merity, et al., 2017) model in Filipino that can be finetuned to a classifier using ULMFiT (Howard & Ruder, 2018). We only provide one model. Do note that we still cannot release our training, finetuning, and scaffolding code as the work related to it is under review in a conference. We'll update this repo as soon as anonymity period stops!

* Pretrained AWD-LSTM [Link](#)

While we use our own handwritten scaffolding, we have done extra work to ensure that our pretrained checkpoints are compatible with this library called [FastAI](https://github.com/fastai/fastai) (yay!), which to this date, is still the only reliable implementation of ULMFiT. We'll add in our own finetuning code (standalone, no need for extra packages) to this repository soonest we can. To use the model, you can follow the instructions in the FastAI repository.

**Requirements**
* A sizeable enough GPU -- You will perform finetuning twice (language model finetuning and classifier finetuning) so you'd need space for that much data. Do note that while an AWD-LSTM is in tself a slim model (three recurrent layers with a dolllop of dropout, 400 dimension embeddings and 1150 hidden units per layer), the scaffollding for transfer learning is quite expensive. We have tested on an NVIDIA Tesla V100 GPU 9eating up around 12GB/16GB). For small-medium sized datasets (anything not beyond 10k) you can probably get away with using a Tesla K80, but do note that training performance and time may be compromised.
* FastAI - Check their repository for instructions on how to install the library
* Appropriate CUDA version -- FastAI does not work without a GPU (you need one for finetuning, you dont need one for inferencing). Be sure to install the corrent CUDA version. Volta GPUs and newer (including Turing cards) only support CUDA 10. Be sure your PyTorch version uses the same CUDA compatibiliy as your own installation.

### OpenAi GPT-2
This is currently a work in progress. We will release a pretrained GPT-2 (Radford, et al., 2019) model soon in Filipino for you to use. We do, however, have demo code for finetuning a pretrained Transformer for classification in [this repo](https://github.com/dlsucelt/Transformer).

*Coming soon.*

## Tokenization models

### SentencePiece
We provide a pretrained SentencePiece model for BPE subword tokenization (Seinnrich, et al., 2016). All modern translation models use subword tokenization now.

* Pretrained SentencePiece [Link](#)

**Requirements**
* SentencePiece -- All the instructions are in Google [repository](https://github.com/google/sentencepiece). Be sure to use the same version of compilers listed in their system requirements.

## Word Embeddings

### Monolingual Embeddings
We provide monolingual embeddings in both GloVe and Fasttext formats.

*Uploads in progress.*

### Mutiliungual Embeddings
We provide multillingual and cross-lingual embeddings in the Fasttext format via MUSE (Lample, et al., 2018), as well as other models in other vector-alignment techniques.

*Uploads in progress.*

## Footnotes
We would like to thank the TensorFow Research Cloud initiative for making TPUs more accessible, alllowing us to perform benchmarks on BERT models in Philippine languages. If you have any comments or concerns, do be sure to drop by our issues tracker!

This repository is managed by the De La Salle University Machine Learning Group