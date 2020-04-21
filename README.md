# seq2seq-keras
LSTM-based Neural Machine Translation model with Attention

## Description

The implemented model is similar to Google’s Neural Machine Translation (GNMT) system [3] and has the potential to achieve competitive performance with GNMT by using larger and deeper networks.

English-Vietnamese parallel corpus of TED talks, provided by the [IWSLT Evaluation Campaign](https://sites.google.com/site/iwsltevaluation2015/), was used for training and evaluating the model. This implementation is made to translate from Vietnamese into English.


## Evaluation

The model achieves the following BLEU scores on validation and test sets (random 10% of data each):

Without Attention:
- validation set: **5.37**
- test set: **5.64**

With Attention:
- validation set: **15.21**
- test set: **15.43**

## References
**[1]** Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. "Sequence to sequence learning with neural networks." In Proceedings of the 27th International Conference on Neural Information Processing Systems - Volume 2 (NIPS’14). MIT Press, Cambridge, MA, USA, 3104–3112 (2014)

**[2]** Kyunghyun Cho, Bart van Merrienboer, Çaglar Gülçehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk and Yoshua Bengio. “Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.” ArXiv abs/1406.1078 (2014)

**[3]** Wu, Yonghui & Schuster, Mike & Chen, Zhifeng & Le, Quoc & Norouzi, Mohammad & Macherey, Wolfgang & Krikun, Maxim & Cao, Yuan & Gao, Qin & Macherey, Klaus & Klingner, Jeff & Shah, Apurva & Johnson, Melvin & Liu, Xiaobing & Kaiser, ukasz & Gouws, Stephan & Kato, Yoshikiyo & Kudo, Taku & Kazawa, Hideto & Dean, Jeffrey. "Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation." (2016)
