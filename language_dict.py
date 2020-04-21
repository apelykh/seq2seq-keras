import collections


class LanguageDict:
    def __init__(self, sents):
        word_counter = collections.Counter(tok.lower() for sent in sents for tok in sent)

        self.vocab = []
        # zero paddings
        self.vocab.append('<pad>')
        self.vocab.append('<unk>')
        self.vocab.extend([t for t, c in word_counter.items() if c > 10])

        self.word2ids = {w: id for id, w in enumerate(self.vocab)}
        self.UNK = self.word2ids['<unk>']
        self.PAD = self.word2ids['<pad>']
