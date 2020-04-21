import numpy as np
from keras.preprocessing.sequence import pad_sequences
from language_dict import LanguageDict


def load_dataset(source_path, target_path, max_num_examples=30000):
    source_lines = open(source_path).readlines()
    target_lines = open(target_path).readlines()

    assert len(source_lines) == len(target_lines)

    if max_num_examples > 0:
        max_num_examples = min(len(source_lines), max_num_examples)
        source_lines = source_lines[:max_num_examples]
        target_lines = target_lines[:max_num_examples]

    source_sents = [[tok.lower() for tok in sent.strip().split(' ')]
                    for sent in source_lines]
    target_sents = [[tok.lower() for tok in sent.strip().split(' ')]
                    for sent in target_lines]
    for sent in target_sents:
        sent.append('<end>')
        sent.insert(0, '<start>')

    source_lang_dict = LanguageDict(source_sents)
    target_lang_dict = LanguageDict(target_sents)

    unit = len(source_sents) // 10

    source_words = [[source_lang_dict.word2ids.get(tok, source_lang_dict.UNK) for tok in sent]
                    for sent in source_sents]
    source_words_train = pad_sequences(
        source_words[:8 * unit],
        padding='post'
    )
    source_words_dev = pad_sequences(
        source_words[8 * unit:9 * unit],
        padding='post'
    )
    source_words_test = pad_sequences(
        source_words[9 * unit:],
        padding='post'
    )

    eos = target_lang_dict.word2ids['<end>']

    target_words = [[target_lang_dict.word2ids.get(tok, target_lang_dict.UNK) for tok in sent[:-1]]
                    for sent in target_sents]
    target_words_train = pad_sequences(
        target_words[:8 * unit],
        padding='post'
    )

    target_words_train_labels = [sent[1:] + [eos] for sent in target_words[:8 * unit]]
    target_words_train_labels = pad_sequences(
        target_words_train_labels,
        padding='post'
    )
    target_words_train_labels = np.expand_dims(target_words_train_labels, axis=2)

    target_words_dev_labels = pad_sequences(
        [sent[1:] + [eos] for sent in target_words[8 * unit:9 * unit]],
        padding='post'
    )
    target_words_test_labels = pad_sequences(
        [sent[1:] + [eos] for sent in target_words[9 * unit:]],
        padding='post'
    )

    train_data = [source_words_train, target_words_train, target_words_train_labels]
    dev_data = [source_words_dev, target_words_dev_labels]
    test_data = [source_words_test, target_words_test_labels]

    return train_data, dev_data, test_data, source_lang_dict, target_lang_dict
