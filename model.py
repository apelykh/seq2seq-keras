import time
import numpy as np
from keras.layers import Embedding, LSTM, Dropout, Dense
from keras import Model, Input
from keras.optimizers import Adam
from nltk.translate.bleu_score import corpus_bleu
from attention_layer import AttentionLayer
from dataset import load_dataset


class NMTModel:
    def __init__(self, source_dict, target_dict, use_attention):
        self.hidden_size = 200
        self.embedding_size = 100
        self.hidden_dropout_rate = 0.2
        self.embedding_dropout_rate = 0.2
        self.batch_size = 100
        self.max_target_step = 30
        self.vocab_target_size = len(target_dict.vocab)
        self.vocab_source_size = len(source_dict.vocab)
        self.target_dict = target_dict
        self.source_dict = source_dict
        self.SOS = target_dict.word2ids['<start>']
        self.EOS = target_dict.word2ids['<end>']
        self.use_attention = use_attention
        self.train_model = None
        self.encoder_model = None
        self.decoder_model = None

        print("Source vocab size: {}, target vocab size: {}".format(
            self.vocab_source_size, self.vocab_target_size))

    def _build_training_encoder(self, source_words, target_words):
        embedding_source = Embedding(
            input_dim=self.vocab_source_size,
            output_dim=100,
            mask_zero=True
        )
        embedding_target = Embedding(
            input_dim=self.vocab_target_size,
            output_dim=100,
            mask_zero=True
        )

        source_words_embeddings = embedding_source(source_words)
        target_words_embeddings = embedding_target(target_words)

        embedding_dropout = Dropout(rate=self.embedding_dropout_rate)

        source_words_embeddings = embedding_dropout(source_words_embeddings)
        target_words_embeddings = embedding_dropout(target_words_embeddings)

        encoder_lstm = LSTM(
            self.hidden_size,
            return_sequences=True,
            return_state=True
        )
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(source_words_embeddings)

        return target_words_embeddings, encoder_outputs, encoder_state_h, encoder_state_c

    def build(self):
        source_words = Input(shape=(None,), dtype='int32')
        target_words = Input(shape=(None,), dtype='int32')

        target_words_embeddings, encoder_outputs, encoder_state_h, encoder_state_c = \
            self._build_training_encoder(source_words, target_words)

        encoder_states = [encoder_state_h, encoder_state_c]

        # decoder for training
        decoder_lstm = LSTM(
            self.hidden_size,
            recurrent_dropout=self.hidden_dropout_rate,
            return_sequences=True,
            return_state=True
        )
        decoder_outputs_train, _, _ = decoder_lstm(
            target_words_embeddings,
            initial_state=encoder_states
        )

        if self.use_attention:
            decoder_attention = AttentionLayer()
            decoder_outputs_train = decoder_attention(
                [encoder_outputs, decoder_outputs_train]
            )

        decoder_dense = Dense(self.vocab_target_size,
                              activation='softmax')
        decoder_outputs_train = decoder_dense(decoder_outputs_train)

        adam = Adam(lr=0.01, clipnorm=5.0)
        self.train_model = Model(
            [source_words, target_words],
            decoder_outputs_train
        )
        self.train_model.compile(
            optimizer=adam,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.train_model.summary()

        self.encoder_model = Model(
            source_words,
            [encoder_outputs, encoder_state_h, encoder_state_c]
        )
        self.encoder_model.summary()

        decoder_state_input_h = Input(shape=(self.hidden_size,))
        decoder_state_input_c = Input(shape=(self.hidden_size,))
        encoder_outputs_input = Input(shape=(None, self.hidden_size,))

        # decoder for inference
        decoder_input_states = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs_test, decoder_state_output_h, decoder_state_output_c = \
            decoder_lstm(target_words_embeddings, initial_state=decoder_input_states)

        if self.use_attention:
            decoder_outputs_test = decoder_attention(
                [encoder_outputs_input, decoder_outputs_test]
            )

        decoder_outputs_test = decoder_dense(decoder_outputs_test)

        self.decoder_model = Model(
            [target_words, decoder_state_input_h, decoder_state_input_c, encoder_outputs_input],
            [decoder_outputs_test, decoder_state_output_h, decoder_state_output_c]
        )
        self.decoder_model.summary()

    def time_used(self, start_time):
        curr_time = time.time()
        used_time = curr_time - start_time
        m = used_time // 60
        s = used_time - 60 * m
        return "%d m %d s" % (m, s)

    def train(self, train_data, dev_data, test_data, epochs):
        start_time = time.time()

        for epoch in range(epochs):
            print("Starting training epoch {}/{}".format(epoch + 1, epochs))
            epoch_time = time.time()
            source_words_train, target_words_train, target_words_train_labels = train_data

            self.train_model.fit(
                [source_words_train, target_words_train],
                target_words_train_labels,
                batch_size=self.batch_size
            )

            print("Time used for epoch {}: {}".format(epoch + 1, self.time_used(epoch_time)))
            dev_time = time.time()
            print("Evaluating on dev set after epoch {}/{}:".format(epoch + 1, epochs))
            self.eval(dev_data)
            print("Time used for evaluate on dev set: {}".format(self.time_used(dev_time)))

        print("Training finished!")
        print("Time used for training: {}".format(self.time_used(start_time)))

        print("Evaluating on test set:")
        test_time = time.time()
        self.eval(test_data)
        print("Time used for evaluate on test set: {}".format(self.time_used(test_time)))

    def get_target_sentences(self, sents, vocab, reference=False):
        str_sents = []
        num_sent, max_len = sents.shape
        for i in range(num_sent):
            str_sent = []
            for j in range(max_len):
                t = sents[i, j].item()
                if t == self.SOS:
                    continue
                if t == self.EOS:
                    break

                str_sent.append(vocab[t])
            if reference:
                str_sents.append([str_sent])
            else:
                str_sents.append(str_sent)
        return str_sents

    def eval(self, dataset):
        source_words, target_words_labels = dataset
        vocab = self.target_dict.vocab

        encoder_outputs, state_h, state_c = self.encoder_model.predict(
            source_words, batch_size=self.batch_size)
        predictions = []
        step_target_words = np.ones([source_words.shape[0], 1]) * self.SOS

        for _ in range(self.max_target_step):
            step_decoder_outputs, state_h, state_c = self.decoder_model.predict(
                [step_target_words, state_h, state_c, encoder_outputs],
                batch_size=self.batch_size
            )
            step_target_words = np.argmax(step_decoder_outputs, axis=2)
            predictions.append(step_target_words)

        candidates = self.get_target_sentences(np.concatenate(predictions, axis=1), vocab)
        references = self.get_target_sentences(target_words_labels, vocab, reference=True)

        score = corpus_bleu(references, candidates)
        print("Model BLEU score: %.2f" % (score * 100.0))


if __name__ == '__main__':
    max_example = 30000
    use_attention = True
    train_data, dev_data, test_data, source_dict, target_dict = \
        load_dataset("data.30.vi", "data.30.en", max_num_examples=max_example)

    print("read {}/{}/{} train/dev/test batches".format(
        len(train_data[0]), len(dev_data[0]), len(test_data[0])))

    model = NMTModel(source_dict, target_dict, use_attention)
    model.build()
    model.train(train_data, dev_data, test_data, 10)
