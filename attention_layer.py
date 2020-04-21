import keras.backend as K
from keras.layers import Layer


class AttentionLayer(Layer):
    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return mask[1]

    def compute_output_shape(self, input_shape):
        return input_shape[1][0], input_shape[1][1], input_shape[1][2] * 2

    def call(self, inputs, mask=None):
        # encoder_outputs: [batch_size, max_source_sent_len, hidden_size]
        # decoder_outputs: [batch_size, max_target_sent_len, hidden_size]​
        encoder_outputs, decoder_outputs = inputs

        decoder_outputs_t = K.permute_dimensions(decoder_outputs, (0, 2, 1))
        # [batch_size, max_source_sent_len, max_target_sent_len]​
        luong_score = K.batch_dot(encoder_outputs, decoder_outputs_t)
        luong_score = K.softmax(luong_score, axis=1)

        encoder_vector = K.expand_dims(encoder_outputs, axis=2) * K.expand_dims(luong_score, axis=3)
        encoder_vector = K.sum(encoder_vector, axis=1)

        # [batch_size, max_target_sent_len, 2 * hidden_size]
        new_decoder_outputs = K.concatenate([decoder_outputs, encoder_vector])

        return new_decoder_outputs
