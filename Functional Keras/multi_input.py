from keras.models import Model
from keras import layers, Input

text_vocab = 10000
question_vocab = 10000
answer_vocab = 500

text_input = Input(shape=(None,), dtype='int32', name='text')
embedded_text = layers.Embedding(64, text_vocab)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)

question_input = Input(shape=(None,), dtype='int32', name='question')
embedding_question = layers.Embedding(32, question_vocab)(question_input)
encoded_question = layers.LSTM(16)(embedding_question)

concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)

answer = layers.Dense(answer_vocab, activation='softmax')(concatenated)

model = Model([text_input, question_input], answer)
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# when you want to fit the model there are 2 ways: with/without input names.
# the "name='question/text" above are input names and if your data has that, then you
# fit a model like so:
# model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)
# otherwise, you fit a model like so:
# model.fit([text, question], answers, epochs=10, batch_size=128)

