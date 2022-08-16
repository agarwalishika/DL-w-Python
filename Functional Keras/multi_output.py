from keras import layers, Input, models

# create a model (not fleshed out for reasons)

input = Input(shape=(None,))
# ...
x = layers.Dense(128, activation='relu')(input) # final layer

# multiple outputs - age, income and gender
pred_one = layers.Dense(1, name='age')(x)
pred_two = layers.Dense(10, activation='softmax', name='income')(x)
pred_three = layers.Dense(1, activation='sigmoid', name='gender')(x)

model = models.Model(input, [pred_one, pred_two, pred_three])

# when you compile, wouldn't you need to evaluate the predictions differently?
# mae vs binary_crossentropy vs categorical_crossentropy? how would you do that?

model.compile(
    optimizer='rmsprop',
    loss=['mse', 'categorical_crossentropy', 'binary_crossentropy']
    # loss_weights=[0.25, 1, 10] -> used for if the losses need to be weighted differently
)