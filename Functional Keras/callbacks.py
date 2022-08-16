from subprocess import call
import keras

# making data usable
x = 'features'
y = 'labels'
x_val = 'features validation'
y_val = 'labels validation'

callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor='acc',
        patience=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='my_model.h5',
        monitor='val_loss',
        save_best_only=True
    )
]

model = keras.models.Model()

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x, y,
    epochs=10,
    batch_size=32,
    callbacks=callbacks_list,
    validation_data=(x_val, y_val) # we will be monitoring for the validation loss
)