import numpy as np
import tensorflow as tf
import keras

from utils import DataGenerator, get_dataset_c1

tf.config.experimental.enable_tensor_float_32_execution(False)


'''
Load data sets
'''

inhom_data = np.load("data/LJ_T1.5_inhom.npy", allow_pickle=True).item()

batch_size = 256
windowSigma = 3.5

generatorOptions = dict(batch_size=batch_size, windowSigma=windowSigma, inputKeys=["rho"], outputKeys=["c1"])
trainingGenerator = DataGenerator(inhom_data["training"], **generatorOptions)

train_dataset_c1 = get_dataset_c1(trainingGenerator)


'''
Define model
'''

inputs = {"rho": keras.Input(shape=trainingGenerator.inputShape, name="rho")}
x = keras.layers.Dense(512, activation="gelu")(inputs["rho"])
x = keras.layers.Dense(512, activation="gelu")(x)
x = keras.layers.Dense(512, activation="gelu")(x)
outputs = {"fexc": keras.layers.Dense(1, name="fexc")(x)}

model = keras.Model(inputs=inputs, outputs=outputs)
optimizer = keras.optimizers.Adam()
loss = keras.losses.MeanSquaredError()
metrics = [keras.metrics.MeanAbsoluteError()]
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics,
)
model.summary()
keras.utils.plot_model(model, "model_with_shape_info.png", show_shapes=True, show_layer_names=True, show_layer_activations=True)


'''
Custom training loop for calculation of c1 value from local fexc model
'''

@tf.function
def train_step(x, y, dx=0.01, beta=1/1.5):
    with tf.GradientTape() as tape:
        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(x)
            fexc_model = model(x, training=True)["fexc"]
        dfexc_model = tape2.gradient(fexc_model, x["rho"]) / dx
        c1_model = fexc_model + tf.reduce_sum(dfexc_model * x["rho"], axis=-1, keepdims=True) * dx
        c1_model *= -beta
        loss_c1 = loss(y["c1"], c1_model)
    grads = tape.gradient(loss_c1, model.trainable_weights)
    optimizer.apply(grads, model.trainable_weights)
    for metric in metrics:
        metric.update_state(y["c1"], c1_model)
    return loss_c1

for epoch in range(200):
    print(f"Epoch: {epoch}")
    print(f"\tlearning rate: {optimizer.learning_rate.numpy():.4g}")

    for step, ((x, y)) in enumerate(train_dataset_c1):
        loss_c1 = train_step(x, y)

    print(f"\tsteps: {step}")
    print(f"\tloss_c1: {loss_c1:.4g}")

    for metric in metrics:
        print(f"\t{metric.name} (c1): {metric.result():.4g}")
        metric.reset_state()

    model.save("models/current_fexc.keras")
    optimizer.learning_rate *= 0.95
