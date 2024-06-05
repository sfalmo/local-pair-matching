import numpy as np
import tensorflow as tf
import keras

from utils import DataGenerator, get_dataset_c1, get_dataset_c1b, get_dataset_c2x

tf.config.experimental.enable_tensor_float_32_execution(False)


'''
Load data sets
'''

inhom_data = np.load("data/LJ_T1.5_inhom.npy", allow_pickle=True).item()
pc_data = np.load("data/LJ_T1.5_pc.npy", allow_pickle=True).item()
pc_data = list(pc_data.values())

batch_size = 256
windowSigma = 3.5

generatorOptions = dict(batch_size=batch_size, windowSigma=windowSigma, inputKeys=["rho"], outputKeys=["c1"])
trainingGenerator = DataGenerator(inhom_data["training"], **generatorOptions)

train_dataset_c1 = get_dataset_c1(trainingGenerator)
train_dataset_c1b = get_dataset_c1b(pc_data, trainingGenerator.inputShape)
train_dataset_c2x = get_dataset_c2x(pc_data, windowSigma, trainingGenerator.inputShape)


'''
Define model
'''

inputs = {"rho": keras.Input(shape=trainingGenerator.inputShape, name="rho")}
x = keras.layers.Dense(512, activation="gelu")(inputs["rho"])
x = keras.layers.Dense(512, activation="gelu")(x)
x = keras.layers.Dense(512, activation="gelu")(x)
outputs = {"c1": keras.layers.Dense(trainingGenerator.outputShape[0], name="c1")(x)}

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
Custom training loop with optional pair-correlation matching
'''

@tf.function
def train_step(x_c1, y_c1, x_c2x=None, y_c2x=None, alpha_c1=1, alpha_c2x=0.01, dx=0.01):
    with tf.GradientTape() as tape:
        c1_model = model(x_c1, training=True)["c1"]
        loss_c1 = loss(y_c1["c1"], c1_model)
        loss_c2x = 0
        if alpha_c2x > 0:
            with tf.GradientTape(watch_accessed_variables=False) as tape2:
                tape2.watch(x_c2x)
                c1_model_pc = model(x_c2x, training=True)["c1"]
            c2x_model = tape2.gradient(c1_model_pc, x_c2x["rho"]) / dx
            loss_c2x = loss(y_c2x["c2"], c2x_model)
        loss_total = alpha_c1 * loss_c1 + alpha_c2x * loss_c2x
    grads = tape.gradient(loss_total, model.trainable_weights)
    optimizer.apply(grads, model.trainable_weights)
    for metric in metrics:
        metric.update_state(y_c1["c1"], c1_model)
    return loss_c1, loss_c2x

for epoch in range(200):
    print(f"Epoch: {epoch}")
    print(f"\tlearning rate: {optimizer.learning_rate.numpy():.4g}")

    # # pure inhom
    # for step, (x_c1, y_c1) in enumerate(train_dataset_c1):
    #     loss_c1, loss_c2x = train_step(x_c1, y_c1, alpha_c1=1, alpha_c2x=0)

    # inhom + pc
    for step, ((x_c1, y_c1), (x_c2x, y_c2x)) in enumerate(zip(train_dataset_c1, train_dataset_c2x)):
        loss_c1, loss_c2x = train_step(x_c1, y_c1, x_c2x, y_c2x, alpha_c1=1, alpha_c2x=0.01)

    # # pure pc
    # for step, ((x_c1b, y_c1b), (x_c2x, y_c2x)) in enumerate(zip(train_dataset_c1b, train_dataset_c2x)):
    #     loss_c1, loss_c2x = train_step(x_c1b, y_c1b, x_c2x, y_c2x, alpha_c1=0.01, alpha_c2x=1)

    print(f"\tsteps: {step}")
    print(f"\tloss_c1: {loss_c1:.4g}")
    print(f"\tloss_c2x: {loss_c2x:.4g}")

    for metric in metrics:
        print(f"\t{metric.name} (c1): {metric.result():.4g}")
        metric.reset_state()

    model.save("models/current_c1.keras")
    optimizer.learning_rate *= 0.95
