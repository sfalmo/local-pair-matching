import numpy as np
import tensorflow as tf
import keras

from utils import CyclicConv1D, Integrate1D

tf.config.experimental.enable_tensor_float_32_execution(False)


'''
Load data sets without using the DataGenerator (we keep the whole profiles for this model)
'''

inhom_data = np.load("data/LJ_T1.5_inhom.npy", allow_pickle=True).item()

inputProfiles = {"rho": np.array([sim["profiles"]["rho"] for sim in inhom_data["training"].values()])}
outputProfiles = {"c1": np.array([sim["profiles"]["c1"] for sim in inhom_data["training"].values()])}

train_dataset = tf.data.Dataset.from_tensor_slices((inputProfiles, outputProfiles)).shuffle(buffer_size=1024, reshuffle_each_iteration=True).repeat(16).batch(128).prefetch(tf.data.AUTOTUNE)


'''
Define model
'''

inputs = {"rho": keras.Input(shape=(None,), name="rho")}  # None to allow variable input size
x = keras.layers.Reshape((-1, 1))(inputs["rho"])
x = CyclicConv1D(conv_kwargs=dict(kernel_size=11, filters=16, dilation_rate=1, activation="gelu"))(x)
x = CyclicConv1D(conv_kwargs=dict(kernel_size=11, filters=16, dilation_rate=2, activation="gelu"))(x)
x = CyclicConv1D(conv_kwargs=dict(kernel_size=11, filters=32, dilation_rate=4, activation="gelu"))(x)
x = CyclicConv1D(conv_kwargs=dict(kernel_size=11, filters=32, dilation_rate=8, activation="gelu"))(x)
x = CyclicConv1D(conv_kwargs=dict(kernel_size=11, filters=64, dilation_rate=16, activation="gelu"))(x)
x = CyclicConv1D(conv_kwargs=dict(kernel_size=11, filters=64, dilation_rate=32, activation="gelu"))(x)
x = CyclicConv1D(conv_kwargs=dict(kernel_size=11, filters=16, dilation_rate=64, activation="gelu"))(x)
x = CyclicConv1D(conv_kwargs=dict(kernel_size=11, filters=1, dilation_rate=1, activation=None))(x)
fexc = keras.layers.Flatten(name="fexc")(x)
rho_fexc = keras.layers.Multiply(name="rho_fexc")([inputs["rho"], fexc])
Fexc = Integrate1D(dx=0.01, name="Fexc")(rho_fexc)
outputs = {"fexc": fexc, "Fexc": Fexc}

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
Custom training loop for calculation of c1 profile from global Fexc model
'''

@tf.function
def train_step(x, y, dx=0.01, beta=1/1.5):
    valid_mask = tf.math.is_finite(y["c1"])
    c1_valid = y["c1"][valid_mask]
    with tf.GradientTape() as tape:
        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(x)
            result_model = model(x, training=True)
            Fexc_model = result_model["Fexc"]
        c1_model = -beta * tape2.gradient(Fexc_model, x["rho"]) / dx
        c1_model_valid = c1_model[valid_mask]
        loss_c1 = loss(c1_valid, c1_model_valid)
    grads = tape.gradient(loss_c1, model.trainable_weights)
    optimizer.apply(grads, model.trainable_weights)
    for metric in metrics:
        metric.update_state(c1_valid, c1_model_valid)
    return loss_c1

for epoch in range(200):
    print(f"Epoch: {epoch}")
    print(f"\tlearning rate: {optimizer.learning_rate.numpy():.4g}")

    for step, ((x, y)) in enumerate(train_dataset):
        loss_c1 = train_step(x, y)

    print(f"\tsteps: {step}")
    print(f"\tloss_c1: {loss_c1:.4g}")

    for metric in metrics:
        print(f"\t{metric.name} (c1): {metric.result():.4g}")
        metric.reset_state()

    model.save("models/current_Fexc.keras")
    optimizer.learning_rate *= 0.95
