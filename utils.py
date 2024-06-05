import itertools
import numpy as np
import tensorflow as tf
import keras


'''
Custom layer for cyclic convolutions in 1D
'''

@keras.saving.register_keras_serializable()
class CyclicConv1D(keras.layers.Layer):
    def __init__(self, conv_kwargs={}, **kwargs):
        super().__init__(**kwargs)
        self.conv_kwargs = conv_kwargs
        self.concat = keras.layers.Concatenate(axis=-2)
        self.conv1d = keras.layers.Conv1D(**conv_kwargs)
        self.pad = (self.conv1d.kernel_size[0] // 2) * self.conv1d.dilation_rate[0]

    def call(self, inputs):
        pad_right = inputs[:,:self.pad,:]
        pad_left = inputs[:,-self.pad:,:]
        concat = self.concat([pad_left, inputs, pad_right])
        return self.conv1d(concat)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "conv_kwargs": self.conv_kwargs,
        }
        return {**base_config, **config}


'''
Custom layer for spatial integration in 1D
'''

@keras.saving.register_keras_serializable()
class Integrate1D(keras.layers.Layer):
    def __init__(self, dx=0.01, **kwargs):
        super().__init__(**kwargs)
        self.dx = dx

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=-1, keepdims=True) * self.dx

    def get_config(self):
        base_config = super().get_config()
        config = {
            "dx": self.dx
        }
        return {**base_config, **config}


'''
Data generator which yields batches of input windows and output values from whole profiles (useful for local learning of neural functionals).
'''

class DataGenerator(keras.utils.PyDataset):
    def __init__(self, simData, batch_size=32, steps_per_execution=1, shuffle=True, inputKeys=["rho"], paramsKeys=[], outputKeys=["c1"], windowSigma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.simData = simData
        self.inputKeys = inputKeys
        self.paramsKeys = paramsKeys
        self.outputKeys = outputKeys
        self.windowSigma = windowSigma
        firstSimData = list(self.simData.values())[0]
        self.dz = 2 * firstSimData["profiles"]["z"][0]
        self.simDataBins = len(firstSimData["profiles"]["z"])
        self.windowBins = int(round(self.windowSigma/self.dz))
        self.validBins = {}
        self.inputDataPadded = {}
        for simId in self.simData.keys():
            valid = np.full(self.simDataBins, True)
            for k in self.outputKeys:
                valid = np.logical_and(valid, ~np.isnan(self.simData[simId]["profiles"][k]))
            self.validBins[simId] = np.flatnonzero(valid)
            self.inputDataPadded[simId] = np.pad(self.simData[simId]["profiles"][self.inputKeys], self.windowBins, mode="wrap")
        self.batch_size = batch_size
        self.steps_per_execution = steps_per_execution
        self.inputShape = (2*self.windowBins+1,)
        self.outputShape = (len(self.outputKeys),)
        self.shuffle = shuffle
        self.on_epoch_end()
        print(f"Initialized DataGenerator from {len(self.simData)} simulations which will yield up to {len(self.indices)} input/output samples in batches of {self.batch_size}")

    def __len__(self):
        return int(np.floor(len(self.indices) / (self.batch_size * self.steps_per_execution))) * self.steps_per_execution

    def __getitem__(self, index):
        ids = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        profiles = {key: np.empty((self.batch_size, *self.inputShape)) for key in self.inputKeys}
        params = {key: np.empty((self.batch_size, 1)) for key in self.paramsKeys}
        output = {key: np.empty((self.batch_size, *self.outputShape)) for key in self.outputKeys}
        for b, (simId, i) in enumerate(ids):
            for key in self.inputKeys:
                profiles[key][b] = self.inputDataPadded[simId][key][i:i+2*self.windowBins+1]
            for key in self.paramsKeys:
                params[key][b] = self.simData[simId]["params"][key]
            for key in self.outputKeys:
                output[key][b] = self.simData[simId]["profiles"][key][i]
        return (profiles | params), output

    def on_epoch_end(self):
        self.indices = []
        for simId in self.simData.keys():
            self.indices.extend(list(itertools.product([simId], list(self.validBins[simId]))))
        if self.shuffle == True:
            np.random.default_rng().shuffle(self.indices)

    def pregenerate(self):
        print("Pregenerating data from DataGenerator")
        batch_size_backup = self.batch_size
        self.batch_size *= len(self)
        data = self[0]
        self.batch_size = batch_size_backup
        return data


def get_dataset_c1(trainingGenerator):
    def gen():
        for i in range(len(trainingGenerator)):
            yield trainingGenerator[i]

    return tf.data.Dataset.from_generator(gen, output_signature=(
        {
            "rho": tf.TensorSpec(shape=(trainingGenerator.batch_size, trainingGenerator.inputShape[0]), dtype=tf.float32),
        },
        {
            "c1": tf.TensorSpec(shape=(trainingGenerator.batch_size, 1), dtype=tf.float32),
        }
    )).prefetch(tf.data.AUTOTUNE)


def get_dataset_c1b(pc_data, inputShape):
    def calc_c1b(rho, mu, T):
        return np.log(rho) - mu / T

    training_inputs_c1b = {
        "rho": np.array([np.full(inputShape, pc["rhob"]) for pc in pc_data]),
    }

    training_output_c1b = {
        "c1": np.array([calc_c1b(pc["rhob"], pc["mu"], pc["T"]) for pc in pc_data if np.isfinite(calc_c1b(pc["rhob"], pc["mu"], pc["T"]))])
    }

    return tf.data.Dataset.from_tensor_slices((training_inputs_c1b, training_output_c1b)).shuffle(buffer_size=1024, reshuffle_each_iteration=True).repeat(64).batch(16).prefetch(tf.data.AUTOTUNE)


def get_dataset_c2x(pc_data, windowSigma, inputShape):
    def construct_c2x(c2x, xs):
        c2x = c2x[xs<windowSigma]
        result = np.concatenate((c2x[:0:-1], c2x))
        if result.shape != inputShape:
            raise ValueError(f"The pair-correlation matching is not commensurable with the model. Is the discretization the same?")
        return result

    training_inputs_pc = {
        "rho": np.array([np.full(inputShape, pc["rhob"]) for pc in pc_data]),
    }

    training_output_pc = {
        "c2": np.array([construct_c2x(pc["c2x"], pc["xs"]) for pc in pc_data])
    }

    return tf.data.Dataset.from_tensor_slices((training_inputs_pc, training_output_pc)).shuffle(buffer_size=1024, reshuffle_each_iteration=True).repeat().batch(16).prefetch(tf.data.AUTOTUNE)

