import tensorflow as tf
import numpy as np

def _index_predictions(predictions, labels):
    '''
    Indexes predictions, a [batch_size, num_classes]-shaped tensor,
    by labels, a [batch_size]-shaped tensor that indicates which
    class each sample should be indexed by.

    Args:
        predictions: A [batch_size, num_classes]-shaped tensor. The input to a model.
        labels: A [batch_size, num_classes]-shaped tensor.
                The tensor used to index predictions, in one-hot encoding form.
    Returns:
        A tensor of shape [batch_size] representing the predictions indexed by the labels.
    '''
    current_batch_size = tf.shape(predictions)[0]
    sample_indices = tf.range(current_batch_size)
    sparse_labels  = tf.argmax(labels, axis=-1)
    indices_tensor = tf.stack([sample_indices, tf.cast(sparse_labels, tf.int32)], axis=1)
    predictions_indexed = tf.gather_nd(predictions, indices_tensor)
    return predictions_indexed

@tf.function
def gradients(inputs, labels, model, index_true_class=True, multiply_by_input=False):
    '''
    Computes the gradients of the output with respect to the input. Optionally mulitplies those
    gradients by the input to the model.

    Args:
        inputs: A [batch_size, ...]-shaped tensor. The input to a model.
        labels: A [batch_size]-shaped tensor. The true class labels, assuming a multi-class problem.
        model:  A tf.keras.Model object, or a subclass object thereof.
        index_true_class: Whether or not to take the gradients of the output with respect to the true
            class. True by default. This should be set to True in the multi-class setting, and False
            in the regression setting.
        multiply_by_input: Whether or not to multiply the gradients by the input to the model.
            Defaults to False.
    Returns:
        A tensor the same shape as the input representing the gradients of the output with
        respect to the input.
    '''
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = model(inputs, training=True)

        if index_true_class:
            predictions_indexed = _index_predictions(predictions, labels)
        else:
            predictions_indexed = predictions

    input_gradients = tape.gradient(predictions_indexed, inputs)

    if multiply_by_input:
        input_gradients = input_gradients * inputs

    return input_gradients

@tf.function
def gradients_multi_output(inputs, model, num_classes, multiply_by_input=False):
    '''
    Computes the gradients of the output with respect to the input. Optionally mulitplies those
    gradients by the input to the model.

    Args:
        inputs: A [batch_size, ...]-shaped tensor. The input to a model.
        model:  A tf.keras.Model object, or a subclass object thereof.
        num_classes: The numver of classes to take the gradient with respect to
        multiply_by_input: Whether or not to multiply the gradients by the input to the model.
            Defaults to False.
    Returns:
        A tensor the same shape as the input representing the gradients of the output with
        respect to the input.
    '''
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inputs)
        predictions = model(inputs, training=True)

        predictions_indexed = []
        for output_class in range(num_classes):
            predictions_indexed.append(predictions[:, output_class])

    gradients_array = []
    for output_class in range(num_classes):
        input_gradients = tape.gradient(predictions_indexed[output_class], inputs)
        if multiply_by_input:
            input_gradients = input_gradients * inputs

        gradients_array.append(input_gradients)

    del tape
    stacked_gradients = tf.stack(gradients_array, axis=1)

    return stacked_gradients

@tf.function
def expected_gradients(inputs, labels, model, index_true_class=True):
    '''
    Given a batch of inputs and labels, and a model,
    symbolically computes a single sample of expected gradients.

    Args:
        inputs: A [batch_size, ...]-shaped tensor. The input to a model.
        labels: A [batch_size, num_classes]-shaped tensor.
                The true class labels in one-hot encoding form,
                assuming a multi-class problem.
        model:  A tf.keras.Model object, or a subclass object thereof.
        index_true_class: Whether or not to take the gradients of the output with respect to the true
            class. True by default. This should be set to True in the multi-class setting, and False
            in the regression setting.
    Returns:
        A tensor the same shape as the input representing a single sample of expected gradients
        of the output of the model with respect to the input.
    '''
    current_batch_size = tf.shape(inputs)[0]

    #Here we have to compute the interpolated input into the model
    references  = tf.roll(inputs, shift=1, axis=0)
    # shape should be (k, 275, 1) instead of (current_batch_size, 1, 1)?
    alphas      = tf.random.uniform(shape=(current_batch_size, 275, 1), minval=0.0, maxval=1.0, dtype=tf.float32)
    interpolated_inputs = alphas * inputs + (1.0 - alphas) * references

    with tf.GradientTape() as tape:
        tape.watch(interpolated_inputs)
        predictions = model(interpolated_inputs, training=True)

        if index_true_class:
            predictions_indexed = _index_predictions(predictions, labels)
        else:
            predictions_indexed = predictions

    input_gradients = tape.gradient(predictions_indexed, interpolated_inputs)
    difference_from_reference = inputs - references
    expected_gradients = input_gradients * difference_from_reference
    return expected_gradients

@tf.function
def integrated_gradients(inputs, labels, model, index_true_class=True):
    '''
    Given a batch of inputs and labels, and a model,
    symbolically computes a single sample of expected gradients.

    Args:
        inputs: A [batch_size, ...]-shaped tensor. The input to a model.
        labels: A [batch_size, num_classes]-shaped tensor.
                The true class labels in one-hot encoding form,
                assuming a multi-class problem.
        model:  A tf.keras.Model object, or a subclass object thereof.
        index_true_class: Whether or not to take the gradients of the output with respect to the true
            class. True by default. This should be set to True in the multi-class setting, and False
            in the regression setting.
    Returns:
        A tensor the same shape as the input representing a single sample of expected gradients
        of the output of the model with respect to the input.
    '''
    current_batch_size = tf.shape(inputs)[0]

    #Here we have to compute the interpolated input into the model

    #references  = tf.roll(inputs, shift=1, axis=0)
    references = tf.zeros(shape=(current_batch_size, 275, 1), dtype=tf.float32)
    # shape should be (k, 275, 1) instead of (current_batch_size, 1, 1)?
    alphas      = tf.random.uniform(shape=(current_batch_size, 275, 1), minval=0.0, maxval=1.0, dtype=tf.float32)
    interpolated_inputs = alphas * inputs + (1.0 - alphas) * references

    with tf.GradientTape() as tape:
        tape.watch(interpolated_inputs)
        print(interpolated_inputs.shape)
        predictions = model(interpolated_inputs, training=True)
        print(predictions.shape)
        print('here')

        if index_true_class:
            predictions_indexed = _index_predictions(predictions, labels)
        else:
            predictions_indexed = predictions

    input_gradients = tape.gradient(predictions_indexed, interpolated_inputs)
    difference_from_reference = inputs - references
    integrated_gradients = input_gradients * difference_from_reference
    return integrated_gradients

########################################################################################################################################
def interpolate_images(baseline,
                       inputs,
                       alpha):
    # alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    # baseline_x = tf.expand_dims(baseline, axis=0)
    # input_x = tf.expand_dims(image, axis=0)
    # delta = input_x - baseline_x
    # images = baseline_x +  alphas_x * delta
    # print('int im inputs', inputs.shape)
    # print('int im alpha')
    interpolated_inputs = alpha * inputs + (1.0 - alpha) * baseline
    return interpolated_inputs

def compute_gradients(interpolated_inputs, labels, model):
    #print('com gra', interpolated_inputs.shape)
    with tf.GradientTape() as tape:
        tape.watch(interpolated_inputs)
        predictions = model(interpolated_inputs)
    
        predictions_indexed = _index_predictions(predictions, labels)
    return tape.gradient(predictions_indexed, interpolated_inputs)

def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients

def integrated_gradients_tf(baseline,
                         inputs,
                         labels,
                         model,
                         m_steps=4,
                         batch_size=128):
    # Generate alphas.
    #alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
    alphas = [0, 0.25, 0.5, 0.75, 1]

    # Collect gradients.    
    gradient_batches = []

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in alphas:
        
        # print('main ig', inputs.shape)
        # print('main ig alphas', alphas)
        # print('main ig alph', alpha)
        
        gradient_batch = one_batch(baseline, inputs, alpha, labels, model)
        gradient_batches.append(gradient_batch)

    # Concatenate path gradients together row-wise into single tensor.
    total_gradients = tf.concat(gradient_batches, axis=0)

    # Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)

    # Scale integrated gradients with respect to input.
    integrated_gradients = (inputs - baseline) * avg_gradients

    return integrated_gradients

@tf.function
def one_batch(baseline, inputs, alpha, labels, model):
    # Generate interpolated inputs between baseline and input.
#/print('one batch', inputs.shape)
    interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                       inputs=inputs,
                                                       alpha=alpha)

    # Compute gradients between model outputs and interpolated inputs.
    gradient_batch = compute_gradients(interpolated_inputs=interpolated_path_input_batch,
                                       labels=labels, model=model)
    return gradient_batch
#############################################################################################################################

@tf.function
def integrated_gradients_testing(inputs, labels, model, index_true_class=True, alpha_val=0):
    '''
    Given a batch of inputs and labels, and a model,
    symbolically computes a single sample of expected gradients.

    Args:
        inputs: A [batch_size, ...]-shaped tensor. The input to a model.
        labels: A [batch_size, num_classes]-shaped tensor.
                The true class labels in one-hot encoding form,
                assuming a multi-class problem.
        model:  A tf.keras.Model object, or a subclass object thereof.
        index_true_class: Whether or not to take the gradients of the output with respect to the true
            class. True by default. This should be set to True in the multi-class setting, and False
            in the regression setting.
    Returns:
        A tensor the same shape as the input representing a single sample of expected gradients
        of the output of the model with respect to the input.
    '''
    current_batch_size = tf.shape(inputs)[0]

    #Here we have to compute the interpolated input into the model

    #references  = tf.roll(inputs, shift=1, axis=0)
    references = tf.zeros(shape=(current_batch_size, 275, 1), dtype=tf.float32)
    # shape should be (k, 275, 1) instead of (current_batch_size, 1, 1)?
    #alphas      = tf.random.uniform(shape=(current_batch_size, 275, 1), minval=0.0, maxval=1.0, dtype=tf.float32)
    interpolated_inputs = alpha_val * inputs + (1.0 - alpha_val) * references

    with tf.GradientTape() as tape:
        tape.watch(interpolated_inputs)
        #print(interpolated_inputs.shape)
        predictions = model(interpolated_inputs, training=True)
        #print(predictions.shape)
        #print('here :)')

        if index_true_class:
            predictions_indexed = _index_predictions(predictions, labels)
        else:
            predictions_indexed = predictions

    input_gradients = tape.gradient(predictions_indexed, interpolated_inputs)

    del predictions, predictions_indexed, interpolated_inputs, tape
    # difference_from_reference = inputs - references
    # integrated_gradients = input_gradients * difference_from_reference
    return input_gradients


# @tf.function
# def integrated_gradients_testing(inputs, labels, model, index_true_class=True, alpha_val):
#     '''
#     Given a batch of inputs and labels, and a model,
#     symbolically computes a single sample of expected gradients.

#     Args:
#         inputs: A [batch_size, ...]-shaped tensor. The input to a model.
#         labels: A [batch_size, num_classes]-shaped tensor.
#                 The true class labels in one-hot encoding form,
#                 assuming a multi-class problem.
#         model:  A tf.keras.Model object, or a subclass object thereof.
#         index_true_class: Whether or not to take the gradients of the output with respect to the true
#             class. True by default. This should be set to True in the multi-class setting, and False
#             in the regression setting.
#     Returns:
#         A tensor the same shape as the input representing a single sample of expected gradients
#         of the output of the model with respect to the input.
#     '''
#     current_batch_size = tf.shape(inputs)[0]

#     #Here we have to compute the interpolated input into the model

#     #references  = tf.roll(inputs, shift=1, axis=0)
#     references = tf.zeros(shape=(current_batch_size, 275, 1), dtype=tf.float32)

#     interpolated_gradients = tf.zeros_like(references)
#     # shape should be (k, 275, 1) instead of (current_batch_size, 1, 1)?
#     # alphas      = tf.random.uniform(shape=(current_batch_size, 275, 1), minval=0.0, maxval=1.0, dtype=tf.float32)

#     alphas = [0, 0.2, 0.4, 0.6, 0.8, 1]
#     #interpolated_gradients = []

#     with tf.GradientTape() as tape:

#     for i in range(len(alphas)):

#         print(i)
#         print(references.shape)
#         print(inputs.shape)
#         interpolated_inputs = alphas[i] * inputs + (1.0 - alphas[i]) * references
#         print(interpolated_inputs.shape)

#             print('before tape watch')
#             tape.watch(interpolated_inputs)
#             print('after tape watch')
#             predictions = model(interpolated_inputs, training=True)
#             print('after predictions')

#             if index_true_class:
#                 predictions_indexed = _index_predictions(predictions, labels)
#             else:
#                 predictions_indexed = predictions

#         input_gradients = tape.gradient(predictions_indexed, interpolated_inputs)
#         print('input gradients', input_gradients.shape)
#         # interpolated_gradients.append(input_gradients)
#         interpolated_gradients = interpolated_gradients + input_gradients
#         print('interp grads', interpolated_gradients.shape)

#         del predictions, input_gradients, predictions_indexed, interpolated_inputs, tape


#     #print(np.asarray(interpolated_gradients).shape)
#     # riemann_approx = np.sum(interpolated_gradients) / 5
#     # print(riemann_approx.shape)

#     # difference_from_reference = inputs - references
#     # integrated_gradients = riemann_approx * difference_from_reference
#     return integrated_gradients

def expected_gradients_full(inputs, references, model, k=100, index_true_class=False, labels=None):
    '''
    Given a batch of inputs and labels, and a model,
    symbolically computes expected gradients with k references.

    Args:
        inputs: A [batch_size, ...]-shaped tensor. The input to a model.
        references: A numpy array representing background training data to sample from.
        model:  A tf.keras.Model object, or a subclass object thereof.
        k: The number of samples to use when computing expected gradients.
        index_true_class: Whether or not to take the gradients of the output with respect to the true
            class. True by default. This should be set to True in the multi-class setting, and False
            in the regression setting.
        labels: A [batch_size, num_classes]-shaped tensor.
                The true class labels in one-hot encoding,
                assuming a multi-class problem.
    Returns:
        A tensor the same shape as the input representing the expected gradients
        feature attributions with respect to the output predictions.
    '''
    eg_array = []

    for i in range(tf.shape(inputs)[0]):
        sample_indices = np.random.choice(references.shape[0], size=k, replace=False)
        sample_references = references[sample_indices]
        # shape should be (k, 275, 1) instead of (k, 1, 1)?
        alphas = tf.random.uniform(shape=(k, 275, 1), minval=0.0, maxval=1.0, dtype=tf.float32)
        current_input  = tf.expand_dims(inputs[i], axis=0)

        interpolated_inputs = alphas * current_input + (1.0 - alphas) * sample_references

        with tf.GradientTape() as tape:
            tape.watch(interpolated_inputs)
            predictions = model(interpolated_inputs, training=False)

            if index_true_class:
                current_labels = tf.expand_dims(labels[i, :], axis=0)
                current_labels = tf.tile(current_labels, multiples=(k, 1))
                predictions_indexed = _index_predictions(predictions, current_labels)
            else:
                predictions_indexed = predictions

        input_gradients = tape.gradient(predictions_indexed, interpolated_inputs)
        difference_from_reference = current_input - sample_references
        expected_gradients_samples = input_gradients * difference_from_reference
        expected_gradients = tf.reduce_mean(expected_gradients_samples, axis=0)
        eg_array.append(expected_gradients)

    return tf.stack(eg_array, axis=0)

def expected_gradients_multi_output(inputs, references, model, num_classes, k=100):
    '''
    Given a batch of inputs and labels, and a model,
    symbolically computes expected gradients with k references. Unlike
    expected_gradients_full, this function is used when you want the
    expected gradients values with respect to all output classes, not just a single one.

    Args:
        inputs: A [batch_size, ...]-shaped tensor. The input to a model.
        references: A numpy array representing background training data to sample from.
        model:  A tf.keras.Model object, or a subclass object thereof.
        num_classes: The number of classes to take expected gradients with respect to.
        k: The number of samples to use when computing expected gradients.
    Returns:
        A tensor the same shape as the input representing the expected gradients
        feature attributions with respect to the output predictions.
    '''
    alpha_shape = 0
    eg_array = []
    for i in range(tf.shape(inputs)[0]):
        sample_indices = np.random.choice(references.shape[0], size=k, replace=False)
        sample_references = references[sample_indices]

        predictions_indexed = []
        # shape should be (k, 275, 1) instead of (k, 1, 1)?
        alphas = tf.random.uniform(shape=(k, 275, 1), minval=0.0, maxval=1.0, dtype=tf.float32)
        alpha_shape = alphas.shape
        current_input = tf.expand_dims(inputs[i], axis=0)
        interpolated_inputs = alphas * current_input + (1.0 - alphas) * sample_references

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(interpolated_inputs)

            predictions = model(interpolated_inputs, training=False)
            for output_class in range(num_classes):
                predictions_indexed.append(predictions[:, output_class])

        sample_eg_array = []
        for output_class in range(num_classes):
            input_gradients = tape.gradient(predictions_indexed[output_class], interpolated_inputs)
            difference_from_reference = current_input - sample_references
            expected_gradients_samples = input_gradients * difference_from_reference
            expected_gradients = tf.reduce_mean(expected_gradients_samples, axis=0)
            sample_eg_array.append(expected_gradients)

        del tape
        eg_array.append(tf.stack(sample_eg_array, axis=0))

    return tf.stack(eg_array, axis=0)

def integrated_gradients_multi_output(inputs, references, model, num_classes, k=100):
    '''
    Given a batch of inputs and labels, and a model,
    symbolically computes expected gradients with k references. Unlike
    expected_gradients_full, this function is used when you want the
    expected gradients values with respect to all output classes, not just a single one.

    Args:
        inputs: A [batch_size, ...]-shaped tensor. The input to a model.
        references: A numpy array representing background training data to sample from.
        model:  A tf.keras.Model object, or a subclass object thereof.
        num_classes: The number of classes to take expected gradients with respect to.
        k: The number of samples to use when computing expected gradients.
    Returns:
        A tensor the same shape as the input representing the expected gradients
        feature attributions with respect to the output predictions.
    '''
    alpha_shape = 0
    ig_array = []
    for i in range(tf.shape(inputs)[0]):
        # sample_indices = np.random.choice(references.shape[0], size=k, replace=False)
        # sample_references = references[sample_indices]

        sample_references = tf.zeros(shape=(k, 275, 1), dtype=tf.float32)

        predictions_indexed = []
        # shape should be (k, 275, 1) instead of (k, 1, 1)?
        alphas = tf.random.uniform(shape=(k, 275, 1), minval=0.0, maxval=1.0, dtype=tf.float32)
        alpha_shape = alphas.shape
        current_input = tf.expand_dims(inputs[i], axis=0)
        interpolated_inputs = alphas * current_input + (1.0 - alphas) * sample_references

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(interpolated_inputs)

            predictions = model(interpolated_inputs, training=False)
            for output_class in range(num_classes):
                predictions_indexed.append(predictions[:, output_class])

        sample_ig_array = []
        for output_class in range(num_classes):
            input_gradients = tape.gradient(predictions_indexed[output_class], interpolated_inputs)
            difference_from_reference = current_input - sample_references
            integrated_gradients_samples = input_gradients * difference_from_reference
            integrated_gradients = tf.reduce_mean(integrated_gradients_samples, axis=0)
            sample_ig_array.append(integrated_gradients)

        del tape
        ig_array.append(tf.stack(sample_ig_array, axis=0))

    return tf.stack(ig_array, axis=0)