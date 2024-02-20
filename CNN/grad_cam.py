import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
from skimage.transform import resize
from tensorflow.keras.models import Model


def VizGradCAM(model, image, interpolant=0.5, plot_results=True):
    """VizGradCAM - Displays GradCAM based on Keras / TensorFlow models
    using the gradients from the last convolutional layer. This function
    should work with all Keras Application listed here:
    https://keras.io/api/applications/

    Parameters:
    model (keras.model): Compiled Model with Weights Loaded
    image: Image to Perform Inference On
    plot_results (boolean): True - Function Plots using PLT
                            False - Returns Heatmap Array

    Returns:
    Heatmap Array?
    """
    assert (
        interpolant > 0 and interpolant < 1
    ), "Heatmap Interpolation Must Be Between 0 - 1"

    last_conv_layer = next(
        x for x in model.layers[::-1] if isinstance(x, K.layers.Conv2D)
    )
    target_layer = model.get_layer(last_conv_layer.name)

    original_img = image
    img = np.expand_dims(original_img, axis=0)
    prediction = model.predict(img)

    prediction_idx = np.argmax(prediction)

    # Compute Gradient of Top Predicted Class
    with tf.GradientTape() as tape:
        gradient_model = Model([model.inputs], [target_layer.output, model.output])
        conv2d_out, prediction = gradient_model(img)
        # Obtain the Prediction Loss
        loss = prediction[:, prediction_idx]

    # Gradient() computes the gradient using operations recorded
    # in context of this tape
    gradients = tape.gradient(loss, conv2d_out)

    # Obtain the Output from Shape [1 x H x W x CHANNEL] -> [H x W x CHANNEL]
    output = conv2d_out[0]

    weights = tf.reduce_mean(gradients[0], axis=(0, 1))

    activation_map = np.zeros(output.shape[0:2], dtype=np.float32)

    for idx, weight in enumerate(weights):
        activation_map += weight * output[:, :, idx]

    activation_map = cv2.resize(
        activation_map.numpy(), (original_img.shape[1], original_img.shape[0])
    )

    activation_map = np.maximum(activation_map, 0)

    activation_map = (activation_map - activation_map.min()) / (
        activation_map.max() - activation_map.min()
    )
    activation_map = np.uint8(255 * activation_map)

    heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)

    original_img = np.uint8(
        (original_img - original_img.min())
        / (original_img.max() - original_img.min())
        * 255
    )

    cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    plt.rcParams["figure.dpi"] = 100

    if plot_results == True:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].imshow(original_img)
        axs[0].axis('off')
        axs[0].set_title('Pierwotny obraz')

        axs[1].imshow(np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant)))
        axs[1].axis('off')
        axs[1].set_title('GradCAM')

        plt.show()
    else:
        return cvt_heatmap


def VizGradCAM_for_feature_map(model, img_array, interpolant=0.5, plot_results=False):
    last_conv_layer = next(x for x in model.layers[::-1] if isinstance(x, tf.keras.layers.Conv2D))
    grads_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grads_model(np.array([img_array]))
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(conv_outputs[0], pooled_grads[..., tf.newaxis])
    cam = cam[:, :, 0]

    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    cam = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
    heatmap = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(255 - heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * interpolant + img_array * (1 - interpolant)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    if plot_results:
        plt.imshow(superimposed_img)
        plt.title('GradCAM')
        plt.axis('off')
        plt.show()

    return superimposed_img


def display_feature_map_with_predictions_and_gradcam(model, dataset, class_names, VizGradCAM_for_feature_map,
                                                     target_class=None):
    class_names_list = class_names.tolist() if isinstance(class_names, np.ndarray) else class_names
    found = False

    if target_class is not None and target_class in class_names_list:
        target_class_index = class_names_list.index(target_class)

        for images, labels in dataset.unbatch().as_numpy_iterator():
            label_index = np.argmax(labels)
            if label_index == target_class_index:
                image = images.astype('uint8')
                true_class = class_names[label_index]
                found = True
                break
    if not found:
        for images, labels in dataset.shuffle(1024).take(1).unbatch().as_numpy_iterator():
            image = images.astype('uint8')
            true_label_index = np.argmax(labels)
            true_class = class_names[true_label_index]
            break

    if image is not None:
        image_expanded = np.expand_dims(image, axis=0)
        prediction_scores = model.predict(image_expanded)
        predicted_index = np.argmax(prediction_scores[0])
        predicted_class = class_names[predicted_index]

        gradcam_img = VizGradCAM_for_feature_map(model, image, interpolant=0.5, plot_results=False)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f'Faktyczna klasa: {true_class}\nPrzewidziana: {predicted_class}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(gradcam_img)
        plt.title(f'GradCAM')
        plt.axis('off')
        plt.show()

        conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        activation_model = Model(inputs=model.input, outputs=[layer.output for layer in conv_layers])
        feature_maps = activation_model.predict(image_expanded)

        num_layers = len(feature_maps)
        cols = 3
        rows = (num_layers + cols - 1) // cols
        plt.figure(figsize=(cols * 3, rows * 3))
        for i, feature_map in enumerate(feature_maps):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(feature_map[0, :, :, 0], cmap='viridis')
            plt.title(conv_layers[i].name)
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("Nie znaleziono obrazu z podanej klasy.")


def predict_and_gradcam(model, dataset, class_names, VizGradCAM, target_class=None):
    class_names_list = class_names.tolist() if isinstance(class_names, np.ndarray) else class_names
    found = False

    if target_class is not None and target_class in class_names_list:
        target_class_index = class_names_list.index(target_class)

        for images, labels in dataset.unbatch().filter(lambda img, lbl: tf.argmax(lbl) == target_class_index).shuffle(1024).take(1):
            image = images.numpy().astype('uint8')
            label = labels.numpy()
            found = True
            break

    if not found:
        images, labels = next(iter(dataset.shuffle(1024).take(1).unbatch()))
        image = images.numpy().astype('uint8')
        label = labels.numpy()

    image_expanded = np.expand_dims(image, axis=0)
    prediction_scores = model.predict(image_expanded)
    predicted_index = np.argmax(prediction_scores[0])
    true_class = class_names[np.argmax(label)]
    predicted_class = class_names[predicted_index]

    gradcam_img = VizGradCAM(model, image)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f'Faktyczna klasa: {true_class}\nPrzewidziana: {predicted_class}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(gradcam_img, cmap='jet', alpha=1)
    plt.title('GradCAM')
    plt.axis('off')
    plt.show()

    return true_class, predicted_class


def display_feature_map_with_predictions_and_gradcam2(model, dataset, class_names, VizGradCAM_for_feature_map,
                                                      target_class=None):
    class_names_list = class_names.tolist() if isinstance(class_names, np.ndarray) else class_names
    found = False

    if target_class is not None and target_class in class_names_list:
        target_class_index = class_names_list.index(target_class)

        for images, labels in dataset.unbatch().as_numpy_iterator():
            label_index = np.argmax(labels)
            if label_index == target_class_index:
                image = images.astype('uint8')
                true_class = class_names[label_index]
                found = True
                break
    if not found:
        for images, labels in dataset.shuffle(1024).take(1).unbatch().as_numpy_iterator():
            image = images.astype('uint8')
            true_label_index = np.argmax(labels)
            true_class = class_names[true_label_index]
            break

    if image is not None:
        image_expanded = np.expand_dims(image, axis=0)
        prediction_scores = model.predict(image_expanded)
        predicted_index = np.argmax(prediction_scores[0])
        predicted_class = class_names[predicted_index]

        gradcam_img = VizGradCAM_for_feature_map(model, image, interpolant=0.5, plot_results=False)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f'Faktyczna klasa: {true_class}\nPrzewidziana: {predicted_class}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(gradcam_img)
        plt.title('GradCAM')
        plt.axis('off')
        plt.show()

        conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        activation_model = Model(inputs=model.input, outputs=[layer.output for layer in conv_layers])
        feature_maps = activation_model.predict(image_expanded)

        for layer_name, feature_map in zip([layer.name for layer in conv_layers], feature_maps):
            n_filters = feature_map.shape[-1]
            n_filters = min(n_filters, 64)
            n_cols = 8
            n_rows = n_filters // n_cols if n_filters % n_cols == 0 else (n_filters // n_cols) + 1

            plt.figure(figsize=(n_cols * 2, n_rows * 2))
            for i in range(n_filters):
                ax = plt.subplot(n_rows, n_cols, i + 1)
                plt.imshow(feature_map[0, :, :, i], cmap='viridis')
                plt.axis('off')
            plt.suptitle(f"Feature maps for layer {layer_name}", fontsize=16)
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.show()
    else:
        print("Nie znaleziono obrazu z podanej klasy.")

