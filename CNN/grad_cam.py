# #Define Some Functions :
# from skimage.io import imread, imsave
#
# last_conv_layer_name = model.layers[-12].name
#
# def get_img_array(img_path, size = (224 , 224)):
#     img = tf.keras.utils.load_img(img_path, target_size=size)
#     array = tf.keras.utils.img_to_array(img)
#     array = np.expand_dims(array, axis=0)
#     return array
#
#
#
# def make_gradcam_heatmap(img_array, model = model , last_conv_layer_name = last_conv_layer_name, pred_index=None):
#     # First, we create a model that maps the input image to the activations
#     # of the last conv layer as well as the output predictions
#     grad_model = keras.models.Model(
#         model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
#     )
#
#     # Then, we compute the gradient of the top predicted class for our input image
#     # with respect to the activations of the last conv layer
#     with tf.GradientTape() as tape:
#         last_conv_layer_output, preds = grad_model(img_array)
#         if pred_index is None:
#             pred_index = tf.argmax(preds[0])
#         class_channel = preds[:, pred_index]
#
#     # This is the gradient of the output neuron (top predicted or chosen)
#     # with regard to the output feature map of the last conv layer
#     grads = tape.gradient(class_channel, last_conv_layer_output)
#
#     # This is a vector where each entry is the mean intensity of the gradient
#     # over a specific feature map channel
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#
#     # We multiply each channel in the feature map array
#     # by "how important this channel is" with regard to the top predicted class
#     # then sum all the channels to obtain the heatmap class activation
#     last_conv_layer_output = last_conv_layer_output[0]
#     heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#
#     # For visualization purpose, we will also normalize the heatmap between 0 & 1
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     return heatmap.numpy()
#
# def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4 , view = False):
#     # Load the original image
#     img = tf.keras.utils.load_img(img_path)
#     img = tf.keras.utils.img_to_array(img)
#
#     # Rescale heatmap to a range 0-255
#     heatmap = np.uint8(255 * heatmap)
#
#     # Use jet colormap to colorize heatmap
#     jet = mpl.colormaps["jet"]
#
#     # Use RGB values of the colormap
#     jet_colors = jet(np.arange(256))[:, :3]
#     jet_heatmap = jet_colors[heatmap]
#
#     # Create an image with RGB colorized heatmap
#     jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
#     jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
#     jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
#
#     # Superimpose the heatmap on original image
#     superimposed_img = jet_heatmap * alpha + img
#     superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
#
#     # Save the superimposed image
#     superimposed_img.save(cam_path)
#
#     # Display Grad CAM
#     if view :
#         display(Image(cam_path))
#
# def decode_predictions(preds):
#     classes = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis',
#        'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps',
#        'ulcerative-colitis']
#     prediction = classes[np.argmax(preds)]
#     return prediction
#
# def make_prediction (img_path , model = model , last_conv_layer_name = last_conv_layer_name , campath = "cam.jpeg" , view = False):
#     img = get_img_array(img_path = img_path)
#     img_array = get_img_array(img_path, size=(224 , 224))
#     preds = model.predict(img_array)
#     heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
#     save_and_display_gradcam(img_path, heatmap , cam_path=campath , view = view)
#     return [campath , decode_predictions(preds)]

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
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * interpolant + img_array * (1 - interpolant)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    if plot_results:
        plt.imshow(superimposed_img)
        plt.title('GradCAM')
        plt.axis('off')
        plt.show()

    return superimposed_img

