import warnings
warnings.filterwarnings('ignore')
import random
from torchvision import transforms
from datasets import load_dataset
from pytorch_grad_cam import run_dff_on_image, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import cv2
import torch
from typing import List, Callable, Optional
import matplotlib.pyplot as plt
import io
from PIL import Image as PILImage


class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits


""" Translate the category name to the category index.
    Some models aren't trained on Imagenet but on even larger datasets,
    so we can't just assume that 761 will always be remote-control.

"""


def category_name_to_index(model, category_name):
    name_to_index = dict((v, k) for k, v in model.config.id2label.items())
    return name_to_index[category_name]


""" Helper function to run GradCAM on an image and create a visualization.
    (note to myself: this is probably useful enough to move into the package)
    If several targets are passed in targets_for_gradcam,
    e.g different categories,
    a visualization for each of them will be created.

"""


def run_grad_cam_on_image(model: torch.nn.Module,
                          target_layer: torch.nn.Module,
                          targets_for_gradcam: List[Callable],
                          reshape_transform: Optional[Callable],
                          input_tensor: torch.nn.Module,
                          input_image: Image,
                          method: Callable = GradCAM):
    with method(model=HuggingfaceToTensorModelWrapper(model),
                target_layers=[target_layer],
                reshape_transform=reshape_transform) as cam:
        # Replicate the tensor for each of the categories we want to create Grad-CAM for:
        repeated_tensor = input_tensor[None, :].repeat(len(targets_for_gradcam), 1, 1, 1)

        batch_results = cam(input_tensor=repeated_tensor,
                            targets=targets_for_gradcam)
        results = []
        for grayscale_cam in batch_results:
            visualization = show_cam_on_image(np.float32(input_image) / 255,
                                              grayscale_cam,
                                              use_rgb=True)
            # Make it weight less in the notebook:
            visualization = cv2.resize(visualization,
                                       (visualization.shape[1] // 2, visualization.shape[0] // 2))
            results.append(visualization)
        return np.hstack(results)


def print_top_categories(model, img_tensor, top_k=5):
    logits = model(img_tensor.unsqueeze(0)).logits
    indices = logits.cpu()[0, :].detach().numpy().argsort()[-top_k:][::-1]
    for i in indices:
        print(f"Predicted class {i}: {model.config.id2label[i]}")

def reshape_transform_vit_huggingface(x, img_size=224, patch_size=16):

    # Obliczenie liczby łatek na bok obrazu
    n_patches_side = img_size // patch_size

    # Pomijamy token klasy i zmieniamy kształt
    activations = x[:, 1:, :]
    activations = activations.view(activations.shape[0],
                                   n_patches_side, n_patches_side,
                                   activations.shape[2])

    # Transpozycja do formatu CHW
    activations = activations.transpose(2, 3).transpose(1, 2)
    return activations

def display_images_with_gradcam(model, dataset, reshape_transform, num_classes, method, image_size=(224, 224),
                                custom_labels=None):
    # Automatyczne określenie warstwy docelowej dla Grad-CAM
    target_layer_gradcam = model.vit.encoder.layer[-2].output

    # Iteracja przez klasy i wybór losowego obrazu dla każdej klasy
    for class_id in range(num_classes):
        # Zbierz wszystkie obrazy należące do aktualnej klasy
        class_images = [(image, label) for image, label in dataset if label == class_id]

        # Jeśli nie ma obrazów dla klasy, kontynuuj pętlę
        if not class_images:
            continue

        # Wybierz losowy obraz dla klasy
        image, label = random.choice(class_images)

        # Przeskaluj obraz do odpowiedniego rozmiaru
        image_resized = image.resize(image_size)
        tensor_resized = transforms.ToTensor()(image_resized)

        # Ustawienie celów dla Grad-CAM
        targets_for_gradcam = [ClassifierOutputTarget(class_id)]

        # Uruchomienie Grad-CAM
        grad_cam_result = run_grad_cam_on_image(model=model,
                                                target_layer=target_layer_gradcam,
                                                targets_for_gradcam=targets_for_gradcam,
                                                input_tensor=tensor_resized,
                                                input_image=image_resized,
                                                reshape_transform=reshape_transform,
                                                method=method)

        buffer_grad_cam = io.BytesIO()
        PILImage.fromarray(grad_cam_result).save(buffer_grad_cam, format='PNG')
        buffer_grad_cam.seek(0)

        display_label = custom_labels[label] if custom_labels else label
        print(f"Label: {display_label}")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image_resized)
        ax[0].axis('off')
        ax[0].set_title('Original Image')

        ax[1].imshow(PILImage.open(buffer_grad_cam))
        ax[1].axis('off')
        ax[1].set_title(f'{method}')

        plt.show()