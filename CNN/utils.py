import tensorflow as tf
from keras import Model
from sklearn.metrics import classification_report
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Function to count class instances
def count_classes(dataset, class_names):
    class_counts = {class_name: 0 for class_name in class_names}
    for images, labels in dataset.unbatch().take(-1):
        label_index = np.argmax(labels)
        class_name = class_names[label_index]
        class_counts[class_name] += 1
    return class_counts

# Funkcja do rysowania wykresu z użyciem seaborn
def plot_class_distribution_seaborn(class_counts, title, dataset_size):
    labels = list(class_counts.keys())
    counts = np.array(list(class_counts.values()))
    percentages = 100 * counts / dataset_size
    data = {'Klasa': labels, 'Procent': percentages}

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    splot = sns.barplot(x="Klasa", y="Procent", data=df, palette="viridis")

    for p in splot.patches:
        splot.annotate(format(int(p.get_height() * dataset_size / 100), 'd'),
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha = 'center', va = 'center',
                       xytext = (0, 9),
                       textcoords = 'offset points')

    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

def show_9_images_from_ds(ds, class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[tf.argmax(labels[i]).numpy()])
            plt.axis("off")

def data_augmentation(images, data_augmentation_layers):
    for layer in data_augmentation_layers:
        images = layer(images, training=True)
    return images

def show_9_augmented_images(ds, data_augmentation_layers):
    plt.figure(figsize=(10, 10))
    for images, _ in ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images, data_augmentation_layers)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(augmented_images[10]).astype("uint8"))
            plt.axis("off")

def get_predictions_and_labels(model, dataset):
    all_predictions = []
    all_labels = []

    for images, labels in dataset:
        predictions = model.predict(images)
        predicted_labels = np.argmax(predictions, axis=1)
        all_predictions.extend(predicted_labels)
        all_labels.extend(labels.numpy())

    return all_predictions, all_labels

def plot_confusion_matrix(cm, classes, title='Macierz pomyłek', figsize=(10, 8)):
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Faktyczne etykiety')
    plt.xlabel('Przewidziane etykiety')
    plt.show()


def plot_confusion_matrix_percent(cm, classes, title='Confusion Matrix (%)', figsize=(10, 8)):
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)

    plt.colorbar(im, ax=ax)

    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    thresh = cm_percent.max() / 2.

    for i, j in itertools.product(range(cm_percent.shape[0]), range(cm_percent.shape[1])):
        plt.text(j, i, "{:.1%}".format(cm_percent[i, j]),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm_percent[i, j] > thresh else "black")

    for i in range(len(classes) + 1):
        ax.axhline(i - 0.5, linestyle='-', color='black', linewidth=0.5)
        ax.axvline(i - 0.5, linestyle='-', color='black', linewidth=0.5)

    plt.ylabel('Prawdziwe etykiety')
    plt.xlabel('Przewidziane etykiety')

    ax.set_aspect('equal')
    plt.subplots_adjust(bottom=0.2)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def make_classification_report(ds, model, model_name):
    y_true = []
    y_pred = []
    for images, labels in ds.as_numpy_iterator():
        y_pred_probs = model.predict(images, batch_size=1, verbose=0)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_true.extend(labels)
        y_pred.extend(y_pred_classes)

    y_true_classes = [np.argmax(i) for i in y_true]
    report = classification_report(y_true_classes, y_pred, output_dict=True)

    print(f"Raport klasyfikacji dla {model_name}:")
    print(classification_report(y_true_classes, y_pred))

    return report

def show_loss_accuracy_plots(history):
    tr_acc = history.history['accuracy']
    tr_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]
    Epochs = [i+1 for i in range(len(tr_acc))]
    loss_label = f'najlepsza epoka= {str(index_loss + 1)}'
    acc_label = f'najlepsza epoka= {str(index_acc + 1)}'

    plt.figure(figsize= (20, 8))
    plt.style.use('fivethirtyeight')

    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label= 'Zbiór uczący - strata')
    plt.plot(Epochs, val_loss, 'g', label= 'Zbiór walidacyjny - strata')
    plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
    plt.title('Funkcja straty dla zbioru uczącego oraz walidacyjnego')
    plt.xlabel('Epoki')
    plt.ylabel('Strata')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label= 'Dokładność zbioru uczącego')
    plt.plot(Epochs, val_acc, 'g', label= 'Dokładność zbioru walidacyjnego')
    plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
    plt.title('Dokładność dla zbioru uczącego oraz walidacyjnego')
    plt.xlabel('Epoki')
    plt.ylabel('Dokładność')
    plt.legend()

    plt.tight_layout
    plt.show


def find_best_model(models, ds):
    best_scores = {metric: 0 for metric in ['precision', 'recall', 'f1-score', 'accuracy']}
    best_model_names = {metric: None for metric in ['precision', 'recall', 'f1-score', 'accuracy']}
    metrics_translation = {
        'precision': 'precyzja',
        'recall': 'pełność',
        'f1-score': 'wynik F1',
        'accuracy': 'dokładność'
    }

    for model_name, model in models.items():
        report = make_classification_report(ds, model, model_name)

        for metric in best_scores.keys():
            if metric == 'accuracy':
                score = report['accuracy']
            else:
                score = np.mean([report[label][metric] for label in report if
                                 label not in ['accuracy', 'macro avg', 'weighted avg']])

            if score > best_scores[metric]:
                best_scores[metric] = score
                best_model_names[metric] = model_name

    best_models_pl = {metrics_translation[metric]: (best_model_names[metric], best_scores[metric]) for metric in
                      best_scores}

    return best_models_pl


def save_models(folder_name, models):
    base_path = r'C:\Users\Jan\SGH\magisterka\weights'

    full_path = os.path.join(base_path, folder_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    for i, model in enumerate(models, start=1):
        model_path = os.path.join(full_path, f'model{i}.h5')
        print(f"Zapisywanie {model_path}...")  # Informacja o zapisie
        model.save(model_path)


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
        # Przygotowanie obrazu do predykcji
        image_expanded = np.expand_dims(image, axis=0)
        prediction_scores = model.predict(image_expanded)
        predicted_index = np.argmax(prediction_scores[0])
        predicted_class = class_names[predicted_index]

        # Generowanie GradCAM
        gradcam_img = VizGradCAM_for_feature_map(model, image, interpolant=0.5, plot_results=False)

        # Wyświetlanie oryginalnego obrazu i GradCAM
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

        # Mapy cech dla warstw konwolucyjnych
        conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        activation_model = Model(inputs=model.input, outputs=[layer.output for layer in conv_layers])
        feature_maps = activation_model.predict(image_expanded)

        # Wyświetlanie ograniczonej liczby map cech dla wszystkich filtrów
        for layer_name, feature_map in zip([layer.name for layer in conv_layers], feature_maps):
            n_filters = feature_map.shape[-1]
            n_filters = min(n_filters, 64)  # Ograniczenie do maksymalnie 64 filtrów
            n_cols = 8  # Filtry w wierszu
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
        # Przygotowanie obrazu do predykcji
        image_expanded = np.expand_dims(image, axis=0)
        prediction_scores = model.predict(image_expanded)
        predicted_index = np.argmax(prediction_scores[0])
        predicted_class = class_names[predicted_index]

        # Generowanie GradCAM
        gradcam_img = VizGradCAM_for_feature_map(model, image, interpolant=0.5, plot_results=False)

        # Wyświetlanie oryginalnego obrazu, GradCAM i przewidywanej klasy
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

        # Mapy cech dla warstw konwolucyjnych
        conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        activation_model = Model(inputs=model.input, outputs=[layer.output for layer in conv_layers])
        feature_maps = activation_model.predict(image_expanded)

        # Wyświetlanie map cech
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


def classification_report_to_latex(report, class_names):
    """parser do konwertowania raportu klasyfikacji z scikit-learn do kodu LaTeX"""
    if isinstance(class_names, (list, np.ndarray)):
        class_names_dict = {i: name for i, name in enumerate(class_names)}
    else:
        raise ValueError("class_names must be a list or a numpy array")

    latex_code = """
\\begin{table}[h]
\\centering
\\caption{Raport klasyfikacji dla modelu1}
\\label{tab:classification_report}
\\begin{tabular}{lcccc}
\\toprule
Klasy & Precyzja & Czułość & F1-score & Liczebność \\\\
\\midrule
"""
    lines = report.split('\n')
    for line in lines[2:6]:
        parts = line.split()
        class_id = int(parts[0])
        precision = parts[1]
        recall = parts[2]
        f1_score = parts[3]
        support = parts[4]
        latex_code += f"{class_names_dict[class_id]} & {precision} & {recall} & {f1_score} & {support} \\\\\n"

    accuracy_line = lines[7].split()
    macro_avg_line = lines[8].split()
    weighted_avg_line = lines[9].split()

    accuracy = accuracy_line[2]
    macro_precision = macro_avg_line[2]
    macro_recall = macro_avg_line[3]
    macro_f1 = macro_avg_line[4]
    macro_support = macro_avg_line[5]
    weighted_precision = weighted_avg_line[2]
    weighted_recall = weighted_avg_line[3]
    weighted_f1 = weighted_avg_line[4]
    weighted_support = weighted_avg_line[5]

    latex_code += f"\\midrule\nDokładność & & & {accuracy} & {macro_support} \\\\\n"
    latex_code += f"Średnia makro & {macro_precision} & {macro_recall} & {macro_f1} & {macro_support} \\\\\n"
    latex_code += f"Średnia ważona & {weighted_precision} & {weighted_recall} & {weighted_f1} & {weighted_support} \\\\\n"
    latex_code += "\\bottomrule\n\\end{tabular}\n\\end{table}"

    return latex_code
