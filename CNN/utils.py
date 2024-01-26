import tensorflow as tf
from sklearn.metrics import classification_report
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

    # Tworzenie DataFrame
    df = pd.DataFrame(data)

    # Tworzenie wykresu
    plt.figure(figsize=(10, 6))
    splot = sns.barplot(x="Klasa", y="Procent", data=df, palette="viridis")

    # Dodanie etykiet z liczbą instancji (jako liczby całkowite)
    for p in splot.patches:
        splot.annotate(format(int(p.get_height() * dataset_size / 100), 'd'),
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha = 'center', va = 'center',
                       xytext = (0, 9),
                       textcoords = 'offset points')

    # Dodanie tytułu i etykiet osi
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

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', figsize=(10, 8)):
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
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_confusion_matrix_percent(cm, classes, title='Confusion Matrix (%)', figsize=(10, 8)):
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=figsize)
    plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm_percent.max() / 2.
    for i, j in itertools.product(range(cm_percent.shape[0]), range(cm_percent.shape[1])):
        plt.text(j, i, f"{cm_percent[i, j]:.2%}",
                 horizontalalignment="center",
                 color="white" if cm_percent[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def make_classification_report(ds, model):
    y_true = []
    y_pred = []
    for images, labels in ds.as_numpy_iterator():
        y_pred_probs = model.predict(images, batch_size=1, verbose=0)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_true.extend(labels)
        y_pred.extend(y_pred_classes)

    y_true_classes = [np.argmax(i) for i in y_true]
    report = classification_report(y_true_classes, y_pred)

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

