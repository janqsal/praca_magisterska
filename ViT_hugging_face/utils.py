import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from transformers import ViTFeatureExtractor, ViTForImageClassification
from sklearn.metrics import classification_report , confusion_matrix
import itertools
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from utils import *
from grad_cam import *

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


def get_predictions_and_labels(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            _, predicted_labels = torch.max(logits, 1)

            all_predictions.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_predictions, all_labels

def evaluate_model(model, test_loader, device):
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            logits = outputs['logits']
            _, predicted = torch.max(logits, 1)

            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    report = classification_report(y_true, y_pred, target_names=test_loader.dataset.dataset.classes)
    print(report)

def show_loss_accuracy_plots(history):
    tr_acc = history['train_accuracy']
    tr_loss = history['train_loss']
    val_acc = history['val_accuracy']
    val_loss = history['val_loss']
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]
    Epochs = range(1, len(tr_acc) + 1)
    loss_label = f'Najlepsza epoka (strata) = {index_loss + 1}'
    acc_label = f'Najlepsza epoka (dokładność) = {index_acc + 1}'

    plt.figure(figsize=(20, 8))
    plt.style.use('fivethirtyeight')

    # Wykres strata
    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label='Zbiór uczący - strata')
    plt.plot(Epochs, val_loss, 'g', label='Zbiór walidacyjny - strata')
    plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
    plt.title('Funkcja straty dla zbioru uczącego oraz walidacyjnego')
    plt.xlabel('Epoki')
    plt.ylabel('Strata')
    plt.legend()

    # Wykres dokładność
    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label='Dokładność zbioru uczącego')
    plt.plot(Epochs, val_acc, 'g', label='Dokładność zbioru walidacyjnego')
    plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
    plt.title('Dokładność dla zbioru uczącego oraz walidacyjnego')
    plt.xlabel('Epoki')
    plt.ylabel('Dokładność')
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_and_validate(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs, device, save_path, model_name):
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }
    best_validation_accuracy = 0.0
    best_model = None

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        train_loss = running_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)

        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.logits, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        val_loss = val_running_loss / len(valid_loader)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        if val_accuracy > best_validation_accuracy:
            best_validation_accuracy = val_accuracy
            best_model = model.state_dict()
            best_model_path = os.path.join(save_path, model_name)
            torch.save(best_model, best_model_path)
            print(f"Najlepszy model został zapisany do {best_model_path}")

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}% "
              f"Czas trwania epoki: {epoch_duration:.2f}s")

        scheduler.step()

    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"Całkowity czas treningu: {total_training_time:.2f}s")

    if best_model is not None:
        model.load_state_dict(best_model)

    return model, history
