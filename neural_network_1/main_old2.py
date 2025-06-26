import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# Utwórz folder na wykresy, jeśli nie istnieje
plots_dir = "plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Parametry
image_size = (224, 224)  # Change image size to 224x224 (standard for ImageNet models)
batch_size = 32
data_dir = "../Chess"  # <- zmień jeśli folder ma inną nazwę/ścieżkę

# Wczytywanie danych
original_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

train_ds = original_train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=image_size,
    batch_size=batch_size
).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Klasy
class_names = original_train_ds.class_names
print("Klasy:", class_names)

# Augmentacja danych
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# Tworzenie modelu
base_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=image_size + (3,)
)
base_model.trainable = False  # Freeze the base model

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.6),  # Większy dropout
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

# Kompilacja
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
]

# Trenowanie
history = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=30,  # Możesz zwiększyć liczbę epok
    callbacks=callbacks
)

# Wykres accuracy/loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Dokładność treningu')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend()
plt.title('Funkcja straty')
plt.savefig(os.path.join(plots_dir, "1_accuracy_loss.png"))
plt.close()

# Macierz konfuzji
y_pred = []
y_true = []

for images, labels in val_ds:
    predictions = model.predict(images)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(labels.numpy())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.xlabel('Przewidywane klasy')
plt.ylabel('Prawdziwe klasy')
plt.title('Macierz konfuzji')
plt.savefig(os.path.join(plots_dir, "2_confusion_matrix.png"))
plt.close()

# Raport klasyfikacji
report = classification_report(y_true, y_pred, target_names=class_names)
print("Raport klasyfikacji:")
print(report)

# Zapisz raport klasyfikacji do pliku
with open(os.path.join(plots_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# Wizualizacja przewidywań
plt.figure(figsize=(15, 12))
sample_batch_images, sample_batch_labels = next(iter(val_ds.take(1)))
predictions = model.predict(sample_batch_images)

for i in range(min(15, len(sample_batch_images))):
    plt.subplot(3, 5, i + 1)
    plt.imshow(sample_batch_images[i].numpy().astype("uint8"))
    
    true_class = class_names[sample_batch_labels[i]]
    predicted_class = class_names[np.argmax(predictions[i])]
    
    title_color = "green" if true_class == predicted_class else "red"
    
    plt.title(f"Prawdziwa: {true_class}\nPrzew.: {predicted_class}", 
              color=title_color)
    plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "3_sample_predictions.png"))
plt.close()

# Top 10 niepoprawnych klasyfikacji
incorrect_indices = []

batch_index = 0
for images, labels in val_ds:
    predictions = model.predict(images)
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        pred_label = np.argmax(pred)
        if pred_label != label.numpy():
            confidence = pred[pred_label]
            incorrect_indices.append((batch_index, i, confidence))
    batch_index += 1

incorrect_indices.sort(key=lambda x: x[2], reverse=True)

plt.figure(figsize=(15, 10))
plot_count = 0

for batch_idx, img_idx, conf in incorrect_indices[:10]:
    for i, (images, labels) in enumerate(val_ds):
        if i == batch_idx:
            image = images[img_idx]
            true_label = labels[img_idx].numpy()
            prediction = model.predict(tf.expand_dims(image, 0))[0]
            pred_label = np.argmax(prediction)
            
            plot_count += 1
            plt.subplot(2, 5, plot_count)
            plt.imshow(image.numpy().astype("uint8"))
            plt.title(f"Prawdziwa: {class_names[true_label]}\n"
                      f"Przew.: {class_names[pred_label]}\n"
                      f"Pewność: {conf:.2f}", color="red")
            plt.axis("off")
            break
    
    if plot_count >= 10:
        break

plt.tight_layout()
plt.suptitle("Top 10 najbardziej pewnych błędnych klasyfikacji", fontsize=16)
plt.subplots_adjust(top=0.85)
plt.savefig(os.path.join(plots_dir, "4_top10_errors.png"))
plt.close()

print(f"Zapisano wszystkie wykresy do folderu '{plots_dir}'")