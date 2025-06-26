import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# Ustawienia GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Folder na wykresy
plots_dir = "plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Parametry - rozpocznijmy od prostszego podejścia
image_size = (96, 96)  # Jeszcze mniejszy rozmiar
batch_size = 32
data_dir = "../Chess"

print("=== ANALIZA DANYCH ===")

# Wczytywanie danych
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Klasy:", class_names)
print(f"Liczba klas: {len(class_names)}")

# Sprawdź dokładnie rozkład danych
class_counts = {}
total_train = 0
for images, labels in train_ds.unbatch():
    total_train += 1
    label_name = class_names[labels.numpy()]
    class_counts[label_name] = class_counts.get(label_name, 0) + 1

print(f"\nRozkład danych treningowych (total: {total_train}):")
for class_name, count in class_counts.items():
    print(f"  {class_name}: {count} ({count/total_train*100:.1f}%)")

# MINIMALNA augmentacja - poprzednia była za mocna
simple_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.02),  # Bardzo małe rotacje
    layers.RandomBrightness(0.1),
])

# Bardzo proste przetwarzanie
def preprocess_train(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    # Tylko 30% szans na augmentację
    if tf.random.uniform([]) < 0.3:
        image = simple_augmentation(image)
    return image, label

def preprocess_val(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Przygotowanie danych
train_ds = train_ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# BARDZO PROSTY model - powrót do basics
def create_simple_chess_cnn(input_shape, num_classes):
    """
    Bardzo prosty CNN - testujemy czy dane w ogóle są użyteczne
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Pierwszy blok - bardzo podstawowy
        layers.Conv2D(16, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.1),
        
        # Drugi blok
        layers.Conv2D(32, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.15),
        
        # Trzeci blok
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.2),
        
        # Klasyfikator - bardzo prosty
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ], name='SimpleChessCNN')
    
    return model

# Jeszcze prostszy model - na wypadek gdyby powyższy nie działał
def create_minimal_chess_cnn(input_shape, num_classes):
    """
    Minimalny CNN - ostatnia deska ratunku
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(8, 7, activation='relu', padding='same'),
        layers.MaxPooling2D(4),
        
        layers.Conv2D(16, 5, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ], name='MinimalChessCNN')
    
    return model

# Rozpocznij od prostego modelu
print("=== TESTOWANIE PROSTEGO MODELU ===")
model = create_simple_chess_cnn(input_shape=image_size + (3,), num_classes=len(class_names))

# Kompilacja z wyższym learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),  # Wyższy LR
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
print(f"Parametry prostego modelu: {model.count_params():,}")

# Prosty callback
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=15,
        restore_best_weights=True,
        monitor='val_accuracy',
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        monitor='val_loss',
        verbose=1
    )
]

# Krótkie trenowanie - test
print("=== TRENOWANIE PROSTEGO MODELU ===")
history = model.fit(
    train_ds,
    epochs=30,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1
)

# Sprawdź wyniki
val_acc_simple = max(history.history['val_accuracy'])
print(f"\nProsty model - najlepsza dokładność: {val_acc_simple:.2%}")

# Jeśli prosty model też słabo, spróbuj minimalnego
if val_acc_simple < 0.4:
    print("\n=== PROSTY MODEL SŁABY - PRÓBUJE MINIMALNEGO ===")
    
    model_minimal = create_minimal_chess_cnn(input_shape=image_size + (3,), num_classes=len(class_names))
    model_minimal.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),  # Jeszcze wyższy LR  
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Parametry minimalnego modelu: {model_minimal.count_params():,}")
    
    history_minimal = model_minimal.fit(
        train_ds,
        epochs=20,
        validation_data=val_ds,
        verbose=1
    )
    
    val_acc_minimal = max(history_minimal.history['val_accuracy'])
    print(f"Minimalny model - najlepsza dokładność: {val_acc_minimal:.2%}")
    
    # Użyj lepszego modelu
    if val_acc_minimal > val_acc_simple:
        model = model_minimal
        history = history_minimal
        val_acc_best = val_acc_minimal
        print("Używam minimalnego modelu")
    else:
        val_acc_best = val_acc_simple
        print("Używam prostego modelu")
else:
    val_acc_best = val_acc_simple
    print("Prosty model działa dobrze")

# Wykresy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(15, 10))

# Accuracy
plt.subplot(2, 3, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', linewidth=2)
plt.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2)
plt.legend()
plt.title('Model Accuracy - Simple Approach')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

# Loss
plt.subplot(2, 3, 2)
plt.plot(epochs_range, loss, label='Training Loss', linewidth=2)
plt.plot(epochs_range, val_loss, label='Validation Loss', linewidth=2)
plt.legend()
plt.title('Model Loss')
plt.grid(True, alpha=0.3)

# Overfitting monitor
plt.subplot(2, 3, 3)
diff = np.array(acc) - np.array(val_acc)
plt.plot(epochs_range, diff, label='Overfitting Gap', color='red', linewidth=2)
plt.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Warning')
plt.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Critical')
plt.legend()
plt.title('Overfitting Monitor')
plt.grid(True, alpha=0.3)

# Class distribution
plt.subplot(2, 3, 4)
class_names_short = [name[:4] for name in class_names]
class_values = list(class_counts.values())
plt.bar(class_names_short, class_values, alpha=0.7)
plt.title('Training Data Distribution')
plt.ylabel('Number of samples')
plt.xticks(rotation=45)

# Learning rate
plt.subplot(2, 3, 5)
if 'lr' in history.history:
    plt.plot(epochs_range, history.history['lr'], 'g-', linewidth=2)
    plt.title('Learning Rate')
    plt.ylabel('LR')
    plt.yscale('log')
else:
    plt.text(0.5, 0.5, 'No LR history', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Learning Rate')

# Validation accuracy detail
plt.subplot(2, 3, 6)
plt.plot(epochs_range, val_acc, 'b-o', linewidth=2, markersize=4)
plt.title(f'Val Accuracy: {val_acc_best:.1%}')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "1_simple_model_results.png"), dpi=300, bbox_inches='tight')
plt.close()

# Ewaluacja
print("\n=== EWALUACJA ===")
y_pred = []
y_true = []

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(labels.numpy())

# Macierz konfuzji
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 8))

# Macierz konfuzji
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix\nAccuracy: {val_acc_best:.1%}')

# Per-class accuracy
plt.subplot(1, 2, 2)
class_accuracies = [cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0 for i in range(len(class_names))]
colors = ['red' if acc < 0.3 else 'orange' if acc < 0.6 else 'green' for acc in class_accuracies]
bars = plt.bar(class_names, class_accuracies, color=colors, alpha=0.7)
plt.title('Per-Class Accuracy')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim(0, 1)

for bar, acc in zip(bars, class_accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "2_simple_model_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()

# Raport
report = classification_report(y_true, y_pred, target_names=class_names)
print("\n=== RAPORT KLASYFIKACJI ===")
print(report)

print("\nWyniki per figura:")
for i, class_name in enumerate(class_names):
    accuracy = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
    support = cm[i].sum()
    print(f"  {class_name}: {accuracy:.1%} (support: {support})")

# Diagnostyka - sprawdź przykładowe obrazy
print("\n=== DIAGNOSTYKA DANYCH ===")
plt.figure(figsize=(15, 10))
sample_count = 0
for images, labels in train_ds.take(1):
    for i in range(min(12, len(images))):
        plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(f"{class_names[labels[i]]}")
        plt.axis("off")
        sample_count += 1

plt.suptitle('Sample Training Images - Check Data Quality', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "3_data_samples.png"), dpi=300, bbox_inches='tight')
plt.close()

# Zapisz raport
with open(os.path.join(plots_dir, "simple_model_report.txt"), "w") as f:
    f.write("=== SIMPLE CHESS MODEL REPORT ===\n")
    f.write(f"Final Validation Accuracy: {val_acc_best:.2%}\n")
    f.write(f"Model Parameters: {model.count_params():,}\n")
    f.write(f"Image Size: {image_size}\n\n")
    
    f.write("DATA DISTRIBUTION:\n")
    for class_name, count in class_counts.items():
        f.write(f"{class_name}: {count} samples ({count/total_train*100:.1f}%)\n")
    
    f.write(f"\nPER-CLASS ACCURACY:\n")
    for i, class_name in enumerate(class_names):
        accuracy = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        f.write(f"{class_name}: {accuracy:.1%}\n")
    
    f.write(f"\nCLASSIFICATION REPORT:\n{report}")

# Zapisz model
try:
    model.save("simple_chess_model.h5")
    print(f"\nModel zapisany jako simple_chess_model.h5")
except Exception as e:
    print(f"Błąd zapisu: {e}")

print("\n=== DIAGNOZA ===")
print(f"Finalna dokładność: {val_acc_best:.2%}")
print(f"Parametry modelu: {model.count_params():,}")

if val_acc_best < 0.3:
    print("\n❌ PROBLEM Z DANYMI LUB PODEJŚCIEM:")
    print("1. Sprawdź jakość danych - czy obrazy są czytelne?")
    print("2. Czy klasy są dobrze wybalansowane?")
    print("3. Czy augmentacja nie psuje danych?")
    print("4. Może problema jest w preprocessing?")
elif val_acc_best < 0.5:
    print("\n⚠️  SŁABE WYNIKI - MOŻLIWE PRZYCZYNY:")
    print("1. Za mało danych treningowych")
    print("2. Zbyt trudne rozróżnienie King/Queen") 
    print("3. Potrzeba lepszej augmentacji")
elif val_acc_best < 0.7:
    print("\n🔶 PRZECIĘTNE WYNIKI - MOŻNA POPRAWIĆ:")
    print("1. Transfer learning może pomóc")
    print("2. Większy model z więcej danych")
else:
    print("\n✅ DOBRE WYNIKI!")

print(f"\nSprawdź pliki w folderze: {plots_dir}")