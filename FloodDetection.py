import os
import random
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from IPython.display import Image as IPImage

# Class labels
labels = ['Flooding', 'No Flooding']

# Dataset paths
train_path = 'data/train'
valid_path = 'data/valid'
test_path = 'data/test'

# Data generators
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)\
    .flow_from_directory(directory=train_path, target_size=(224,224), batch_size=10)

valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)\
    .flow_from_directory(directory=valid_path, target_size=(224,224), batch_size=10)

test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)\
    .flow_from_directory(directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)

# Load base model
base_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False)

# Add custom layers
x = base_model.layers[-12].output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
output = Dense(units=2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Freeze early layers
for layer in model.layers[:-23]:
    layer.trainable = False

model.summary()

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    x=train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=valid_batches,
    validation_steps=len(valid_batches),
    epochs=10,
    verbose=2
)

# Save the model
model.save("fine_tuned_flood_detection_model.h5")

# Evaluate on test data
test_labels = test_batches.classes
predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

# Metrics
cm = confusion_matrix(y_true=test_labels, y_pred=predicted_classes)
precision = precision_score(test_labels, predicted_classes)
f1 = f1_score(test_labels, predicted_classes)
accuracy = accuracy_score(test_labels, predicted_classes)

print('Precision: ', precision)
print('F1 Score: ', f1)
print('Accuracy: ', accuracy)

# Plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
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
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

cm_plot_labels = ['Flooding','No Flooding']
plot_confusion_matrix(cm, classes=cm_plot_labels)

# Preprocess and predict on a new image
def preprocess_image(file):
    img_path = 'evaluate/'
    img = load_img(img_path + file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


# Display and evaluate a sample image
IPImage(filename='evaluate/1.jpg', width=300, height=200)

preprocessed_image = preprocess_image('1.jpg')
single_prediction = model.predict(preprocessed_image)
predicted_index = np.argmax(single_prediction)
confidence = single_prediction[0][predicted_index] * 100

print(f"Predicted: {labels[predicted_index]} ({confidence:.2f}%)")
