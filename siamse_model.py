import numpy as np
from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Dropout, BatchNormalization
from keras import backend as K
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

def siamese_network(input_shape):
    input_layer = Input(shape=input_shape)

    # Convolutional layers for each channel
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer[:, :, :, 0:1])
    conv2 = Conv2D(32, (3, 3), activation='relu')(input_layer[:, :, :, 1:2])


    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)

    # Concatenate the outputs of the convolutional layers
    merged_channels = Concatenate()([conv1, conv2])
    
    
    # Convolutional layers
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    
    # Dense layers
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    flattened = Flatten()(merged_channels)
    output_layer = Dense(1, activation='sigmoid')(flattened)

    return Model(inputs=input_layer, outputs=output_layer)    

def contrastive_loss(y_true, y_pred):
    margin = 1
    y_true = K.cast(y_true, dtype=K.floatx())
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def learning_rate_schedule(epoch):
    initial_lr = 0.001
    if epoch < 10:
        return initial_lr
    elif epoch < 20:
        return initial_lr * 0.1
    else:
        return initial_lr * 0.01

def create_pairs(x, digit_indices):
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    
    for d in range(10):
        for i in range(n):
            # positive pair
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            
            # Ensure indices are within bounds
            z1 = z1 % len(x)
            z2 = z2 % len(x)
            
            pairs.append([x[z1], x[z2]])
            labels.append(1)
            
            # negative pair
            inc = np.random.randint(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            
            # Ensure indices are within bounds
            z1 = z1 % len(x)
            z2 = z2 % len(x)
            
            pairs.append([x[z1], x[z2]])
            labels.append(0)
    
    return np.array(pairs), np.array(labels)


# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reshape the data to add a channel dimension (for Conv2D)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Create digit indices for pairing
digit_indices = [np.where(y_train == i)[0] for i in range(10)]

# Create pairs for training and testing
train_pairs, train_labels = create_pairs(x_train, digit_indices)
test_pairs, test_labels = create_pairs(x_test, digit_indices)

# Set the input shape for MNIST images
input_shape = (28, 28, 2)

# Create the Siamese model
model = siamese_network(input_shape)
# model.compile(loss=contrastive_loss, optimizer='adam')

# Compile the model with the contrastive loss function and Adam optimizer
optimizer = Adam(learning_rate=0.001)
model.compile(loss=contrastive_loss, optimizer=optimizer)

# Set up callbacks for learning rate scheduling and model checkpointing
lr_scheduler = LearningRateScheduler(learning_rate_schedule)
# Define a ModelCheckpoint callback to save the model
model_checkpoint = ModelCheckpoint("/Users/btelisma/Documents/Work_Org/Appium-Projects/best_model_checkpoint.keras", 
                                   save_best_only=True,  # Save only the best model
                                   monitor='val_loss',   # Monitor validation loss
                                   mode='min',           # Save the model when validation loss decreases
                                   verbose=1)            # Show progress


# # Concatenate pairs along a new axis
# train_pairs_concatenated = np.concatenate([train_pairs[:, 0, :, :, np.newaxis], train_pairs[:, 1, :, :, np.newaxis]], axis=-1)
# test_pairs_concatenated = np.concatenate([test_pairs[:, 0, :, :, np.newaxis], test_pairs[:, 1, :, :, np.newaxis]], axis=-1)
# Assuming train_pairs[:, 0] and train_pairs[:, 1] have shape (num_pairs, 28, 28, 1)
test_pairs_concatenated = np.concatenate([test_pairs[:, 0], test_pairs[:, 1]], axis=-1)
train_pairs_concatenated = np.concatenate([train_pairs[:, 0], train_pairs[:, 1]], axis=-1)



# Train the Siamese model with ModelCheckpoint
history = model.fit(train_pairs_concatenated, train_labels,
                    validation_data=(test_pairs_concatenated, test_labels),
                    epochs=30, batch_size=128, 
                    callbacks=[lr_scheduler, model_checkpoint])


