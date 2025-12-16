import tensorflow as tf
from keras.applications import VGG16, ResNet50, InceptionV3
from keras import layers, models, Model
from keras.optimizers import Adam
import config.settings as settings


class CNNLSTMModel:
    def __init__(
        self,
        sequence_length=settings.SEQUENCE_LENGTH,
        img_height=settings.IMG_HEIGHT,
        img_width=settings.IMG_WIDTH,
    ):
        self.sequence_length = sequence_length
        self.img_height = img_height
        self.img_width = img_width
        self.model = None

    def create_feature_extractor(self, backbone="vgg16"):
        """
        Create CNN feature extractor using pre-trained model
        """
        if backbone == "vgg16":
            base_model = VGG16(
                weights="imagenet",
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3),
            )
        elif backbone == "resnet50":
            base_model = ResNet50(
                weights="imagenet",
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3),
            )
        elif backbone == "inception_v3":
            base_model = InceptionV3(
                weights="imagenet",
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3),
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze base model layers initially
        base_model.trainable = False

        # Add custom layers on top
        inputs = layers.Input(shape=(self.img_height, self.img_width, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(settings.DROPOUT_RATE)(x)

        feature_extractor = Model(inputs, x, name="feature_extractor")
        return feature_extractor

    def build_model(self, backbone="vgg16"):
        """
        Build the complete CNN-LSTM model
        """
        # Create feature extractor
        feature_extractor = self.create_feature_extractor(backbone)

        # Build sequence model
        sequence_input = layers.Input(
            shape=(self.sequence_length, self.img_height, self.img_width, 3),
            name="sequence_input",
        )

        # Apply feature extractor to each frame in the sequence
        x = layers.TimeDistributed(feature_extractor)(sequence_input)

        # LSTM layers for temporal modeling
        x = layers.LSTM(settings.LSTM_UNITS, return_sequences=True)(x)
        x = layers.Dropout(settings.DROPOUT_RATE)(x)
        x = layers.LSTM(settings.LSTM_UNITS // 2, return_sequences=False)(x)
        x = layers.Dropout(settings.DROPOUT_RATE)(x)

        # Classification head
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(settings.DROPOUT_RATE)(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(1, activation="sigmoid", name="classification")(x)

        # Create model
        self.model = Model(sequence_input, outputs)

        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=settings.LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"],
        )

        return self.model

    def unfreeze_backbone(self, unfreeze_layers=5):
        """
        Unfreeze some layers of the backbone for fine-tuning
        """
        if self.model is None:
            raise ValueError("Model must be built before unfreezing")

        # Get the feature extractor
        feature_extractor = self.model.get_layer("time_distributed").layer

        # Unfreeze last few layers
        for layer in feature_extractor.layers[-unfreeze_layers:]:
            layer.trainable = True

        # Recompile model
        self.model.compile(
            optimizer=Adam(learning_rate=settings.LEARNING_RATE / 10),
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"],
        )

    def summary(self):
        """Print model summary"""
        if self.model:
            return self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")


# Convenience function to create model
def create_cnn_lstm_model():
    model_builder = CNNLSTMModel()
    return model_builder.build_model(settings.CNN_BACKBONE)


if __name__ == "__main__":
    # Test model creation
    model = create_cnn_lstm_model()
    model.summary()
