import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing.data_generator import SequenceDataGenerator
import config.settings as settings


def evaluate_model(model_path, test_data_dir):
    """
    Evaluate the trained model on test data
    """
    # Load model
    model = tf.keras.models.load_model(model_path)

    # Create test data generator
    test_generator = SequenceDataGenerator(
        test_data_dir,
        sequence_length=settings.SEQUENCE_LENGTH,
        batch_size=settings.BATCH_SIZE,
        img_size=(settings.IMG_HEIGHT, settings.IMG_WIDTH),
        shuffle=False,
    )

    # Predict
    y_true = []
    y_pred = []

    for i in range(len(test_generator)):
        X_batch, y_batch = test_generator[i]
        y_true.extend(y_batch)

        predictions = model.predict(X_batch, verbose=0)
        y_pred.extend(predictions.flatten())

    # Convert to binary predictions
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)

    # Print classification report
    print("Classification Report:")
    print(
        classification_report(y_true, y_pred_binary, target_names=["Normal", "Anomaly"])
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("confusion_matrix.png")
    plt.show()

    # ROC AUC Score
    auc_score = roc_auc_score(y_true, y_pred)
    print(f"ROC AUC Score: {auc_score:.4f}")


if __name__ == "__main__":
    evaluate_model(
        "models/trained/airport_anomaly_detector_final.h5", "data/processed/frames/test"
    )
