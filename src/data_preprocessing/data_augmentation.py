import albumentations as A
import cv2
import numpy as np
from pathlib import Path
import os


class DataAugmentor:
    def __init__(self):
        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomGamma(p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.RandomRotate90(p=0.3),
                A.HueSaturationValue(p=0.3),
            ]
        )

    def augment_image(self, image):
        """
        Apply augmentation to a single image
        """
        augmented = self.transform(image=image)
        return augmented["image"]

    def augment_directory(self, input_dir, output_dir, num_augmentations=3):
        """
        Augment all images in a directory
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))

        print(f"Augmenting {len(image_files)} images from {input_dir}")

        for img_file in image_files:
            # Read original image
            image = cv2.imread(str(img_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Save original
            original_output = output_path / f"original_{img_file.name}"
            cv2.imwrite(str(original_output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Create augmented versions
            for i in range(num_augmentations):
                augmented_image = self.augment_image(image)
                aug_output = output_path / f"aug_{i}_{img_file.name}"
                cv2.imwrite(
                    str(aug_output), cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                )
