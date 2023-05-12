""" Copyright 2023 Lakera AI. All Rights Reserved.

Run Lakera's MLTest on a Roboflow dataset and model.
"""

import os
from typing import Iterable
import uuid
import yaml

import numpy as np
from PIL import Image

from lakera import (
    Batch,
    DataMapper,
    Predictor,
    Runner,
    RunnerOptions,
    Sample,
    TargetType,
)
from roboflow import Roboflow


class RoboflowDataset(DataMapper):
    def __init__(self, options: RunnerOptions):
        self.options = options
        self.path_to_dataset = "safety/test"
        with open("safety/data.yaml") as f:
            names = yaml.safe_load(f)["names"]
        self.id_to_label = {i: name for i, name in enumerate(names)}

    def get_input_identifiers(self) -> Iterable[str]:
        """Returns a list of all the image names in the dataset."""
        return os.listdir(os.path.join(self.path_to_dataset, "images"))

    def identifier_to_sample(self, identifier: str) -> Sample:
        """Given an image name, loads the image and its annotations."""
        # Load the image from disk.
        image = Image.open(
            os.path.join(self.path_to_dataset, "images", identifier)
        )

        # Create a Sample object and indicate we are working with object detection.
        sample = Sample(
            target_type=TargetType.BOUNDING_BOX,
            identifier=identifier,
        )
        # Add the image to the sample.
        sample.add_input(np.array(image))
        # Load the labels from disk.
        labels_file = identifier.replace(".jpg", ".txt")
        predictions = []
        with open(
            os.path.join(self.path_to_dataset, "labels", labels_file)
        ) as f:
            predictions = f.readlines()

        for p in predictions:
            class_id, x, y, w, h = p.split()
            label = self.id_to_label[int(class_id)]
            # Add the bounding box to the sample.
            sample.add_yolo_target_bbox(
                float(x), float(y), float(w), float(h), label
           )
        return sample


class RoboflowPredictorAPI(Predictor):
    def __init__(self, options: RunnerOptions):
        self.options = options
        # Here, specify your own Roboflow API key.
        self.rf = Roboflow(api_key="your_api_key")
        project = self.rf.workspace().project("site-safety-lk")
        self.model = project.version(1).model

    def predict(self, batch: Batch) -> Batch:
        """Makes predictions on an input batch of images."""
        for sample in batch:
            h, w, _ = sample.input.shape
            # Save the image to disk to be uploaded to Roboflow.
            path_to_image = f"/tmp/rf/{uuid.uuid4()}.jpg"
            Image.fromarray(sample.input).save(path_to_image)

            # Compute predictions by making a request to Roboflow.
            pred = self.model.predict(
                path_to_image, confidence=40, overlap=30
            ).json()
            if "predictions" not in pred:
                continue
            # Add the model predictions to the sample.
            for p in pred["predictions"]:
                sample.add_yolo_prediction_bbox(
                    p["x"] / float(w),
                    p["y"] / float(h),
                    p["width"] / float(w),
                    p["height"] / float(h),
                    p["class"],
                    p["confidence"],
                )
        return batch


if __name__ == "__main__":
    path_to_options = "./options.yaml"
    # Where to store the MLTest results.
    path_to_output = "./mltest_results"

    # Load your test suite.
    options = RunnerOptions(
        path_to_yaml=path_to_options,
    )
    # Create a Lakera Runner
    runner = Runner(
        RoboflowDataset, RoboflowPredictorAPI, options, path_to_output
    )
    # Run the runner to get your results!
    runner_results = runner.run()
    runner_results.print_summary()
