# Lakera integration with Roboflow

This repository shows you how to run Lakera's MLTest with models running behind Roboflow's hosted API. It provides the code used to run the experiments in [this blog post](). You can explore the results in more depth at [roboflow.lakera.ai](https://roboflow.lakera.ai).

Learn more about Lakera on our [website](https://lakera.ai).

## Core concepts

This repository contains two files: 
- `lakera_integration.py` contains the code required to run MLTest on a Roboflow model. It implements a `RoboflowDataset`, which reads images and labels from a Roboflow dataset, and a `RoboflowPredictorAPI`, a wrapper around Roboflow's hosted API. You can adapt these to fit an arbitrary setup, learn more in [MLTest's API reference](https://docs.lakera.ai/configuration/configuration-basics).
- `options.yaml` configures the tests that will be run by MLTest. You can have an in-depth look at possible configurations in our [documentation](https://docs.lakera.ai/configuration/configuration-basics).


## Running MLTest on your Roboflow model

To get started, simply go to `lakera_integrations.py` and provide your `ROBOFLOW_API_KEY`, `ROBOFLOW_PROJECT` and `ROBOFLOW_MODEL_VERSION`. These will be used in the `RoboflowPredictorAPI` to initialize your model: 

```python
self.rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = self.rf.workspace().project(ROBOFLOW_PROJECT)
self.model = project.version(ROBOFLOW_MODEL_VERSION).model
```

You also need to download your target dataset from the Roboflow platform. You can then specify `PATH_TO_DATASET` to indicate the path to the downloaded dataset.

Once that's done, run the following to get insights into your model: 

```bash
python lakera_integration.py
```

You can then run the Dashboard as follows to explore the results: 

```bash
docker run --rm -it -p 5000:5000 \
  -v $(pwd)/mltest_results:/home/results \
  registry.gitlab.com/lakeraai/lakera/dashboard:latest
```

## Where can I go to next? 

The test configuration in this example focuses on robustness, but you can go much deeper with MLTest, from model failure clustering to automatic labelling of the images that matter most to you. You can learn more at [our documentation](https://docs.lakera.ai).

Here are a few examples of what you can do with MLTest: 
- [Deep-dive into your model's failures with Voxel51](https://youtu.be/lEssnBNRtEE).
- [Add auto-tags to your dataset to super-power failure discovery](https://youtu.be/r8uTG7dhWQI).
- [Export all MLTest insights to your favourite experiment tracking tool](https://youtu.be/DKm7Z3muDbs).


[Get early access](https://r8m00ml29mf.typeform.com/to/R64QfhfS) to get started.






