# Quick Start Tutorial
This notebook provides a quick overview of developing and deploying machine learning applications to production using MLRun MLOps orchestration framework.

In order to run the demo enter the proper values to the env file according to the [remote setup guide](https://docs.mlrun.org/en/latest/install/remote.html). 
Once completed you can run the script on your IDE.

## Define MLRun project and ML functions
In the first section of the script, we will define a new MLRun project along with the function we'll be using.
```python
import mlrun

mlrun.set_env_from_file(env_file="remote.env")

project = mlrun.new_project("breast-cancer", "./", user_project=True, init_git=True)

project.set_function("gen_breast_cancer.py", "gen-breast-cancer", image="mlrun/mlrun").apply(mlrun.auto_mount())
project.set_function("trainer.py", "trainer", handler="train", image="mlrun/mlrun").apply(mlrun.auto_mount())
project.set_function("serving.py", "serving", image="mlrun/mlrun", kind="serving").apply(mlrun.auto_mount())
project.save()
```
Eventually we export project as a YAML file which can be used CI\CD and automation pipelines.

## Run data processing function and log artifacts
Once we have the functions defined within the project, we can run them using `mlrun.run_function`:
```python
gen_data_run = project.run_function("gen-breast-cancer", params={"format": "csv"}, local=False)
```
The outputs and artifacts are stored and tracked in MLRun DB:
```python
print(gen_data_run.state())

print(gen_data_run.outputs)
```

In case the script is run remotely, we can get the artifact using `v3io://` prefix: 
```python
df_url = gen_data_run.artifact("dataset")
df = df_url.url.replace("/v3io", "v3io://")
print(mlrun.get_dataitem(df).as_df().head())
```
## Use MLRun built-in marketplace functions (data analysis)
You can import a pre-made function from the marketplace, in this case we are using `describe`:
```python
describe = mlrun.import_function('hub://describe')
```
Now we can run it:
```python
describe.apply(mlrun.auto_mount())
describe_run = describe.run(params={'label_column': 'label'},
                            inputs={"table": gen_data_run.outputs['dataset']}, local=False)
```
And see the outputs of the step:
```python
print(describe_run.outputs)
```

## Train, track, and register models
In the trainer function we are using `apply_mlrun` function. It allows us to provide model object along with various parameters. Metrics and charts will be automatically logged and registered.
```python
trainer_run = project.run_function(
    "trainer",
    inputs={"dataset": gen_data_run.outputs['dataset']},
    params = {"n_estimators": 100, "learning_rate": 1e-1, "max_depth": 3},
    local=False
)
```
```python
print(trainer_run.outputs)
```

## Hyper-parameter tuning and model/experiment comparison
Run a `GridSearch` with a couple of parameters, and select the best run with respect to the max accuracy:
```python
hp_tuning_run = project.run_function(
    "trainer",
    inputs={"dataset": gen_data_run.outputs['dataset']},
    hyperparams={
        "n_estimators": [10, 100, 1000],
        "learning_rate": [1e-1, 1e-3],
        "max_depth": [2, 8]
    },
    selector="max.accuracy",
    local=False
)
```
```python
print(hp_tuning_run.outputs)
```
List the models in the project:
```python
models = project.list_models()
for model in models:
    print(f"uri: {model.uri}, metrics: {model.metrics}")
```

## Build, test and deploy Model serving functions
Once we have the model, we can easily deploy it using Nuclio real-time serverless engine:
```python
serving_fn = mlrun.code_to_function("serving", filename="serving.py", image="mlrun/mlrun", kind="serving")
serving_fn.apply(mlrun.auto_mount())
serving_fn.add_model('cancer-classifier',model_path=hp_tuning_run.outputs["model"], class_name='ClassifierModel')
mlrun.deploy_function(serving_fn)
```

And test it:
```python
my_data = {
    "inputs":[
        [
            1.371e+01, 2.083e+01, 9.020e+01, 5.779e+02, 1.189e-01, 1.645e-01,
            9.366e-02, 5.985e-02, 2.196e-01, 7.451e-02, 5.835e-01, 1.377e+00,
            3.856e+00, 5.096e+01, 8.805e-03, 3.029e-02, 2.488e-02, 1.448e-02,
            1.486e-02, 5.412e-03, 1.706e+01, 2.814e+01, 1.106e+02, 8.970e+02,
            1.654e-01, 3.682e-01, 2.678e-01, 1.556e-01, 3.196e-01, 1.151e-01
        ],
        [
            1.308e+01, 1.571e+01, 8.563e+01, 5.200e+02, 1.075e-01, 1.270e-01,
            4.568e-02, 3.110e-02, 1.967e-01, 6.811e-02, 1.852e-01, 7.477e-01,
            1.383e+00, 1.467e+01, 4.097e-03, 1.898e-02, 1.698e-02, 6.490e-03,
            1.678e-02, 2.425e-03, 1.450e+01, 2.049e+01, 9.609e+01, 6.305e+02,
            1.312e-01, 2.776e-01, 1.890e-01, 7.283e-02, 3.184e-01, 8.183e-02]
    ]
}
ret = serving_fn.invoke("/v2/models/cancer-classifier/infer", body=my_data)
print(ret)
```

## Build and run automated ML pipelines and CI/CD
You can easily compose a workflow from your functions that automatically prepares data, trains, tests, and deploys the model - every time you change the code or data, or need a refresh:
```python
run_id = project.run(workflow_path="./workflow.py",
                     arguments={"model_name": "breast_cancer_classifier"},
                     watch=True,
                     local=False)
```