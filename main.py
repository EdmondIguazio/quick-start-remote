import mlrun

mlrun.set_env_from_file(env_file="remote.env")

project = mlrun.new_project("breast-cancer", "./", user_project=True, init_git=True)

project.set_function("gen_breast_cancer.py", "gen-breast-cancer", image="mlrun/mlrun").apply(mlrun.auto_mount())
project.set_function("trainer.py", "trainer", handler="train", image="mlrun/mlrun").apply(mlrun.auto_mount())
project.set_function("serving.py", "serving", image="mlrun/mlrun", kind="serving").apply(mlrun.auto_mount())
project.save()

gen_data_run = project.run_function("gen-breast-cancer", params={"format": "csv"}, local=False)

print(gen_data_run.state())

print(gen_data_run.outputs)

# Generate V3IO URL to retrieve the dataset remotely
df_url = gen_data_run.artifact("dataset")
df = df_url.url.replace("/v3io", "v3io://")
print(mlrun.get_dataitem(df).as_df().head())

describe = mlrun.import_function('hub://describe')
describe.apply(mlrun.auto_mount())
describe_run = describe.run(params={'label_column': 'label'},
                            inputs={"table": gen_data_run.outputs['dataset']}, local=False)
print(describe_run.outputs)

trainer_run = project.run_function(
    "trainer",
    inputs={"dataset": gen_data_run.outputs['dataset']},
    params = {"n_estimators": 100, "learning_rate": 1e-1, "max_depth": 3},
    local=False
)

print(trainer_run.outputs)

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

print(hp_tuning_run.outputs)

# list the models in the project (can apply filters)
models = project.list_models()
for model in models:
    print(f"uri: {model.uri}, metrics: {model.metrics}")


serving_fn = mlrun.code_to_function("serving", filename="serving.py", image="mlrun/mlrun", kind="serving")
serving_fn.apply(mlrun.auto_mount())
serving_fn.add_model('cancer-classifier',model_path=hp_tuning_run.outputs["model"], class_name='ClassifierModel')
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

mlrun.deploy_function(serving_fn)
ret = serving_fn.invoke("/v2/models/cancer-classifier/infer", body=my_data)
print(ret)

run_id = project.run(workflow_path="./workflow.py",
                     arguments={"model_name": "breast_cancer_classifier"},
                     watch=True,
                     local=False)