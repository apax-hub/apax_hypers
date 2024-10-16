from pathlib import Path
import uuid
import numpy as np
import pandas as pd
import optuna
import yaml
from optuna.trial import Trial

from apax.train.run import run as apax_run

study_name = "study"


models = ["gmnn", "so3krates", "equiv-mp"]
opts = ["sgd", "adam", "adamw", "ademamix", "lamb", "sam"]
schedules = ["linear", "cyclic_cosine"]
bases = ["bessel", "gaussian"]
repulsions = ["None", "exponential", "zbl"]


def get_suggestions(trial: Trial):
    params = {
        "model": {"basis":{}},
        "empirical_corrections": [],
        "optimizer": {},
    }
    # MODEL
    model = trial.suggest_categorical("model", models)
    params["model"]["name"] = model
    if model == "gmnn":
        params["model"]["n_contr"] = trial.suggest_int("n_contr",1,8,)
        params["model"]["n_radial"] = trial.suggest_int("n_radial",3,8,)
    elif model == "equiv-mp":
        params["model"]["features"] = trial.suggest_int("features",4,32,)
        params["model"]["max_degree"] = trial.suggest_int("max_degree",1,3,)
        params["model"]["num_iterations"] = trial.suggest_int("num_iterations",1,3,)
    elif model == "so3krates":
        params["model"]["num_layers"] = trial.suggest_int("num_layers", 1, 3)
        params["model"]["max_degree"] = trial.suggest_int("max_degree", 1,3)
        params["model"]["num_features"] = trial.suggest_int("num_features", 8, 256)
        params["model"]["num_heads"] = trial.suggest_int("num_heads", 1,8)
        params["model"]["use_layer_norm_1"] = trial.suggest_categorical("use_layer_norm_1", [True, False])
        params["model"]["use_layer_norm_2"] = trial.suggest_categorical("use_layer_norm_2", [True, False])
        params["model"]["use_layer_norm_final"] = trial.suggest_categorical("use_layer_norm_final", [True, False])
        params["model"]["transform_input_features"] = trial.suggest_categorical("transform_input_features", [True, False])
    

    ## BASIS
    basis = trial.suggest_categorical("basis", bases)
    n_basis = trial.suggest_int("n_basis", 4, 32)
    r_max = trial.suggest_float("r_max", 3.0, 7.0)
    params["model"]["basis"]["name"] = basis
    params["model"]["basis"]["n_basis"] = n_basis
    params["model"]["basis"]["r_max"] = r_max
    if basis == "gaussian":
        r_min = trial.suggest_float("r_min", 0.5, 1.0)
        params["model"]["basis"]["r_min"] = r_min


    ## NN
    n_layers = trial.suggest_int("n_layers", 1, 4)

    layers = []
    for i in range(n_layers):
        n_units = trial.suggest_int("units{}".format(i), 8, 512, log=True)
        layers.append(n_units)
    params["model"]["nn"] = layers


    ## REPULSION
    repulsion = trial.suggest_categorical("repulsion", repulsions)
    if repulsion != "None":
        rep_r_max = trial.suggest_float("rep_r_max", 0.5, 2.5)
        params["model"]["empirical_corrections"] = [{"name": repulsion, "r_max": rep_r_max}]

    # OPT
    optimizer = trial.suggest_categorical("optimizer", opts)

    emb_lr = trial.suggest_float("emb_lr", 1e-5, 1.0, log=True)
    nn_lr = trial.suggest_float("nn_lr", 1e-5, 1.0, log=True)
    scale_lr = trial.suggest_float("scale_lr", 1e-5, 1.0, log=True)
    shift_lr = trial.suggest_float("shift_lr", 1e-5, 1.0, log=True)

    if repulsion == "exponential":
        rep_scale_lr = trial.suggest_float("rep_scale_lr", 1e-5, 1.0, log=True)
        rep_prefactor_lr = trial.suggest_float("rep_prefactor_lr", 1e-5, 1.0, log=True)
    else: 
        rep_scale_lr = 0
        rep_prefactor_lr = 0

    if repulsion == "exponential":
        zbl_lr = trial.suggest_float("zbl_lr", 1e-5, 1.0, log=True)
    else: 
        zbl_lr = 0

    gradient_clipping = trial.suggest_float("gradient_clipping", 1.0, 15)

    optParams = {
        "name": optimizer,
        "emb_lr": emb_lr,
        "nn_lr": nn_lr,
        "scale_lr": scale_lr,
        "shift_lr": shift_lr,
        "rep_scale_lr": rep_scale_lr,
        "rep_prefactor_lr": rep_prefactor_lr,
        "zbl_lr": zbl_lr,
        "gradient_clipping": gradient_clipping,
        "kwargs": {},
        "schedule": {}
    }

    if optimizer in ["adamw", "ademamix"]:
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        optParams["kwargs"]["weight_decay"] = weight_decay

    if optimizer == "ademamix":
        alpha = trial.suggest_int("alpha", 1, 20)
        optParams["kwargs"]["alpha"] = alpha
    if optimizer == "sam":
        sync_period = trial.suggest_int("sync_period", 1, 20)
        optParams["kwargs"]["sync_period"] = sync_period

    params["optimizer"].update(optParams)

    ## SCHEDULE
    schedule = trial.suggest_categorical("schedule", schedules)
    params["optimizer"]["schedule"]["name"] = schedule
    if schedule == "cyclic_cosine":
        period = trial.suggest_int("period", 1,200)
        decay_factor = trial.suggest_float("decay_factor", 0.5, 1.0)
        params["optimizer"]["schedule"]["period"] = period
        params["optimizer"]["schedule"]["decay_factor"] = decay_factor

    return params


def load_and_update_config(path, new_params, name):
    with path.open("r") as f:
        template_params = yaml.safe_load(f)
    template_params["model"].update(new_params["model"])
    template_params["optimizer"].update(new_params["optimizer"])
    template_params["data"]["experiment"] = name
    return template_params


def run_and_eval(parameters):
    directory = parameters["data"]["directory"]
    exp = parameters["data"]["experiment"]
    model_dir = Path(directory) / exp
    # try:
    apax_run(parameters, "info")
    metrics_df = pd.read_csv(model_dir / "log.csv")
    best_epoch = np.argmin(metrics_df["val_loss"])
    metrics = metrics_df.iloc[best_epoch].to_dict()
    loss = metrics["val_loss"]
    # except:
    #     loss = np.inf
    return loss


def objective_fn(trial):
    name = uuid.uuid4()
    name = str(name)
    path = Path("train_template.yaml")
    sugg_params = get_suggestions(trial)
    params = load_and_update_config(path, sugg_params, name)
    loss = run_and_eval(params)
    return loss



storage_name = "sqlite:///{}.db".format(study_name)

if not Path(f"{study_name}.db").is_file():
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize")
else:
    study = optuna.load_study(study_name=study_name, storage=storage_name)

study.optimize(objective_fn, n_trials=1000)
