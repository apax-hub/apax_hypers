# Apax Hypers

Very rudimentary script to run hyperparameter tuning of apax models with optuna.
You need to do the following:

1. adapt the `train_template.yaml` to specify your data and study paths. Things like batch size / number of epochs / ensembles are not optimized. Specify those as well.
2. adapt `hypers.py` to adjust the search space to your needs. The current setup optmizes almost all parameters in apax over a relatively wide search space. If you would like to, for example, only optimize GMNN models, you have to remove the others models from the options in the python

Once you have everything set up, you can just run the script.
It is possible to run this in parallel by just launching it multiple times.

It is recommended to use a large number of epochs with a lenient patience.
This will ensure that models are fully trained without wasting times on diverging runs.
Further, there is a jupyter notebook for very basic tracking and analysis of the study.

If the need arises, I will turn this into more of a proper package with a more convenient interface, but this is currently not planned.