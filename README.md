# ML-based meal recommendation

This project was originally built to solve the practical problem of planning a set of compatible meals.

A regression model recommends a set of three dishes which are dissimilar from each other in terms of taste, price, and other factors.

<img width="773" height="467" alt="image" src="https://github.com/user-attachments/assets/49ff65f3-e905-4828-b5bb-8e906e41a6d1" />

*Screenshot of the recommendation app*

## Project structure

- `notebooks/`: building a training dataset, eda, and model development
- `src/`: prediction pipeline + webapps
- `data/`: omitted here for privacy

Recommended order of reading: `training_data.ipynb` -> `eda.ipynb` -> `model.ipynb` -> `src/`

## How to run locally

- Install packages in `requirements.txt`
- Create an xlsx file `data/Ruokasuositusalgoritmi - data.xlsx` with dish data in the format specified in `training_data.ipynb`
- Run `training_data.ipynb`
- Create training data by scoring dishes using `recommendation_app.py`
- Run `train_and_predict.py`
- Run `recommendation_app.py`
