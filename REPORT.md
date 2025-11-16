— how I ran and checked the app

1) Activate venv and install deps:

```bash
source venv-py311/bin/activate
python -m pip install -r requirements.txt
```

2) Train :

```bash
python src/train.py
```

3) Test :

```bash
python src/predict.py
```

Example output I got:

```
Iris Classifier Prediction
Model loaded successfully!
Target names: ['setosa', 'versicolor', 'virginica']

 Example Predictions:
Features: [sepal length, sepal width, petal length, petal width]

Example 1: [5.1, 3.5, 1.4, 0.2]
Prediction: setosa
Probabilities:
	setosa: 0.9784
	versicolor: 0.0216
	virginica: 0.0000

Example 2: [6.7, 3.0, 5.2, 2.3]
Prediction: virginica
Probabilities:
	setosa: 0.0001
	versicolor: 0.0923
	virginica: 0.9076

Example 3: [5.9, 3.0, 4.2, 1.5]
Prediction: versicolor
Probabilities:
	setosa: 0.0183
	versicolor: 0.8789
	virginica: 0.1028
```