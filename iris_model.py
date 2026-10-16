# This file trains a simple KNN model on the Iris dataset

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data      # flower measurements
y = iris.target    # flower types (0, 1, 2)

# Create the KNN model (look at the 3 nearest neighbors)
model = KNeighborsClassifier(n_neighbors=3)

# Train the model
model.fit(X, y)

# Names of the three flowers
flower_names = ["Setosa", "Versicolor", "Virginica"]


def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
    """Takes 4 measurements and returns the flower name."""
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return flower_names[prediction[0]]
