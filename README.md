# How-to-Make-Python-Machine-Learning-Models-Usable-Across-Any-Platform-

How to Make Python Machine Learning Models Usable Across Any Platform?
 

Python has become the go-to language for developing machine learning models, thanks to its powerful libraries such as Pandas and Scikit-Learn. While Python models integrate seamlessly into Python applications, using them in other languages like C++, Java, or C# requires extra effort. You cannot simply call a Python function from another language as if it were native.
 
So, how can we make a Python model accessible to applications written in any language and running on any platform? There are two main strategies:
Exposing the Model via a REST API: The model is wrapped inside a web service using a framework like Flask, allowing any application capable of sending HTTP(S) requests to interact with it. This approach is easy to deploy, especially when using Docker containers.

Exporting the Model to ONNX (Open Neural Network Exchange): ONNX is a platform-independent format that enables loading Python models in Java, C++, C#, and other environments using an ONNX runtime.
 Of course, if both the client application and the model are written in Python, these solutions are unnecessary.
In this article series, we will explore several practical scenarios:
Saving and calling a trained Python model from a Python client
Invoking a Python model from a non-Python client using a web service
Containerizing a Python model (and web service) for seamless deployment
Using ONNX to run a Python model in other programming languages
 In this first article, we will cover the first point, while the other articles will focus on the rest.

# 1-Consuming a Python Model from a Python Client 
At first glance, using a Python model from a Python client seems straightforward—simply call predict (or predict_proba for a classifier) on the trained model. However, you wouldn’t want to retrain the model every time you need to use it. Instead, the goal is to train it once and allow client applications to reload it in its pre-trained state whenever needed.
To achieve this, Python developers commonly rely on the pickle module.
For illustration, the following code trains a model using the well-known Iris dataset. Instead of immediately using the model for predictions, it saves the trained model to a .pkl file—this process, known as "pickling," is done using pickle.dump on the final line:

The stratify=y parameter in train_test_split ensures that the class distribution in the training and test sets is the same as in the original dataset. This is particularly important in classification problems to make sure that each class is represented proportionally in both sets. Without stratification, you could end up with imbalanced splits, especially when the dataset has uneven class distribution.
In the case of the Iris dataset, the target classes (species of flowers) are fairly balanced, but using stratify=y guarantees that the proportions of each class in the training and test sets will match the original distribution. This improves the model’s performance and ensures more reliable results during evaluation.



To use the model, a Python client utilizes pickle.load to load the serialized model from the .pkl file, effectively restoring it to its trained state. The client then calls predict_proba to estimate the probabilities of a passenger's survival.
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import pickle
import numpy as np
from sklearn.metrics import accuracy_score

## Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target


## Display the class distribution in the original dataset
print("Class distribution in the original dataset:")
print(Counter(y))


## Split the dataset into training and test sets, using stratify to ensure class distribution is similar in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


## Initialize and train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


## Save the trained model to a .pkl file
with open('iris_model.pkl', 'wb') as file:
   pickle.dump(model, file)


print("Model saved successfully!")
## Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)


# Évaluer la performance du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

					
Now the client can use the model to make a prediction without retraining it. And once the model is loaded, it can persist for the lifetime of the client and be called upon for predictions whenever needed. 
				
## Load the trained model from the .pkl file
with open('iris_model.pkl', 'rb') as file:
   model = pickle.load(file)


## Example input data (one sample)
sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])


## Use the model to predict the class index
predicted_class_index = model.predict(sample_data)


## Map the class index to the class name
class_names = ['setosa', 'versicolor', 'virginica']
predicted_class_name = class_names[predicted_class_index[0]]


## Use the model to predict probabilities
probabilities = model.predict_proba(sample_data)
print("Probabilities:")
print(probabilities)


## Display the predicted class name
print(f"Predicted class: {predicted_class_name}")


			
# Versioning Pickle Files
In general, a model saved (pickled) using one version of Scikit-learn may not be compatible with another version when attempting to unpickle it. This can lead to warning messages, or in some cases, the model may not load at all. To avoid this issue, it's essential to save and load models using the same version of Scikit-learn. From an engineering standpoint, this requires careful planning, as any updates to the Scikit-learn version in your applications will also necessitate updating the serialized models stored in your repository.


				
			
		

