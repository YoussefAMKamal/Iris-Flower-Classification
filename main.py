#%%
import pandas
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

dataset = pandas.read_csv("IRIS.csv")

images = dataset.drop(columns=['species'])
labels = dataset['species']

training_image, test_image, training_label, test_label = train_test_split(images, labels, test_size=0.4)
svm = SVC()
svm.fit(training_image, training_label)

predictions = svm.predict(test_image)

print("Model Accuracy = {}%".format(accuracy_score(test_label, predictions)*100))

# %%
