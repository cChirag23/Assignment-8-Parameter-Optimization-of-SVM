#Chirag Singla - 102103303 - COE11
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')
my_data_set = pd.read_csv('C:\\Users\\Asus\\Desktop\\dry_bean_dataset.csv')
my_data_set
# Encode target variable
label_encoder = LabelEncoder()
class_labels = ["SEKER", "BARBUNYA", "BOMBAY","CALI","HOROZ","SIRA","DERMASON"]
label_encoder.fit(class_labels)
my_data_set["Class"] = label_encoder.transform(my_data_set["Class"])

def preprocess_data(dataset):
    X = dataset.iloc[:, :-1]
    y = dataset['Class']
    ss = StandardScaler()
    X = ss.fit_transform(X)
    return X, y

def tune_svm(X_train, y_train):
    param_grid = {
        'C': np.logspace(-3, 3, 7),
        'gamma': np.logspace(-3, 3, 7),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
    grid_search = GridSearchCV(SVC(max_iter=100), param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_score_

def plot_learning_curve(X_train, y_train, best_params):
    svm = SVC(**best_params, max_iter=100)
    train_sizes, train_scores, test_scores = learning_curve(svm, X_train, y_train, cv=10, scoring='accuracy',
                                                            train_sizes=np.linspace(0.01, 1.0, 50))
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Score on Training')
    plt.plot(train_sizes, test_mean, label='Score on Cross-Validation')
    plt.xlabel('Training Instances')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend(loc="best")
    plt.show()

def main():
    X, y = preprocess_data(my_data_set)
    samples = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        samples.append((X_train, X_test, y_train, y_test))

    result = pd.DataFrame(columns=['Sample', 'Best Accuracy', 'Best Kernel Value', 'Best C Value', 'Best Gamma Value'])
    for i, (X_train, X_test, y_train, y_test) in enumerate(samples, start=1):
        best_params, best_score = tune_svm(X_train, y_train)
        result.loc[i - 1] = [i, round(best_score, 2), best_params['kernel'], round(best_params['C'], 2),
                             round(best_params['gamma'], 2)]

    print(result)

    best_params = {'kernel': result['Best Kernel Value'].iloc[result['Best Accuracy'].idxmax()],
                   'C': result['Best C Value'].iloc[result['Best Accuracy'].idxmax()],
                   'gamma': result['Best Gamma Value'].iloc[result['Best Accuracy'].idxmax()]}

    X_train, _, y_train, _ = samples[result['Best Accuracy'].idxmax()]
    plot_learning_curve(X_train, y_train, best_params)

if __name__ == "__main__":
    main()

train_sizes, train_scores, test_scores = learning_curve(svm, X_train, y_train, cv=10, scoring='accuracy',train_sizes=np.linspace(0.01, 1.0, 50))
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Score on Training')
plt.xlabel('Training')
plt.ylabel('Accuracy')
plt.title('Convergence Graph')
plt.show()
