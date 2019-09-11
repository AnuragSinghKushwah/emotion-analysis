from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def apply_lr(train_x, train_y, test_x, test_y):
    "Function to apply logistic regression on the dataset"
    try:
        final_ngram = LogisticRegression(C=0.5)
        final_ngram.fit(train_x, train_y)
        predictions = final_ngram.predict(test_x)
        accuracy = accuracy_score(test_y,predictions)
        print("Logistic Regression : Final Accuracy: %s "% accuracy)
        return accuracy, predictions
    except Exception as e:
        print("Exception in applying logistic regression : ",e)
        pass

def apply_svm(train_x, train_y, test_x, test_y):
    "Function to apply SVM on the dataset"
    try:
        final_svm_ngram = LinearSVC(C=0.01)
        final_svm_ngram.fit(train_x, train_y)
        predictions = final_svm_ngram.predict(test_x)
        accuracy = accuracy_score(test_y,predictions)
        print("SVC Algorithm \t : Final Accuracy: %s" % accuracy)
        return accuracy,predictions
    except Exception as e:
        print("Exception in applying svm : ",e)
        pass