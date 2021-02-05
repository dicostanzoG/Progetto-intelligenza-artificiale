import scipy.io
import scikitplot as skplt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from skimage import color
from random import randint
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

def get_file(file):
    data = scipy.io.loadmat(file)
    X = data['X']
    y = data['y']
    y[y == 10] = 0
    return X, y


def get_images(n_img, file, digit1, digit2):
    X, y = get_file(file)
    x, y_tmp = [], []
    ran = randint(0, 5000)

    while len(y_tmp) < n_img:
        try:
            if y[ran]==digit1 or y[ran]==digit2:
                x.append((color.rgb2gray(X[:, :, :, ran])).flatten())
                y_tmp.append(int(y[ran]))
        except IndexError:
            X, y = get_file('extra_32x32.mat')
            if y[ran]==digit1 or y[ran]==digit2:
                x.append((color.rgb2gray(X[:, :, :, ran]).flatten()))
                y_tmp.append(int(y[ran]))
        ran += 1
    return x, np.ravel(y_tmp)


def show_img(x, y, predicted, n_img):
    nplots = 6
    x = np.reshape(x, (n_img, 32, 32))
    for j in range(nplots):
        plt.subplot(3, 3, j+1)
        plt.imshow(x[j, :, :], cmap='gray')
        plt.title('\ny=%d, pred=%d' % (y[j], predicted[j]))
        plt.axis('off')
    plt.show()


def scale_images(train, test):
    sc = StandardScaler()
    train = sc.fit_transform(train)
    test = sc.transform(test)
    return train, test


def balanced_classes(X, y, n, digit1, digit2):
    unique, counts = np.unique(y, return_counts=True)
    print('\nClassi non bilanciate', dict(zip(unique, counts)))

    under = RandomUnderSampler(sampling_strategy={digit1:int(n/2)})
    over = RandomOverSampler(sampling_strategy={digit2:int(n/2)})
    pipeline = Pipeline(steps=[('o', over), ('u', under)])
    X, y = pipeline.fit_resample(X, y)

    unique, counts = np.unique(y, return_counts=True)
    print('Classi bilanciate', dict(zip(unique, counts)))
    return X, y


def c_matrix(y, pred, text):
    print(text)
    skplt.metrics.plot_confusion_matrix(y, pred, normalize=True)
    plt.show()


def get_train_test_sets(num, digit1, digit2):
    try:
        x_train, y_train = get_images(num, 'train_32x32.mat', digit1, digit2)
        x_test, y_test = get_images(int(num*0.3), 'test_32x32.mat', digit1, digit2)

        x_train, y_train = balanced_classes(x_train, y_train, num, digit1, digit2)
    except ValueError:
        x_train, y_train, x_test, y_test = get_train_test_sets(num, digit1, digit2)

    x_train, x_test = scale_images(x_train, x_test)
    return x_train, x_test, y_train, y_test



def analysis(num, digit1, digit2):

    x_train, x_test, y_train, y_test = get_train_test_sets(num, digit1, digit2)

    model = Perceptron()
    y_test_pred = model.fit(x_train, y_train).predict(x_test)
    y_train_pred = model.predict(x_train)

    score_train = accuracy_score(y_train, y_train_pred)
    score_test = accuracy_score(y_test, y_test_pred)

    error_train = 1 - score_train
    error_test = 1 - score_test

    #c_matrix(y_train, y_train_pred, 'train report')
    #c_matrix(y_test, y_test_pred, 'test report')

    #show_img(x_train, y_train, y_train_pred, num)
    #show_img(x_test, y_test, y_test_pred, int(num*0.3))

    print('Train Error %.3f, Test Error %.3f' % (error_train*100, error_test*100))
    return error_train, error_test

def main():
    k = 10
    n_iter = 5
    n_img, mean_train, mean_test = list(), list(), list()
    while pow(2, k) < 80000:
        n = pow(2, k)
        error_train, error_test, score_test = list(), list(), list()

        for i in range(n_iter):
            score = analysis(n, 2, 8)
            error_train.append(score[0])
            error_test.append(score[1])

        mean_error_train = np.mean(error_train)
        mean_error_test = np.mean(error_test)

        mean_train.append(mean_error_train)
        mean_test.append(mean_error_test)

        print('Test Size=%d, Train Error %.3f, Test Error %.3f\n' % (n, mean_error_train*100, mean_error_test*100))

        n_img.append(n)
        k += 1

    plt.plot(n_img, mean_train, label="train_set")
    plt.plot(n_img, mean_test, label="test_set")
    plt.xlabel("Training Size")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

