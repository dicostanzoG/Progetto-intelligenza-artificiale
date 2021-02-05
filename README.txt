## Prerequisiti

Per poter eseguire correttamente il codice è necessario:

* [Scikit-Learn](http://scikit-learn.org/stable/index.html#) da cui è possibile ottenere diverse funzionalità per modificare il dataset
* [Scikit-Image](https://scikit-image.org/) per convertire le immagini in scala di grigi
* [Matplotlib](https://matplotlib.org/) 
* [Imblearn](https://imbalanced-learn.org/) per bilanciare le classi
* [Numpy] (https://numpy.org/)
* [Scipy] (https://docs.scipy.org/doc/scipy/reference/index.html)

* train set (train_32x32.mat), test set (test_32x32.mat), extra file (extra _32x32.mat): (http://ufldl.stanford.edu/housenumbers/)

## Esecuzione 

Il file su cui eseguire il run è "main.py", che restituisce i valori dell'errore del training set e dell'errore del testing set in funzione della dimensione del training set.

## Riferimenti

Nella realizzazione del progetto sono stati consultati:

* [Documentazione Scikit-Learn] (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html, 
                                 https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)
* [Documentazione Imblearn] (https://imbalanced-learn.org/stable/generated/imblearn.pipeline.Pipeline.html, https://imbalanced-learn.org/stable/generated/imblearn.over_sampling.RandomOverSampler.html, 
              https://imbalanced-learn.org/stable/generated/imblearn.under_sampling.RandomUnderSampler.html)
* [Stuart Russell, Peter Norvig. Intelligenza artificiale. Vol. 2: Un approccio moderno] 

