import os, sys, numpy as np, click
import time, re

def tokenize(sent, lower=True) :
    if lower :
        return [w for w in re.split(r"[^a-zA-Z]+", sent.lower()) if len(w) > 1]
    else :
        return [w for w in re.split(r"[^a-zA-Z]+", sent) if len(w) > 1]

def get_data(fname, xs=[], ys=[]) :
    with open(fname) as fin :
        for line in fin :
            words = line.strip().split(' ')
            ys.append([w for w in words if w.startswith('__label__')])
            xs.append(' '.join([w for w in words if not w.startswith('__label__')]))
    return xs, ys


def y_conversion(train, test) :
    y_cnt = np.unique([len(y) for y in train + test])
    labels = [label for y in train + test for label in y]
    labels = { label:id for id, label in enumerate(np.unique(labels).tolist()) }
    if y_cnt.size == 1 and y_cnt[0] == 1 :
        return np.array([[labels[y[0]]+1] for y in train]), np.array([[labels[y[0]]+1] for y in test]), {k:v+1 for k,v in labels.items()}
    else :
        new_train = np.zeros([len(train), len(labels)], dtype=int)
        for i, y in enumerate(train) :
            idx = np.array([ labels[label] for label in y ])
            if idx.size > 0 :
                new_train[i, idx] = 1
        new_test = np.zeros([len(test), len(labels)], dtype=int)
        for i, y in enumerate(test) :
            idx = np.array([ labels[label] for label in y ])
            if idx.size > 0 :
                new_test[i, idx] = 1
    return new_train, new_test, labels
@click.group()
def main() :
    pass



@main.command()
@click.option('-t', '--train')
@click.option('-v', '--test')
def tgrocery(train, test) :
    from tgrocery import Grocery
    trains = train.split(',')
    tests = test.split(',')
    
    truths, predictions = [], []
    for train, test in zip(trains, tests) :
        x_train, y_train = get_data(train, [], [])
        x_train = np.array(x_train)
    
        x_test, y_test = get_data(test, [], [])
        x_test = np.array(x_test)
        y_trains, y_tests, labels = y_conversion(y_train, y_test)

        grocery = Grocery('test')
        pred = np.zeros(y_tests.shape, dtype=int)
        for y_train, y_test, p in zip(y_trains.T, y_tests.T, pred.T) :
            train_src = np.vstack([y_train, x_train]).T.tolist()
            test_src = np.vstack([y_test, x_test]).T.tolist()
            grocery.train(train_src)
            b = grocery.test(test_src)
            p[:] = np.array(b.predicted_y).astype(int)
        truths.append(y_tests)
        predictions.append(pred)
    accs = np.zeros([len(truths)+1, 3])
    for truth, pred, acc in zip(truths, predictions, accs[:-1]) :
        a = np.sum((truth == pred) & (truth > 0)).astype(float)
        b = np.sum(truth > 0).astype(float) - a
        c = np.sum(pred > 0).astype(float) - a
        acc[:] = [a/(a+c), a/(a+b), 2*a/((a+c) + (a+b))]
    truths = np.hstack(truths)
    predictions = np.hstack(predictions)
    accs[-1, :] = np.sum(np.all(truths == predictions, 1)).astype(float)/truths.shape[0]
    print('tgrocery - Prec: {0}'.format(' '.join(['{0:.5f}'.format(x) for x in accs.T[0]])))
    print('tgrocery - Sens: {0}'.format(' '.join(['{0:.5f}'.format(x) for x in accs.T[1]])))
    print('tgrocery - Accu: {0}'.format(' '.join(['{0:.5f}'.format(x) for x in accs.T[2]])))
    
ft_model = None
def embedder(docs) :
    import fasttext as ft
    
    global ft_model
    if ft_model is None :
        ft_model = ft.load_model("/share_space/database_backup/wiki.en.bin")
    res = []
    for sent in docs :
        codes = []
        words = np.array(tokenize(sent, lower=True))
        for w in words :
            if w != '' :
                codes.append(ft_model.get_word_vector(w))
        res.append(np.mean(codes, axis=0))
    return np.array(res)
    
def vectorizer(train, test) :
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vector = TfidfVectorizer(tokenizer=tokenize)
    x1 = tfidf_vector.fit_transform(train).toarray()
    x2 = tfidf_vector.transform(test).toarray()
    return x1, x2


@main.command()
@click.option('-m', '--model', default='RandomForest,GaussianNB,LogisticRegression,LinearSVM,MLPClassifier')
@click.option('-t', '--train', required=True)
@click.option('-v', '--test')
@click.option('-e', '--embed', is_flag=True)
def skmodels(model, train, test, embed) :
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.neural_network import MLPClassifier
    
    grids = dict(
        RandomForest =  {'max_depth' : np.arange(1, 21, 5), 'min_samples_split':[2, 0.1, 0.3, 0.5], 'n_estimators':np.arange(50, 310, 50)}, 
        GaussianNB = {'var_smoothing':np.exp(np.arange(np.log(1e-5), np.log(5e-1), 20))}, 
        LogisticRegression = {'C': np.exp(np.arange(np.log(0.0001), np.log(10000), 30)), 'l1_ratio': [0., 0.1, 0.2, 0.5, 1.]}, 
        LinearSVM = {'C': np.exp(np.arange(np.log(0.0001), np.log(10000), 100))}, 
        MLPClassifier = {
            'hidden_layer_sizes': [(50,100,50), (100,)],
            'alpha': [0.0001, 0.001, 0.01, 0.05],
        }
    )
    models = dict(
        RandomForest = GridSearchCV(RandomForestClassifier(), grids['RandomForest'], n_jobs=-1, cv=5, iid=False, return_train_score=True), 
        GaussianNB = GridSearchCV(GaussianNB(), grids['GaussianNB'], n_jobs=-1, cv=5, iid=False, return_train_score=True), 
        LogisticRegression = GridSearchCV(LogisticRegression(max_iter=3000, solver='saga', multi_class='auto', penalty='elasticnet'), grids['LogisticRegression'], n_jobs=-1, cv=5, iid=False, return_train_score=True), 
        LinearSVM = GridSearchCV(LinearSVC(max_iter=3000), grids['LinearSVM'], n_jobs=-1, cv=5, iid=False, return_train_score=True), 
        MLPClassifier = GridSearchCV(MLPClassifier(max_iter=3000, activation='relu'), grids['MLPClassifier'], n_jobs=-1, cv=5, iid=False, return_train_score=True), 
    )
    
    trains = train.split(',')
    tests = test.split(',')
    datasets = []
    for train, test in zip(trains, tests) :
        x_train, y_train = get_data(train, [], [])
        x_test, y_test = get_data(test, [], [])
        y_trains, y_tests, labels = y_conversion(y_train, y_test)
        if embed :
            x_train2 = embedder(x_train)
            x_test2 = embedder(x_test)
        else :
            x_train2, x_test2 = vectorizer(x_train, x_test)
        datasets.append([x_train2, y_trains, x_test2, y_tests])
    for m in model.split(',') :
        try :
            accs = np.zeros(3)
            truths, predictions = [], []
            for x_train2, y_trains, x_test2, y_tests in datasets :
                pred = np.zeros(y_tests.shape, dtype=int)
                for y_train, y_test, p in zip(y_trains.T, y_tests.T, pred.T) :
                    clf = models[m]
                    clf.fit(x_train2, y_train)
                    import subprocess
                    subprocess.Popen('''ps -ef|grep semaph|grep python|awk '{print "kill "$2}'|bash''', shell=True).wait()
                    subprocess.Popen('''ps -ef|grep semaph|grep python|awk '{print "kill "$2}'|bash''', shell=True).wait()
                    p[:] = clf.best_estimator_.predict(x_test2)
                truths.append(y_tests)
                predictions.append(pred)
            accs = np.zeros([len(truths)+1, 3])
            for truth, pred, acc in zip(truths, predictions, accs[:-1]) :
                a = np.sum((truth == pred) & (truth > 0))
                b = np.sum(truth > 0) - a
                c = np.sum(pred > 0) - a
                acc[:] = [a/(a+c) if a+c>0 else 0, a/(a+b) if a+b>0 else 0, 2*a/((a+c) + (a+b)) if a+b+c>0 else 0]
            truths = np.hstack(truths)
            predictions = np.hstack(predictions)
            accs[-1, :] = np.sum(np.all(truths == predictions, 1))/truths.shape[0]
            print('{1} - Prec: {0}'.format(' '.join(['{0:.5f}'.format(x) for x in accs.T[0]]), m))
            print('{1} - Sens: {0}'.format(' '.join(['{0:.5f}'.format(x) for x in accs.T[1]]), m))
            print('{1} - Accu: {0}'.format(' '.join(['{0:.5f}'.format(x) for x in accs.T[2]]), m))
        except :
            pass

if __name__ == '__main__' :
    main()

