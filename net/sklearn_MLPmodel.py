from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.metrics import accuracy_score

def bpNet(XTrain, XTest, YTrain, YTest):
    params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
              {'solver': 'adam', 'learning_rate_init': 0.01},
              {'solver': 'adam', 'learning_rate_init': 0.001}]
    for param  in params:
        bp = MLPClassifier(hidden_layer_sizes=(500,), activation='relu',
                           random_state=0,
                           max_iter=400, **param)
        bp.fit(XTrain, YTrain.astype('int'))
        y_predict = bp.predict(XTest)
        print("BP神经网络准确度：",accuracy_score(y_predict,YTest))