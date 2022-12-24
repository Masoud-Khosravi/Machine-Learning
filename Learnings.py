class Classifications:
    X = None
    y = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    model = None
    test_size = 0.1
    is_tensor = False

    def __init__(self, X, y, test_size=0.1):
        """
        X - independent features(excluding target variable)
        y - dependent variables, called (target).
        test_size : for evaluate data after training
        """
        self.X = X
        self.y = y
        self.test_size = test_size
        self._train_test_split()

        # create an inner class object
        self.evaluates = self.Evaluates(self)

    def _train_test_split(self):
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=self.test_size,
                                                                                random_state=42)

    def random_forest(self, **kwargs):
        """
        Default values is:
            n_estimators=128,
            criterion='log_loss',
            class_weight=None,
            bootstrap=False,
            oob_score=False,
            warm_start=True,
            max_features='log2',
            random_state=None,
            verbose=0,
            n_jobs=-1
            
        returns model
        """
        if len(kwargs) < 1:
            kwargs = dict(n_estimators=128,
                          criterion='log_loss',
                          class_weight=None,
                          bootstrap=False,
                          oob_score=False,
                          warm_start=True,
                          max_features='log2',
                          random_state=None,
                          verbose=0,
                          n_jobs=-1)

        self.is_tensor = False
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**kwargs)
        model.fit(self.X_train, self.y_train)
        # if is_trade::
        #     from sklearn.utils import shuffle
        #     df = shuffle(df)
        #     X = df.drop('BS', axis=1)
        #     y = pd.to_numeric(df['BS'], downcast='integer')
        #     model.fit(X, y)
        self.model = model
        return model

    def ada_boost(self, **kwargs):
        """
        Default values is:
            n_estimators=384,
            learning_rate=0.7,
            algorithm='SAMME.R',
            random_state=None
                        
        returns model
        """
        if len(kwargs) < 1:
            kwargs = dict(n_estimators=384,
                          learning_rate=0.7,
                          algorithm='SAMME.R',
                          random_state=None)

        self.is_tensor = False
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(**kwargs)
        model.fit(self.X_train, self.y_train)

        self.model = model
        return model

    def k_nearest_neighbors(self, **kwargs):
        """
        Default values is:
                    algorithm='auto',
                    leaf_size=10,
                    metric='minkowski',
                    n_neighbors=4,
                    p=1,
                    weights=distance
                    
        returns model            
        """
        if len(kwargs) < 1:
            kwargs = dict({'algorithm': 'auto',
                           'leaf_size': 10,
                           'metric': 'minkowski',
                           'n_neighbors': 4,
                           'p': 1,
                           'weights': 'distance'})

        self.is_tensor = False
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(**kwargs)
        model.fit(self.X_train, self.y_train)
        self.model = model
        return model

    def decision_trees(self, **kwargs):
        """
        Default values is:
                    criterion='entropy',
                    splitter='random',
                    random_state=3
                    
        returns model            
        """
        if len(kwargs) < 1:
            kwargs = dict({'criterion': 'entropy',
                           'splitter': 'random',
                           'random_state': 3})

        self.is_tensor = False
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(**kwargs)
        model.fit(self.X_train, self.y_train)
        self.model = model
        return model

    def support_vector_machine(self, **kwargs):
        """
        Default values is:
                    gamma='entropy',
                    splitter='random',
                    verbose=True
                    
        returns model            
        """
        if len(kwargs) < 1:
            kwargs = dict({'gamma': 'scale',
                           'kernel': 'poly',
                           'verbose': True})

        self.is_tensor = False
        from sklearn.svm import SVC
        model = SVC(**kwargs)
        model.fit(self.X_train, self.y_train)
        self.model = model
        return model

    def tensorflow(self, epochs=700, *kwargs):
        """
        Default values is:
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_crossentropy']
                    
        returns model    
        """
        if len(kwargs) < 1:
            kwargs = dict(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['sparse_categorical_crossentropy'])

        self.is_tensor = True
        # import tensorflow as tf
        from tensorflow import keras

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(units=11, activation='relu', input_shape=(11,)))
        model.add(keras.layers.Dense(units=22, activation='relu'))
        # model_t.add(Dropout(0.2))
        model.add(keras.layers.Dense(units=33, activation='relu'))
        model.add(keras.layers.Dense(units=22, activation='relu'))
        model.add(keras.layers.Dense(units=11, activation='relu'))
        model.add(keras.layers.Dense(units=6, activation='relu'))
        model.add(keras.layers.Dense(3, activation='softmax'))

        # Compile model
        model.compile(**kwargs)

        model.fit(x=self.X_train, y=self.y_train, epochs=epochs, validation_data=(self.X_test, self.y_test), verbose=0)

        self.model = model
        return model

    class Evaluates:
        def __init__(self, parent):
            self.parent = parent

        def accuracy(self):
            model = self.parent.model
            y_pred = model.predict(self.parent.X_test)
            from sklearn.metrics import accuracy_score
            acs = accuracy_score(self.parent.y_test, y_pred)
            return acs

        def plot_confusion_matrix(self):
            from sklearn.metrics import accuracy_score, balanced_accuracy_score
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            import numpy as np
            import matplotlib.pyplot as plt

            is_tensor = self.parent.is_tensor
            model = self.parent.model
            X_test = self.parent.X_test
            y_test = self.parent.y_test
            if is_tensor:
                predict_x = model.predict(X_test)
                y_pred = np.argmax(predict_x, axis=1)
            else:
                y_pred = model.predict(X_test)

            print('Accuracy: ', accuracy_score(y_test, y_pred))
            print('B_Accuracy: ', balanced_accuracy_score(y_test, y_pred))

            conf_m = confusion_matrix(y_test, y_pred)

            print_final = True
            if print_final:
                true_signal = conf_m[1, 1] + conf_m[2, 2]
                false_signal = conf_m[:, 1:].sum() - true_signal
                final_score = true_signal * 2 - false_signal
                print('final_score=', final_score)

            cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_m)
            cm_display.plot()
            plt.show()


class Regression:
    X = None
    y = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    model = None
    test_size = 0.1
    is_tensor = False
    df_y = None

    def __init__(self, X, y, test_size=0.1):
        """
        X - independent features(excluding target variable)
        y - dependent variables, called (target).
        test_size : for evaluate data after training
        """
        self.X = X
        self.y = y
        self.test_size = test_size
        self._train_test_split()

        # create an inner class object
        self.evaluates = self.Evaluates(self)

    def _train_test_split(self):
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=self.test_size,
                                                                                random_state=42)

    def support_vector_regression(self, **kwargs):
        """
        Default values is:
            kernel = "rbf",
            degree = 3,
            gamma = "scale",
            tol = 0.001,
            C = 1,
            epsilon = 0.1
            ,...
        returns model
        """
        if len(kwargs) < 1:
            kwargs = dict(kernel="rbf",
                          degree=3,
                          C=1,
                          epsilon=0.1)

        self.is_tensor = False
        from sklearn.svm import SVR
        model = SVR(**kwargs)
        model.fit(self.X_train, self.y_train)

        self.model = model
        return model

    def decision_tree_regressor(self, **kwargs):
        """
        Default values is:
            criterion = "squared_error",
            splitter = "best",
            random_state = None
            ,...

        returns model
        """
        if len(kwargs) < 1:
            kwargs = dict(criterion="squared_error",
                          splitter="best")

        self.is_tensor = False
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(**kwargs)
        model.fit(self.X_train, self.y_train)

        self.model = model
        return model

    def ridge(self, **kwargs):
        """
        Default values is:
            alpha: float = 1
            ,...

        returns model
        """
        if len(kwargs) < 1:
            kwargs = dict(alpha=1)

        self.is_tensor = False
        # from sklearn.preprocessing import PolynomialFeatures
        # poly_converters = PolynomialFeatures(degree=2, include_bias=False)
        # poly_X = poly_converters.fit_transform(self.X)
        # from sklearn.model_selection import train_test_split
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(poly_X, self.y, test_size=0.1,
        #                                                                         random_state=42)
        from sklearn.linear_model import Ridge
        model = Ridge(**kwargs)
        model.fit(self.X_train, self.y_train)

        self.model = model
        return model

    def tensorflow(self, epochs=800, *kwargs):
        """
        Default values is:
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            loss=tf.keras.losses.mean_squared_error,

        returns model
        """

        # Because My Cpu is faster than vga, so I create this model by cpu
        # if you have Better VGA you can delete this two rows
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # only cpu

        import tensorflow as tf
        if len(kwargs) < 1:
            kwargs = dict(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                          loss=tf.keras.losses.mean_squared_error)
        self.is_tensor = True

        # from tensorflow import keras
        from keras.models import Sequential
        from keras.layers import Dense, Dropout

        model = Sequential()
        model.add(Dense(units=32, activation='relu', input_shape=(11,)))
        model.add(Dropout(0.1))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(units=16, activation='relu'))
        model.add(Dense(units=1))

        # Compile model
        model.compile(**kwargs)

        model.fit(x=self.X_train, y=self.y_train, epochs=epochs, validation_data=(self.X_test, self.y_test), verbose=0)

        self.model = model
        return model

    class Evaluates:
        def __init__(self, parent):
            self.parent = parent

        def mean_errors(self, printing=True):
            """
            if you want print result :printing=True
            :return: Root_mean_squared_error
            """
            model = self.parent.model
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            import pandas as pd
            import numpy as np
            if self.parent.is_tensor:
                predict_x = model.predict(self.parent.X_test)
                y_pred = pd.Series(predict_x.reshape(len(self.parent.y_test), ))
            else:
                y_pred = pd.Series(model.predict(self.parent.X_test))

            pred_df = pd.DataFrame(self.parent.y_test).reset_index(drop=True)
            pred_df = pd.concat([pred_df, y_pred], axis=1)
            pred_df.columns = ['y_test', 'y_pred']

            Root_MSE = 0
            if printing:
                MAE = mean_absolute_error(pred_df['y_test'], pred_df['y_pred'])
                MSE = mean_squared_error(pred_df['y_test'], pred_df['y_pred'])
                Root_MSE = np.sqrt(MSE)
                print('MAE=', MAE)
                print('MSE=', MSE)
                print('Root MSE=', Root_MSE)

            self.parent.df_y = pred_df
            return Root_MSE

        def plot_df_y(self):
            # to crate df_y
            self.mean_errors(printing=False)

            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.scatterplot(data=self.parent.df_y, x='y_test', y='y_pred')
            plt.show()
