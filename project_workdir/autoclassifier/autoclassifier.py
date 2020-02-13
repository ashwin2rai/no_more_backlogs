# -*- coding: utf-8 -*-

# Think about how the last column (target column) should be, category or binary 

from .utils import create_dir_link

import pandas as pd
import numpy as np
import pickle


class AutoClassifier:
    
    def __init__(self, orig_df, text_col=None):
        assert orig_df.iloc[:,-1].value_counts().shape[0] == 2, "Error: The last column should be the target column and should be binary (True/False, 0/1, Success/Failure etc.) "
        
        self.orig_df = orig_df
        self.text_col = text_col
        
    def preprocess_block(self, cat_var_limit=10, bin_rep=1,
                     max_tfidf_features=100, ngram_range=(1,2),
                     use_feat_select=True, 
                     alpha_space=np.linspace(0.01,0.02,20), 
                     random_state_var=np.random.randint(10000), test_size_var=.3, load_preproc = None,
                     text_fillna = 'No Text', text_clean=False, pre_proc_filename = 'preproc.sav'):
        """
        Preprocessing block: used to preprocess and transform the data columns
        ---------------------------------------------------------------------------
        -df_all (DataFrame): DataFrame with all the data, last column should be 
        target variable
        -text_col (str): name of the text column, default is None for no text columns
        -cat_var_limit (int):tai greatest number of unique values in a column to qualify 
        for conversion into a category column 
        -bin_rep (int): style of integer representation for category variables, 0 for 
        binary integer representation, 1 for 0 to nclass-1 representation
        -max_tfidf_features (int): maximum number of features after vectorizing text 
        column using tfidf metric
        -ngram_range (tuple): 2 tuple consisting of start and end point of ngram
        -use_feat_select (bool): True for applying feature selection using LASSO for 
        non-text columns
        -alpha_space (array of float): testing space for alpha parameter of LASSO
        -random_state_var (int): Random seed for train-test-split
        -test_size_var (float): ratio of test versus train split 
        -load_preproc
        ---------------------------------------------------------------------------
        """
        if load_preproc is None:        
            pre_proc_dict = {}
            pre_proc_dict['text_col'] = self.text_col
            pre_proc_dict['max_tfidf_features'] = max_tfidf_features
            pre_proc_dict['ngram_range'] = ngram_range
            pre_proc_dict['cat_var_limit'] = cat_var_limit
            pre_proc_dict['bin_rep'] = bin_rep
            pre_proc_dict['text_clean'] = text_clean
            pre_proc_dict['text_fillna'] = text_fillna
            
        else:
            try:
                pre_proc_dict = pickle.load(open(create_dir_link(filename = load_preproc), 'rb'))
            except:
                raise IOError('ERROR: Could not load preprocess file. Check filename and/or directory')
            max_tfidf_features = pre_proc_dict['max_tfidf_features']
            ngram_range = pre_proc_dict['ngram_range']
            cat_var_limit = pre_proc_dict['cat_var_limit']
            bin_rep = pre_proc_dict['bin_rep']
            text_clean = pre_proc_dict['text_clean']
            text_fillna = pre_proc_dict['text_fillna']
            
        text_col = pre_proc_dict['text_col']
        
        df_all = self.orig_df
        df_preproc = self._preprocess_df(df_all, pre_proc_dict, load_preproc)
        
        if load_preproc is None:
            if text_col is not None:
                text_numeric_matrix = self._preprocess_df_text(df_all, pre_proc_dict, load_preproc)
           
                self._feat_select(df_preproc, pre_proc_dict, pre_proc_filename, text_numeric_matrix.toarray(), 
                               test_size_var=test_size_var, 
                               use_feat_select=use_feat_select, 
                               alpha_space=alpha_space, 
                               random_state_var=random_state_var, plot=True)
            else:
                self._feat_select(df_preproc, pre_proc_dict, pre_proc_filename, use_feat_select=use_feat_select, 
                               test_size_var=test_size_var,
                               alpha_space=alpha_space, plot=True,
                               random_state_var=random_state_var)
        else:
            df_preproc = df_preproc[pre_proc_dict['feat_columns']]
            if text_col is not None:
                text_numeric_matrix = self._preprocess_df_text(df_all, pre_proc_dict, load_preproc)
                x_vals = np.concatenate((df_preproc.values,text_numeric_matrix.toarray()), axis=1) 
                return x_vals
            else:
                return df_preproc.values

    def _preprocess_df(self, df_all, pre_proc_dict, preproc):
    
        text_col = pre_proc_dict['text_col']
        cat_var_limit = pre_proc_dict['cat_var_limit']
        bin_rep = pre_proc_dict['bin_rep']
        
        if text_col is not None:
            df = df_all.drop([text_col], axis=1).copy() #All columns except text
        else:
            df = df_all.copy()
        
        df = self._impute_most_freq(self._convert_cat_cols(df,cat_var_limit))
        
        print(df.iloc[:,-1].dtype)
        
        if df.iloc[:,-1].dtype.name == 'category':
            from sklearn.preprocessing import LabelEncoder
            df.iloc[:,-1] = LabelEncoder().fit_transform(df.iloc[:,-1])
            
        cat_col_df = df.select_dtypes(include=['category'])
        cat_col_df_cols = cat_col_df.columns.copy()
        
        if not cat_col_df.empty:
            if bin_rep:
                df[cat_col_df.columns] = self._convert_cat_labels(cat_col_df, pre_proc_dict, preproc) 
                #Transforms a category variable column into an integer variable 
                #column
            else:
                temp_df = self._convert_cat_onehot(cat_col_df,pre_proc_dict, preproc) 
                temp_df = temp_df.drop(cat_col_df_cols, axis = 1)
                df = df.drop(cat_col_df_cols, axis = 1)
                df = pd.concat([pd.concat([df.iloc[:,:-1],temp_df],sort=False,axis=1),df.iloc[:,-1]],sort=False,axis=1)
                #Transforms a category variable column using OneHotEncoding
                
        assert (df.notnull().all().all()), 'ERROR: NaNs present in DataFrame, please clean data.'
        assert df.select_dtypes(include=['object']).empty, 'ERROR: Some columns could not be converted to category valuables and encoded. Please segment non-numerical data columns (that is not the text column) or increase cat_var_limit'
        
        return df

    def _preprocess_df_text(self, df_all, pre_proc_dict, preproc):
        
        from sklearn.feature_extraction.text import TfidfVectorizer 
        
        text_col = pre_proc_dict['text_col']
        max_tfidf_features = pre_proc_dict['max_tfidf_features']
        ngram_range = pre_proc_dict['ngram_range']
        text_clean = pre_proc_dict['text_clean']
        text_fillna = pre_proc_dict['text_fillna']
        
        df_all[text_col] = df_all[text_col].fillna(text_fillna)
            
        if text_clean:
            try:
                df_all[text_col] = self._clean_text(df_all[text_col])
            except:
                print('WARNING: Cannot clean text, recheck text column')
                
        if preproc is None:
            pre_proc_dict['TfidVect'] = TfidfVectorizer(max_features=max_tfidf_features,
                                                  ngram_range=ngram_range, 
                                                  stop_words='english').fit(df_all[text_col])
            
            text_numeric_matrix = pre_proc_dict['TfidVect'].transform(df_all[text_col])
        else:
            text_numeric_matrix = pre_proc_dict['TfidVect'].transform(df_all[text_col])
        
        return text_numeric_matrix

    def _clean_text(self, text_series):
        """
        Cleans a column of Tweets. Removes all special characters, 
        websites, mentions.
        """
        from re import sub as resub
        text_series = text_series.apply(
             lambda x:resub(
                    r"[^A-Za-z0-9 ]+|(\w+:\/\/\S+)|htt", " ", x)
                         ).str.strip().str.lower()
        return text_series
        
    def _convert_cat_cols(self, df, cat_var_limit=10, verbose=False):                
        """
        Converts columns with a small amount of unique values that are of
        type Object into categorical variables.
        Number of unique values defined by cat_var_limit
        """
        cat_var_true = df.apply(lambda x: 
                                len(x.value_counts()) < cat_var_limit)
        object_type_true = df.apply(lambda x: 
                             x.value_counts().index.dtype == 'O')
        if cat_var_true[object_type_true].any():
            df[cat_var_true[object_type_true].index] = df[cat_var_true[object_type_true].index].astype('category')
            if verbose:
                print(df[cat_var_true[object_type_true].index].describe())
        return df

    def _impute_most_freq(self, df):
        """
        Imputes the most frequent value in place of NaN's
        """
        most_freq = df.apply(lambda x: x.value_counts().index[0])
        return df.fillna(most_freq)

    def _convert_cat_labels(self, df, pre_proc_dict, preproc):
        """
        Converts columns with factors into integer representation
        """
        from sklearn.preprocessing import LabelEncoder
        if preproc is None:
            LEtransformer = {}
            for col in df.columns:
                LEtransformer[col] = LabelEncoder().fit(df[col])
                df[col] = LEtransformer[col].transform(df[col])
            pre_proc_dict['LabelEncode'] = LEtransformer
        else:
            LEtransformer = pre_proc_dict['LabelEncode']
            for col in df.columns:
                df[col] = LEtransformer[col].transform(df[col])     
        return df
    
    def _convert_cat_onehot(self, df, pre_proc_dict, preproc):
        """
        Converts columns with factors into integer representation
        """
        from sklearn.preprocessing import OneHotEncoder
        if preproc is None:
            OHtransformer = {}
            OHtransformer['col_list'] = df.columns
            OHtransformer['OHencode'] = OneHotEncoder().fit(df)
            temp_arr = OHtransformer['OHencode'].transform(df).toarray()
            for i in range(temp_arr.shape[1]):
                df['OneHotEncCol{}'.format(i)] = temp_arr[:,i]
            pre_proc_dict['OneHotEncode'] = OHtransformer
        else:
            OHtransformer = pre_proc_dict['OneHotEncode']
            temp_arr = OHtransformer['OHencode'].transform(df).toarray()
            for i in range(temp_arr.shape[1]):
                df['OneHotEncCol{}'.format(i)] = temp_arr[:,i]     
        return df
        
    def _feat_select(self, df, pre_proc_dict, pre_proc_filename, text_mat=None, test_size_var=0.3, 
                    alpha_space=np.linspace(0.01,0.02,20), 
                    random_state_var=np.random.randint(10000), use_feat_select=True, plot=True):
        """
        Performs feature selection on a dataframe with a single target 
        variable and n features. Test train split is also performed and only 
        splits of selected features are returned. Feature selection performed 
        using LASSO weight shrinking.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import GridSearchCV
        from sklearn.linear_model import Lasso
        import matplotlib.pyplot as plt
        
        pre_proc_dict['feat_columns'] = df.columns[:-1]
        
        x_train, x_test, y_train, y_test = train_test_split(
            df.iloc[:,:-1], df.iloc[:,-1], test_size=test_size_var, 
            random_state=random_state_var, stratify=df.iloc[:,-1])
    
        if use_feat_select:
            param_grid = {'alpha': alpha_space}
            lasso_gcv = GridSearchCV(Lasso(normalize=False), param_grid, cv=5,
                                     n_jobs=-1, iid=True)
            lasso_coeffs = lasso_gcv.fit(x_train, y_train).best_estimator_.coef_
            if plot:
                plt.barh(y=range(len(df.columns[:-1])), width=np.abs(lasso_coeffs),
                         tick_label=df.columns[:-1].values)
                plt.ylabel('Column features')
                plt.xlabel('Coefficient score')
                plt.xticks(rotation=90)
                plt.show()
            try:
                select_feats = df.columns[:-1][np.abs(lasso_coeffs) > 0].values
                pre_proc_dict['feat_columns'] = select_feats
            except:
                print('WARNING: Lasso Coefficients all turned out to be 0 or could not be calculated. Check your dataset or switch off feature selection.')
                         
            x_train = x_train.loc[:,select_feats]
            x_test = x_test.loc[:,select_feats]
        
        if text_mat is not None:
            self.x_train = np.concatenate((x_train.values,text_mat[x_train.index,:]), axis=1)
            self.x_test = np.concatenate((x_test.values,text_mat[x_test.index,:]), axis=1)
        else:
            self.x_train = x_train.values
            self.x_test = x_test.values

        self.y_train = y_train.values 
        self.y_test = y_test.values

        pickle.dump(pre_proc_dict, open(create_dir_link(filename=pre_proc_filename), 'wb'))

    def shallow_model_fit(self, scaler_ch=0,
                      logreg_C=[0.8,1,1.2,1.4], knn_neigh=np.arange(3,16),
                      svc_c=[0.5,1,1.5,2,2.5,2.6], gb_max_depth=[2,3,4,5],
                      gb_n_est=[40,60,80,100], verbose=True, save=True, scoring = 'accuracy',
                      model_file='Trained_shallow_models.sav'): 
        """
        This function will fit and test several shallow classfication models and 
        save them, models include:
        'logreg': Logistic Regression using the lbfgs solver
        'knnstep': K Nearest Neighbors
        'svcstep': Support Vector Classification model
        'gradbooststep': Gradient Boosted Classification Trees
        
        Scaling options include:
        MinMaxScaler between a range of -1 and 1
        Normalizer 
        StandardScaler
        
        The function will also run a 5 fold cross validated grid search for 
        hyperparameter optimization
        ---------------------------------------------------------------------------
        -x_train (DataFrame or ndarray): Training data consisting of features   
        -x_test (DataFrame or ndarray): Testing data consisting of features
        -y_train (DataFrame, Series or ndarray): Training data for predictions 
        (single class only)   
        -y_train (DataFrame, Series or ndarray): Testing data for predictions 
        (single class only)   
        -scaler_ch (int): Decides which scaler to use, 0 for MinMaxScaler, 1 for 
        Normalizer, 2 for StandarScaler
        -logreg_C (list of float): Hyperparameter space for C to be used in the 
        Log Reg Classifier
        -knn_neigh (list of int): Hyperparameter space for number of neighbors to 
        be used in the KNN Classifier
        -svc_c (list of float): Hyperparameter space for C to be used in the 
        Support Vector Classifier
        -gb_max_depth (list of int): Hyperparameter space for max depth to be used 
        in Gradient Boosted Classifier Trees
        -gb_n_est (list of int): Hyperparameter space for number of estimators to 
        be used in Gradient Boosted Classifier Trees
        -verbose (bool): Prints out details if True
        -save (bool): Switch for saving the trained models in an external data file
        -model_file (str): Filename for storing all the trained models
        ---------------------------------------------------------------------------
        """   
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import Normalizer
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC
        
        x_train, x_test, y_train, y_test = self.x_train, self.x_test, self.y_train, self.y_test

        scaler = [('Scaler', MinMaxScaler(feature_range=(-1, 1))),\
                 ('Scaler', Normalizer()), ('Scaler', StandardScaler())]
    
        classifiers = [('logreg', LogisticRegression(solver='lbfgs', 
                                                     max_iter=1000)),\
            ('knnstep', KNeighborsClassifier()),\
            ('svcstep', SVC(gamma='scale', probability=True)),\
             ('gradbooststep', GradientBoostingClassifier(subsample=.8))]
    
        parameters = {'logreg': {'logreg__C': logreg_C} ,\
                  'knnstep': {'knnstep__n_neighbors': knn_neigh},\
                  'svcstep': {'svcstep__C': svc_c},\
                 'gradbooststep': {'gradbooststep__max_depth': gb_max_depth,\
                                  'gradbooststep__n_estimators': \
                                  gb_n_est}} 
        model_dict = {}
    
        for clf in classifiers:
            pipeline = Pipeline([scaler[scaler_ch], clf])
            print('\nAnalysis for : ' + clf[0])
            gcv = GridSearchCV(pipeline, param_grid=parameters[clf[0]],
                               cv=5, iid=True, scoring = scoring)
            gcv.fit(x_train, y_train)
            model_dict[clf[0]] = (gcv, gcv.score(x_test, y_test))
            if verbose:
                print('The best parameters for: ' + clf[0] + ' are :' +
                      str(gcv.best_params_))
                print(pd.DataFrame(gcv.cv_results_)
                      [['mean_test_score','params']])
            print('The score for ' + clf[0] + ' is ' + 
                  str(gcv.score(x_test, y_test))) 
        
        if save:
            pickle.dump(model_dict, open(create_dir_link(filename=model_file), 'wb'))
        
        self.model_dict = model_dict
    
    def deep_model_fit(self, scaler_ch=0, 
                   patience_val=2, validation_split_val=.2, epochs_val=20,
                   verbose=True, save=True, model_file='Trained_deep_model.h5'):
        """
        This function will fit and test a Deep Neural Network that uses ReLu 
        and softmax activation functions. It also uses an EarlyStopper
        
        Scaling options include:
        MinMaxScaler between a range of -1 and 1
        Normalizer 
        StandardScaler
        
        ---------------------------------------------------------------------
        -x_train (DataFrame or ndarray): Training data consisting of features   
        -x_test (DataFrame or ndarray): Testing data consisting of features
        -y_train (DataFrame, Series or ndarray): Training data for predictions 
        (single class only)   
        -y_train (DataFrame, Series or ndarray): Testing data for predictions 
        (single class only)    
        -scaler_ch (int): Decides which scaler to use, 0 for MinMaxScaler, 
        1 for Normalizer, 2 for StandarScaler
        -patience_val (int): Number of epochs to monitor before exiting 
        training if no major changes in accuracy occurs
        -validation_split_val (float): ratio of split of dataset for testing 
        purposes
        -epochs_val (int): Max number of epochs to train
        -verbose (bool): Model training details will be printed out if True
        -save (bool): Switch for saving the trained models in an external data 
        file
        -model_file (str): Filename for storing the trained model. Must be H5 
        extension
        ----------------------------------------------------------------------
        """   
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import Normalizer
        from sklearn.preprocessing import StandardScaler
        from keras.layers import Dense
        from keras.models import Sequential
        from keras.callbacks import EarlyStopping

        x_train, x_test, y_train, y_test = self.x_train, self.x_test, self.y_train, self.y_test

        
        scaler_list = [('Scaler', MinMaxScaler(feature_range=(-1, 1))),\
                  ('Scaler', Normalizer()),\
                  ('Scaler', StandardScaler())]
    
        scaler = scaler_list[scaler_ch][1]
        
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)
    
        n_cols = x_train.shape[-1]
    
        model = Sequential()
        model.add(Dense(5, activation='relu', input_shape=(n_cols,)))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(2,activation='softmax'))
    
        early_stop_monitor = EarlyStopping(patience=patience_val)
        model.compile (optimizer='adam',
                       loss= 'categorical_crossentropy',
                       metrics=['accuracy'])
        train_dp_model=model.fit(x_train, pd.get_dummies(y_train).values,
                                 validation_split=validation_split_val, 
                                 epochs = epochs_val,
                                 callbacks =[early_stop_monitor],
                                 verbose=verbose)
    
        print('Loss metrics: ' + str(train_dp_model.history['loss'][-1]))
        
        pred_prob = model.predict(x_test)
        accuracy_dp = np.sum((pred_prob[:,1]>=0.5)==y_test) / len(y_test)
        print('DNN Testing accuracy: ' + str(accuracy_dp)) 
        
        if save:
            model.save(create_dir_link(filename=model_file))   
        
        self.model = model


    def load_model(self, type_model='shallow', filename='Trained_shallow_models.sav', 
                   clf='logreg'):
        """
        This function is used to load a previously saved trained model. 
        The model will have been saved in an external file.
        
        ---------------------------------------------------------------------
        -type (str): 'shallow' to load a trained shallow model, 
        'deep' to load a trained deep model
        -filename (str): Name of the file with the saved model
        -clf (str): Only used for retrieving shallow models, this is the label 
        of the classifier -
        'logreg': Logistic Regression using the lbfgs solver
        'knnstep': K Nearest Neighbors
        'svcstep': Support Vector Classification model
        'gradbooststep': Gradient Boosted Classification Trees
        ---------------------------------------------------------------------
        """
        assert (type_model in ['shallow','deep']), "ERROR: Wrong label for argument 'type'." 
        if type_model == 'shallow':
            try:
                model_dict = pickle.load(open(create_dir_link(filename = filename), 'rb'))
            except:
                raise IOError('ERROR: Could not load file. Check filename.')
            for label, model in model_dict.items():
                print(label + ' score is: ' + str(model[1]))
                print('Selecting model: '+ str(clf))
            
            self.model = model_dict[clf][0]
            return self
        
        elif type_model == 'deep':
            from keras.models import load_model 
            try:
                model = load_model(create_dir_link(filename = filename))
            except:
                raise IOError('ERROR: Could not load file. Check filename.')
            print('Model Summary: ')
            model.summary()
            self.model = model
            return self

    def predict(self, val_mat, clf = None):
        if not clf:
            try:
                return self.model.predict(val_mat)
            except:
                raise ValueError('ERROR: Could not predict. Check if model is loaded.')
        else:
            try:
                return self.model_dict[clf][0].predict(val_mat)
            except:
                raise ValueError('ERROR: Could not predict. Check if model is loaded.')
            
    def predict_proba(self, val_mat, clf = None):
        if not clf:
            try:
                return self.model.predict_proba(val_mat)
            except:
                raise ValueError('ERROR: Could not predict. Check if model is loaded.')
        else:
            try:
                return self.model_dict[clf][0].predict_proba(val_mat)
            except:
                raise ValueError('ERROR: Could not predict. Check if model is loaded.')

    def score(self, x_test, y_test, clf = None):
        if not clf:
            try:
                return self.model.score(x_test, y_test)
            except:
                raise ValueError('ERROR: Could not predict. Check if model is loaded.')
        else:
            try:
                return self.model_dict[clf][0].score(x_test, y_test)
            except:
                raise ValueError('ERROR: Could not predict. Check if model is loaded.')
        