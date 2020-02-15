# -*- coding: utf-8 -*-
from .utils import create_dir_link

import pandas as pd
import numpy as np
import pickle


class AutoClassifier:
    """
    Class AutoClassifier: AutoClassifier is a stand along package that can be used to create a pipeline to perform binary classification almost instantly.
    It performs data cleaning, converts eligible features into categories, imputatotion, feature selection, NLP, fits several classification models including a DNN model, performs hyperparameter tuning, and generates predictions with minimal user input.
    
     
    Initialization Parameters
    ----------
    orig_df: Pandas.DataFrame
        DataFrame with features and targets. Only two requirements for this is (i) the target vector should be binary and should be int or category type and should be the last column of the dataframe; (ii) All text columns should be concatenated into a single text feature vector.
    
    text_col: str, optional
        The name of the column with text. If text_col is provided, NLP methods will be applied to it. If None, then analysis is considered Non-NLP.
        Default None
        
    """  
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

        Parameters
        ----------

        cat_var_limit: int, optional
            the greatest number of unique values in a column to qualify for conversion into a category column 
            Default 10
        
        bin_rep: int, optional
            style of integer representation for category variables, 0 for binary integer representation (OneHotEncode), 1 for 0 to nclass-1 representation
            Default 1
            
        max_tfidf_features: int, optional
            maximum number of features to be used after vectorizing text column using tfidf metric
            Default 100
            
        ngram_range: tuple, optional
            2 tuple consisting of start and end point of ngram
            Default (1,2)
            
        use_feat_select: bool, optional
            If True then feature selection using LASSO for non-text columns is applied. Otherwise no feature selection will be performed
            Default True
        
        alpha_space: Numpy.array, optional
            Testing space for alpha parameter of LASSO used in hyperparameter tuning
            Default np.linspace(0.01,0.02,20) 
            
        random_state_var: int, optional
            Random seed for train-test-split
            Default np.random.randint(10000)
        
        test_size_var: float, optional
            Ratio of test versus train split
            Default 0.3

        text_fillna: str, optional
            Used to fill Empty elements in the text column
            Default No Text
            
        text_clean: Bool, optional
            If True then text is cleaned using regEx
            Default False
            
        pre_proc_filename: str, optional
            Preprocessor trained on this dataset will be saved using this filename. Next dataset can be preprocessed by loading this file. Only activated if preprocessor was not previously loaded. 
            Default: preproc.sav
            
        load_preproc: str, optional
            This loads the preprocessor trained on a training set. Use consistent preprocessing.
            If None then preprocessing is performed from scratch.
            Default None

        Returns
        -------
        
        4 - Tuple
            If preprocessing was run from scratch (load_preproc is None). Tuple includes train test matrices and vectors (x_train,x_test,y_train,y_test)
        
        Numpy.Array
            If preprocess was loaded (load_preproc is Not None). This is X matrix which is ready to be passed into the predict method. 
            
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
        """
        Method for preprocessing Non-text columns
        
        Parameters
        ----------

        df_all: Pandas.DataFrame
            Original DataFrame to be preprocessed
        
        pre_proc_dict: dict
            dictionary containing preprocessing objects

        preproc: str
            Filename of a previously trained preprocessor. Used here as an indicator, if None then preprocessing is being done for scratch.

        Returns
        -------
        
        Pandas.DataFrame
            Preprocessed non-text dataframe
        """

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
        """
        Method for preprocessing text column
        
        Parameters
        ----------

        df_all: Pandas.DataFrame
            Original DataFrame to be preprocessed
        
        pre_proc_dict: dict
            dictionary containing preprocessing objects

        preproc: str
            Filename of a previously trained preprocessor. Used here as an indicator, if None then preprocessing is being done for scratch.

        Returns
        -------
        
        Numpy.Array
            Preprocessed text TfIDF matrix
        """        
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
        Cleans a column of text. Removes all special characters, 
        websites, mentions etc.
        
        Parameters
        ----------

        text_series: Pandas.Series
        
        Returns
        -------
        
        Pandas.Series
            Cleaned text
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
        type Object into categorical variables. Number of unique values defined by cat_var_limit
        
        Parameters
        ----------

        df: Pandas.DataFrame
        
        cat_var_limit: int, optional
            the greatest number of unique values in a column to qualify for conversion into a category column 
            Default 10
            
        Verbose: bool
            If True prints summary of conversion
        
        Returns
        -------
        
        Pandas.DataFrame
            DataFrame with eligible columns converted into categoory columns
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
        
        Parameters
        ----------

        df: Pandas.DataFrame
        
        Returns
        -------
        
        Pandas.DataFrame
        """
        most_freq = df.apply(lambda x: x.value_counts().index[0])
        return df.fillna(most_freq)

    def _convert_cat_labels(self, df, pre_proc_dict, preproc):
        """
        Converts columns with factors into integer representation
        
        Parameters
        ----------

        df: Pandas.DataFrame
        
        pre_proc_dict: dict
            dictionary containing preprocessing objects

        preproc: str
            Filename of a previously trained preprocessor. Used here as an indicator, if None then preprocessing is being done for scratch.
        
        Returns
        -------
        
        Pandas.DataFrame
            DataFrame with eligible columns converted into 0 to nclass-1 integer representation

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
        Converts columns with factors into OneHotEncode integer representation
        
        Parameters
        ----------

        df: Pandas.DataFrame
        
        pre_proc_dict: dict
            dictionary containing preprocessing objects

        preproc: str
            Filename of a previously trained preprocessor. Used here as an indicator, if None then preprocessing is being done for scratch.
        
        Returns
        -------
        
        Pandas.DataFrame
            DataFrame with eligible columns converted into OnehotEncoded integer representation

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

        Parameters
        ----------
        df: Pandas.DataFrame

        pre_proc_dict: dict
            dictionary containing preprocessing objects

        pre_proc_filename: str
            Filename used to save the preprocessor 
            
        text_mat: str, optional
            text column name
            Default None
            
        use_feat_select: bool, optional
            If True then feature selection using LASSO for non-text columns is applied. Otherwise no feature selection will be performed
            Default True
        
        alpha_space: Numpy.array, optional
            Testing space for alpha parameter of LASSO used in hyperparameter tuning
            Default np.linspace(0.01,0.02,20) 
            
        random_state_var: int, optional
            Random seed for train-test-split
            Default np.random.randint(10000)
        
        test_size_var: float, optional
            Ratio of test versus train split
            Default 0.3

        plot: bool, optional
            If True then feature selection is vizualized by plotting the feature coef after LASSO.
            Default True

        Returns
        -------
        None
            Important features are saved after feature selection. Train test splits are saved as attributes self.x_train, self.x_test, self.y_train, self.y_test. Preprocessor is saved as a pickled file using filename pre_proc_filename.
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
        
        Parameters
        ----------
        
        scaler_ch: int, optional
            Decides which scaler to use, 0 for MinMaxScaler, 1 for Normalizer, 2 for StandarScaler
            Default 0
            
        logreg_C: List(float), optional
            Hyperparameter space for C to be used in the Log Reg Classifier
            Default [0.8,1,1.2,1.4]
            
        knn_neigh: List(int) , optional
            Hyperparameter space for number of neighbors to be used in the KNN Classifier
            Default np.arange(3,16)
            
        svc_c: List(Float), optional
            Hyperparameter space for C to be used in the Support Vector Classifier
            Default [0.5,1,1.5,2,2.5,2.6]

        gb_max_depth: List(int), optional
            Hyperparameter space for max depth to be used in Gradient Boosted Classifier Trees
            Default [2,3,4,5]
        
        gb_n_est: List(int) , optional
            Hyperparameter space for number of estimators to be used in Gradient Boosted Classifier Trees
            Default [40,60,80,100]
        
        verbose: bool, optional
            Prints out summary of fits if True
            Default True
            
        save: bool, optional
            Switch for saving the trained models in an external data file
            Default True
            
        scoring: str, optional
            Scoring metric
            Default accuracy
        
        model_file: str, optional
            Filename for storing all the trained models
            Default Trained_shallow_models
        
        Returns
        -------
        Self
            Trained models are saved as attribute in self.model_dict
        
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
        return self
    
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
        
        Parameters
        ----------
           
        scaler_ch: int, optional
            Decides which scaler to use, 0 for MinMaxScaler, 1 for Normalizer, 2 for StandarScaler
            Default 0
            
        patience_val: int, optional
            Number of epochs to monitor before exiting training if no major changes in accuracy occurs
            Default 2
            
        validation_split_val: float, optional
            ratio of split of dataset for testing purposes
            Default .2
            
        epochs_val: int, optional
            Max number of epochs to train
            Default 20
            
        verbose: bool, optional
            Model training details will be printed out if True
            Default True
            
        save: bool, optional
            Switch for saving the trained models in an external data file
            Default True
            
        model_file: str, optional
            Filename for storing the trained model. Must be h5 extension 
            Default Trained_deep_model.h5
            
        Returns
        -------
        Self
            DNN model is saved as an attribute in self.model
        
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
        return self


    def load_model(self, type_model='shallow', filename='Trained_shallow_models.sav', 
                   clf='logreg'):
        """
        This function is used to load a previously saved trained model. 
        The model will have been saved in an external file.
        
        Parameters
        ----------
        
        type: str, optional
            'shallow' to load a trained shallow model 'deep' to load a trained deep model
            Default shallow
            
        filename: str, optional
            Name of the file with the saved model
            Default Trained_shallow_models.sav
            
        clf: str, optional
            Only used for retrieving shallow models, this is the label of the classifier -
            'logreg': Logistic Regression using the lbfgs solver
            'knnstep': K Nearest Neighbors
            'svcstep': Support Vector Classification model
            'gradbooststep': Gradient Boosted Classification Trees
            Default logreg
            
        Return
        ------
        Self
            Selected model is saved as attribute in self.model
            
        Raises
        ------
        IOError: If saved files cannot be loaded
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
        """
        Method used to generate predictions using selected model

        Parameters
        ----------
        val_mat : Numpy.Ndarray or Pandas.Dataframe
            Feature matrix (X values)
        clf : str, optional
            Used to load model ONLY if load method was NOT used previously. 
            Default None

        Returns
        -------
        Numpy.array
            binary prediction vector 
            
        Raises
        ------
        ValueError: Model was not loaded correctly.

        """
        
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
        """
        Method used to generate prediction probabilities using selected model

        Parameters
        ----------
        val_mat : Numpy.Ndarray or Pandas.Dataframe
            Feature matrix (X values)
        clf : str, optional
            Used to load model ONLY if load method was NOT used previously. 
            Default None

        Returns
        -------
        Numpy.array
            binary prediction probability matrix shape (N,2) 
            
        Raises
        ------
        ValueError: Model was not loaded correctly.

        """
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
        """
        Method used to generate prediction scores using selected model

        Parameters
        ----------
        x_test : Numpy.Ndarray or Pandas.Dataframe
            Feature matrix (X values)

        x_test : Numpy.Ndarray or Pandas.Series
            Target vector of Ground truths used to compare predictions with 
            
        clf : str, optional
            Used to load model ONLY if load method was NOT used previously. 
            Default None

        Returns
        -------
        Float
            Score of predictions. Same scoring metric specified during fitting will be used.
            
        Raises
        ------
        ValueError: Model was not loaded correctly.

        """
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
        