# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""

DataWig SimpleImputer:
Uses some simple default encoders and featurizers that usually yield decent imputation quality

"""
import glob
import inspect
import json
import os
import pickle
import shutil
from typing import List, Dict, Any, Callable

import mxnet as mx
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

from _hpo import _HPO
from column_encoders import BowEncoder, CategoricalEncoder, NumericalEncoder, TfIdfEncoder
from imputer import Imputer
from mxnet_input_symbols import BowFeaturizer, NumericalFeaturizer
from utils import logger, get_context, random_split, rand_string, flatten_dict, merge_dicts, set_stream_log_level
from imputer import Imputer
from iterators import INSTANCE_WEIGHT_COLUMN



class SimpleImputer:
    """

    SimpleImputer model based on n-grams of concatenated strings of input columns and concatenated
    numerical features, if provided.

    Given a data frame with string columns, a model is trained to predict observed values in label
    column using values observed in other columns.

    The model can then be used to impute missing values.

    :param input_columns: list of input column names (as strings)
    :param output_column: output column name (as string)
    :param output_path: path to store model and metrics
    :param num_hash_buckets: number of hash buckets used for the n-gram hashing vectorizer, only
                                used for non-numerical input columns, ignored otherwise
    :param num_labels: number of imputable values considered after, only used for non-numerical
                        input columns, ignored otherwise
    :param tokens: string, 'chars' or 'words' (default 'chars'), determines tokenization strategy
                for n-grams, only used for non-numerical input columns, ignored otherwise
    :param numeric_latent_dim: int, number of latent dimensions for hidden layer of NumericalFeaturizers;
                only used for numerical input columns, ignored otherwise
    :param numeric_hidden_layers: number of numeric hidden layers
    :param is_explainable: if this is True, a stateful tf-idf encoder is used that allows
                           explaining classes and single instances


    Example usage:


    from datawig.simple_imputer import SimpleImputer
    import pandas as pd

    fn_train = os.path.join(datawig_test_path, "resources", "shoes", "train.csv.gz")
    fn_test = os.path.join(datawig_test_path, "resources", "shoes", "test.csv.gz")

    df_train = pd.read_csv(training_data_files)
    df_test = pd.read_csv(testing_data_files)

    output_path = "imputer_model"

    # set up imputer model
    imputer = SimpleImputer( input_columns=['item_name', 'bullet_point'], output_column='brand')

    # train the imputer model
    imputer = imputer.fit(df_train)

    # obtain imputations
    imputations = imputer.predict(df_test)

    """

    def __init__(self,
                 input_columns: List[str],
                 output_column: str,
                 output_path: str = "",
                 num_hash_buckets: int = int(2 ** 15),
                 num_labels: int = 100,
                 tokens: str = 'chars',
                 numeric_latent_dim: int = 100,
                 numeric_hidden_layers: int = 1,
                 is_explainable: bool = False) -> None:

        for col in input_columns:
            if not isinstance(col, str):
                raise ValueError("SimpleImputer.input_columns must be str type, was {}".format(type(col)))

        if not isinstance(output_column, str):
            raise ValueError("SimpleImputer.output_column must be str type, was {}".format(type(output_column)))

        self.input_columns = input_columns
        self.output_column = output_column
        self.num_hash_buckets = num_hash_buckets
        self.num_labels = num_labels
        self.tokens = tokens
        self.numeric_latent_dim = numeric_latent_dim
        self.numeric_hidden_layers = numeric_hidden_layers
        self.output_path = output_path
        self.imputer = None
        self.hpo = None
        self.numeric_columns = []
        self.string_columns = []
        self.hpo = _HPO()
        self.is_explainable = is_explainable

    def check_data_types(self, data_frame: pd.DataFrame) -> None:
        """

        Checks whether a column contains string or numeric data

        :param data_frame:
        :return:
        """
        self.numeric_columns = [c for c in self.input_columns if is_numeric_dtype(data_frame[c])]
        self.string_columns = list(set(self.input_columns) - set(self.numeric_columns))
        self.output_type = 'numeric' if is_numeric_dtype(data_frame[self.output_column]) else 'string'

        logger.debug(
            "Assuming {} numeric input columns: {}".format(len(self.numeric_columns),
                                                           ", ".join(self.numeric_columns)))
        logger.debug("Assuming {} string input columns: {}".format(len(self.string_columns),
                                                                  ", ".join(self.string_columns)))

    ################
    # num_epoch 변경 부분
    #################
    def fit(self,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame = None,
            ctx: mx.context = get_context(),
            learning_rate: float = 4e-3,
            num_epochs: int = 10,
            patience: int = 5,
            test_split: float = .1,
            weight_decay: float = 0.,
            batch_size: int = 16,
            final_fc_hidden_units: List[int] = None,
            calibrate: bool = True,
            class_weights: dict = None,
            instance_weights: list = None) -> Any:
        """

        Trains and stores imputer model

        :param train_df: training data as dataframe
        :param test_df: test data as dataframe; if not provided, a ratio of test_split of the
                            training data are used as test data
        :param ctx: List of mxnet contexts (if no gpu's available, defaults to [mx.cpu()])
                    User can also pass in a list gpus to be used, ex. [mx.gpu(0), mx.gpu(2), mx.gpu(4)]
        :param learning_rate: learning rate for stochastic gradient descent (default 4e-4)
        :param num_epochs: maximal number of training epochs (default 10)
        :param patience: used for early stopping; after [patience] epochs with no improvement,
                            training is stopped. (default 3)
        :param test_split: if no test_df is provided this is the ratio of test data to be held
                            separate for determining model convergence
        :param weight_decay: regularizer (default 0)
        :param batch_size (default 16)
        :param final_fc_hidden_units: list dimensions for FC layers after the final concatenation
        :param calibrate: Control automatic model calibration
        :param class_weights: Dictionary with labels as keys and weights as values.
                              Weighs each instance's contribution to the likelihood based on the corresponding class.
        :param instance_weights: List of weights for each instance in train_df.
        """

        # add weights to training data if provided
        train_df = self.__add_weights_to_df(train_df, class_weights, instance_weights, in_place=False)

        self.check_data_types(train_df)

        data_encoders = []
        data_columns = []

        if len(self.string_columns) > 0:
            string_feature_column = "ngram_features-" + rand_string(10)
            if self.is_explainable:
                data_encoders += [TfIdfEncoder(input_columns=self.string_columns,
                                               output_column=string_feature_column,
                                               max_tokens=self.num_hash_buckets,
                                               tokens=self.tokens)]
            else:
                data_encoders += [BowEncoder(input_columns=self.string_columns,
                                             output_column=string_feature_column,
                                             max_tokens=self.num_hash_buckets,
                                             tokens=self.tokens)]
            data_columns += [
                BowFeaturizer(field_name=string_feature_column, max_tokens=self.num_hash_buckets)]

        if len(self.numeric_columns) > 0:
            numerical_feature_column = "numerical_features-" + rand_string(10)
            # print(" === numerical_feature_column === ", numerical_feature_column)
            data_encoders += [NumericalEncoder(input_columns=self.numeric_columns,
                                               output_column=numerical_feature_column)]
            # print(" === data_encoders == ", data_encoders)

            data_columns += [
                NumericalFeaturizer(field_name=numerical_feature_column, numeric_latent_dim=self.numeric_latent_dim,
                                    numeric_hidden_layers=self.numeric_hidden_layers)]
            # print(" === data_columns == ", data_columns)

        if is_numeric_dtype(train_df[self.output_column]):
            label_column = [NumericalEncoder(self.output_column, normalize=True)]
            logger.debug("Assuming numeric output column: {}".format(self.output_column))
        else:
            label_column = [CategoricalEncoder(self.output_column, max_tokens=self.num_labels)]
            logger.debug("Assuming categorical output column: {}".format(self.output_column))

        # to make consecutive calls to .fit() continue where the previous call finished
        if self.imputer is None:
            self.imputer = Imputer(data_encoders=data_encoders,
                                   data_featurizers=data_columns,
                                   label_encoders=label_column,
                                   output_path=self.output_path)

        self.output_path = self.imputer.output_path

        self.imputer = self.imputer.fit(train_df, test_df, ctx, learning_rate, num_epochs, patience,
                                        test_split,
                                        weight_decay, batch_size,
                                        final_fc_hidden_units=final_fc_hidden_units,
                                        calibrate=calibrate)
        self.save()

        return self

    def predict(self,
                data_frame: pd.DataFrame,
                precision_threshold: float = 0.0,
                imputation_suffix: str = "_imputed",
                score_suffix: str = "_imputed_proba",
                inplace: bool = False):

        """
        Imputes most likely value if it is above a certain precision threshold determined on the
            validation set
        Precision is calculated as part of the `datawig.evaluate_and_persist_metrics` function.

        Returns original dataframe with imputations and respective likelihoods as estimated by
        imputation model; in additional columns; names of imputation columns are that of the label
        suffixed with `imputation_suffix`, names of respective likelihood columns are suffixed
        with `score_suffix`

        :param data_frame:   data frame (pandas)
        :param precision_threshold: double between 0 and 1 indicating precision threshold
        :param imputation_suffix: suffix for imputation columns
        :param score_suffix: suffix for imputation score columns
        :param inplace: add column with imputed values and column with confidence scores to data_frame, returns the
            modified object (True). Create copy of data_frame with additional columns, leave input unmodified (False).
        :return: data_frame original dataframe with imputations and likelihood in additional column
        """
        imputations = self.imputer.predict(data_frame, precision_threshold, imputation_suffix,
                                           score_suffix, inplace=inplace)

        return imputations


    def save(self):
        """

        Saves model to disk; mxnet module and imputer are stored separately

        """
        self.imputer.save()
        simple_imputer_params = {k: v for k, v in self.__dict__.items() if k not in ['imputer']}
        pickle.dump(simple_imputer_params,
                    open(os.path.join(self.output_path, "simple_imputer.pickle"), "wb"))



    def __add_weights_to_df(self,
                            df: pd.DataFrame,
                            class_weights: dict = None,
                            instance_weights: list = None,
                            in_place: bool = True) -> pd.DataFrame:
        """
        Add additional column to data frame inplace, with entries provided either
        (1) as dict values in class_dictionary with rows selected on dict keys in weights dictionary
            in correspondence to label_column.
        (2) or as list with weights for each instance

        :param df: input data
        :param class_weights: dictionary with label names and corresponding weights
        :param instance_weights: list of weights for each instance
        :param label_column: name of label column
        :param in_place: If true, append column to df, otherwise create copy.
        """
       
        # Nothing to do if no weights have been passed.
        if class_weights is None and instance_weights is None:
            return df

        # Check that only one type of weights is provided
        assert (class_weights is None or instance_weights is None), \
            "Please provide class_weights XOR instance_weights."

        if in_place is False:
            df_new = df.copy()
        else:
            df_new = df

        if class_weights is not None:
            df_new[INSTANCE_WEIGHT_COLUMN] = 0.0
            for label, weight in class_weights.items():
                df_new.loc[df_new[self.output_column] == label, INSTANCE_WEIGHT_COLUMN] = weight

        elif instance_weights is not None:
            df_new[INSTANCE_WEIGHT_COLUMN] = instance_weights

        return df_new