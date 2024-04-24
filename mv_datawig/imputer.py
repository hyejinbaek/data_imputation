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

DataWig Imputer:
Imputes missing values in tables

"""

import glob
import inspect
import itertools
import os
import pickle
import time
from typing import Any, List, Tuple

import mxnet as mx
import numpy as np

import pandas as pd
from mxnet.callback import save_checkpoint
from sklearn.preprocessing import StandardScaler

import calibration
from column_encoders import (CategoricalEncoder, ColumnEncoder,
                              NumericalEncoder, TfIdfEncoder)
from evaluation import evaluate_and_persist_metrics
from iterators import ImputerIterDf, INSTANCE_WEIGHT_COLUMN
from mxnet_input_symbols import Featurizer, NumericalFeaturizer
from utils import (AccuracyMetric, ColumnOverwriteException,
                    LogMetricCallBack, MeanSymbol, get_context, logger,
                    merge_dicts, random_split, timing, log_formatter)
from logging import FileHandler


class Imputer:
    
    """

    Imputer model based on deep learning trained with MxNet

    Given a data frame with string columns, a model is trained to predict observed values in one
    or more column using values observed in other columns. The model can then be used to impute
    missing values.

    :param data_encoders: list of datawig.mxnet_input_symbol.ColumnEncoders,
                            output_column name must match field_name of data_featurizers
    :param data_featurizers: list of Featurizer;
    :param label_encoders: list of CategoricalEncoder or NumericalEncoder
    :param output_path: path to store model and metrics


    """

    def __init__(self,
                 data_encoders: List[ColumnEncoder],
                 data_featurizers: List[Featurizer],
                 label_encoders: List[ColumnEncoder],
                 output_path="") -> None:
        
        self.ctx = None
        self.module = None
        self.data_encoders = data_encoders
        

        self.batch_size = 16

        self.data_featurizers = data_featurizers
        self.label_encoders = label_encoders

        self.final_fc_hidden_units = []

        self.train_losses = None
        self.test_losses = None

        self.training_time = 0.
        self.calibration_temperature = None

        self.precision_recall_curves = {}
        self.calibration_info = {}

        self.__class_patterns = None
        # explainability only works for Categorical and Tfidf inputs with a single categorical output column
        self.is_explainable = np.any([isinstance(encoder, CategoricalEncoder) or isinstance(encoder, TfIdfEncoder)
                                      for encoder in self.data_encoders]) and \
                              (len(self.label_encoders) == 1) and \
                              (isinstance(self.label_encoders[0], CategoricalEncoder))

        if len(self.data_featurizers) != len(self.data_encoders):
            raise ValueError("Argument Number of data_featurizers ({}) \
                              must match number of data_encoders ({})".format(len(self.data_encoders), len(self.data_featurizers)))

        for encoder in self.data_encoders: # len(data_encoders) = 13
            # NumericalEncoder
            encoder_type = type(encoder)
            if not issubclass(encoder_type, ColumnEncoder):
                raise ValueError("Arguments passed as data_encoder must be valid " +
                                 "datawig.column_encoders.ColumnEncoder, was {}".format(
                                     encoder_type))

        for encoder in self.label_encoders:
            encoder_type = type(encoder)
            if encoder_type not in [CategoricalEncoder, NumericalEncoder]:
                raise ValueError("Arguments passed as label_columns must be \
                                 datawig.column_encoders.CategoricalEncoder or NumericalEncoder, \
                                 was {}".format(encoder_type))
       

        encoder_outputs = [encoder.output_column for encoder in self.data_encoders]

        for featurizer in self.data_featurizers:
            featurizer_type = type(featurizer)
            if not issubclass(featurizer_type, Featurizer):
                raise ValueError("Arguments passed as data_featurizers must be valid \
                                 datawig.mxnet_input_symbols.Featurizer type, \
                                 was {}".format(featurizer_type))

            if featurizer.field_name not in encoder_outputs:
                raise ValueError(
                    "List of encoder outputs [{}] does not contain featurizer input for {}".format(
                        ", ".join(encoder_outputs), featurizer_type))
            # TODO: check whether encoder type matches requirements of featurizer

        # collect names of data and label columns
        input_col_names = [c.field_name for c in self.data_featurizers]
        label_col_names = list(itertools.chain(*[c.input_columns for c in self.label_encoders]))

        if len(set(input_col_names).intersection(set(label_col_names))) != 0:
            raise ValueError("cannot train with label columns that are in the input")

        # if there is no output directory provided, try to write to current dir
        if (output_path == '') or (not output_path):
            output_path = '.'

        self.output_path = output_path

        # if there was no output dir provided, name it to the label (-list) fitted
        if self.output_path == ".":
            label_names = [c.output_column.lower().replace(" ", "_") for c in self.label_encoders]
            self.output_path = "-".join(label_names)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.__attach_log_filehandler(filename=os.path.join(self.output_path, 'imputer.log'))

        self.module_path = os.path.join(self.output_path, "model")

        self.metrics_path = os.path.join(self.output_path, "fit-test-metrics.json")

    def __attach_log_filehandler(self, filename: str, level: str = "INFO") -> None:

        """
        Modifies global logger object and attaches filehandler

        :param filename: path to logfile
        :param level: logging level

        """

        if os.access(os.path.dirname(filename), os.W_OK):
            file_handler = FileHandler(filename, mode='a')
            file_handler.setLevel(level)
            file_handler.setFormatter(log_formatter)
            logger.addHandler(file_handler)
        else:
            logger.warning("Could not attach file log handler, {} is not writable.".format(filename))

    def __close_filehandlers(self) -> None:
        """Function to close connection with log file."""

        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)

    def __check_data(self, data_frame: pd.DataFrame) -> None:
        """
        Checks some aspects of data quality, currently just the label distribution

        Currently checked are:
         - label overlap in training and test data

        if these requirements are not met a warning is raised.

        TODO: more data quality checks; note that in order to avoid unneccessary passes through
        data, some warnings are raised in CategoricalEncoder.fit, too.

        """
        for col_enc in self.label_encoders:

            if not col_enc.is_fitted():
                logger.warning(
                    "Data encoder {} for columns {} is not fitted yet, cannot check data".format(
                        type(col_enc), ", ".join(col_enc.input_columns)))
            elif isinstance(col_enc, CategoricalEncoder):
                values_not_in_test_set = set(col_enc.token_to_idx.keys()) - \
                                         set(data_frame[col_enc.input_columns[0]].unique())
                if len(values_not_in_test_set) > 0:
                    logger.warning(
                        "Test set does not contain any ocurrences of values [{}] in column [{}], "
                        "consider using a more representative test set.".format(
                            ", ".join(values_not_in_test_set),
                            col_enc.input_columns[0]))

    def fit(self,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame = None,
            ctx: mx.context = get_context(),
            learning_rate: float = 1e-3,
            num_epochs: int = 100,
            patience: int = 3,
            test_split: float = .1,
            weight_decay: float = 0.,
            batch_size: int = 16,
            final_fc_hidden_units: List[int] = None,
            calibrate: bool = True):
        """
        Trains and stores imputer model

        :param train_df: training data as dataframe
        :param test_df: test data as dataframe; if not provided, [test_split] % of the training
                        data are used as test data
        :param ctx: List of mxnet contexts (if no gpu's available, defaults to [mx.cpu()])
                    User can also pass in a list gpus to be used, ex. [mx.gpu(0), mx.gpu(2), mx.gpu(4)]
        :param learning_rate: learning rate for stochastic gradient descent (default 1e-4)
        :param num_epochs: maximal number of training epochs (default 100)
        :param patience: used for early stopping; after [patience] epochs with no improvement,
                        training is stopped. (default 3)
        :param test_split: if no test_df is provided this is the ratio of test data to be held
                        separate for determining model convergence
        :param weight_decay: regularizer (default 0)
        :param batch_size: default 16
        :param final_fc_hidden_units: list of dimensions for the final fully connected layer.
        :param calibrate: whether to calibrate predictions
        :return: trained imputer model
        """
        if final_fc_hidden_units is None:
            final_fc_hidden_units = []

        # make sure the output directory is writable
        assert os.access(self.output_path, os.W_OK), "Cannot write to directory {}".format(
            self.output_path)

        self.batch_size = batch_size
        self.final_fc_hidden_units = final_fc_hidden_units

        self.ctx = ctx
        logger.debug('Using [{}] as the context for training'.format(ctx))

        if (train_df is None) or (not isinstance(train_df, pd.core.frame.DataFrame)):
            raise ValueError("Need a non-empty DataFrame for fitting Imputer model")

        if test_df is None:
            train_df, test_df = random_split(train_df, [1.0 - test_split, test_split])

        iter_train, iter_test = self.__build_iterators(train_df, test_df, test_split)

        self.__check_data(test_df)

        # to make consecutive calls to .fit() continue where the previous call finished
        if self.module is None:
            self.module = self.__build_module(iter_train)

        self.__fit_module(iter_train, iter_test, learning_rate, num_epochs, patience, weight_decay)

        # Check whether calibration is needed, if so ompute and set internal parameter
        # for temperature scaling that is supplied to self.__predict_mxnet_iter()
        if calibrate is True:
            self.calibrate(iter_test)

        _, metrics = self.__transform_and_compute_metrics_mxnet_iter(iter_test,
                                                                     metrics_path=self.metrics_path)

        for att, att_metric in metrics.items():
            if isinstance(att_metric, dict) and ('precision_recall_curves' in att_metric):
                self.precision_recall_curves[att] = att_metric['precision_recall_curves']

        self.__prune_models()
        self.save()

        if self.is_explainable:
            self.__persist_class_prototypes(iter_train, train_df)

        self.__close_filehandlers()

        return self


    def __fit_module(self,
                     iter_train: ImputerIterDf,
                     iter_test: ImputerIterDf,
                     learning_rate: float,
                     num_epochs: int,
                     patience: int,
                     weight_decay: float) -> None:
        """

        Trains the mxnet module

        :param learning_rate: learning rate of sgd
        :param num_epochs: maximal number of training epochs
        :param patience: early stopping
        :param weight_decay: regularizer

        """
        metric_name = 'cross-entropy'

        train_cb = LogMetricCallBack([metric_name])
        test_cb = LogMetricCallBack([metric_name], patience=patience)

        def checkpoint(epoch, sym, arg, aux):
            save_checkpoint(self.module_path, epoch, sym, arg, aux)

        start = time.time()
        cross_entropy_metric = MeanSymbol(metric_name)
        accuracy_metrics = [
            AccuracyMetric(name=label_col.output_column, label_index=i) for i, label_col in
            enumerate(self.label_encoders)
        ]
        combined_metric = mx.metric.CompositeEvalMetric(
            metrics=[cross_entropy_metric] + accuracy_metrics)

        with timing("fit model"):
            try:
                self.module.fit(
                    train_data=iter_train,
                    eval_data=iter_test,
                    eval_metric=combined_metric,
                    num_epoch=num_epochs,
                    initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
                    optimizer='adam',
                    optimizer_params=(('learning_rate', learning_rate), ('wd', weight_decay)),
                    batch_end_callback=[mx.callback.Speedometer(iter_train.batch_size,
                                                                int(np.ceil(
                                                                    iter_train.df_iterator.data[0][1].shape[0] /
                                                                    iter_train.batch_size / 2)),
                                                                auto_reset=True)],
                    eval_end_callback=[test_cb, train_cb],
                    epoch_end_callback=checkpoint
                )
            except StopIteration:
                # catch the StopIteration exception thrown when early stopping condition is reached
                # this is ugly but the only way to use module api and have early stopping
                logger.debug("Stopping training, patience reached")
                pass

        self.training_time = time.time() - start
        self.train_losses, self.test_losses = train_cb.metrics[metric_name], test_cb.metrics[
            metric_name]

    def __build_module(self, iter_train: ImputerIterDf) -> mx.mod.Module:
        mod = _MXNetModule(self.ctx, self.label_encoders, self.data_featurizers, self.final_fc_hidden_units)
        return mod(iter_train)

    def __build_iterators(self,
                          train_df: pd.DataFrame,
                          test_df: pd.DataFrame,
                          test_split: float) -> Tuple[ImputerIterDf, ImputerIterDf]:
        """

        Builds iterators from data frame

        :param train_df: training data as pandas DataFrame
        :param test_df: test data, can be None
        :param test_split: test data split, used if test_df is None
        :return: train and test iterators for mxnet model

        """

        # if this is a pandas df
        if not isinstance(train_df, pd.core.frame.DataFrame):
            raise ValueError("Only pandas data frames are supported")

        if (test_df is not None) & (not isinstance(test_df, pd.core.frame.DataFrame)):
            raise ValueError("Only pandas data frames are supported")

        input_col_names = [c.field_name for c in self.data_featurizers]
        label_col_names = list(itertools.chain(*[c.input_columns for c in self.label_encoders]))

        missing_columns = set(input_col_names + label_col_names) - set(train_df.columns)

        if len(missing_columns) > 0:
            ValueError("Training DataFrame does not contain required column {}".format(missing_columns))

        for encoder in self.label_encoders:
            if not encoder.is_fitted():
                encoder_type = type(encoder)
                logger.debug("Fitting label encoder {} on {} rows \
                            of training data".format(encoder_type, len(train_df)))
                encoder.fit(train_df)

        # discard all rows which contain labels that will not be learned/predicted
        train_df = self.__drop_missing_labels(train_df, how='all')

        # if there is no test data set provided, split one off the training data
        if test_df is None:
            train_df, test_df = random_split(train_df, [1.0 - test_split, test_split])

        test_df = self.__drop_missing_labels(test_df, how='all')

        logger.debug("Train: {}, Test: {}".format(len(train_df), len(test_df)))

        for encoder in self.data_encoders:
            if not encoder.is_fitted():
                encoder_type = type(encoder)
                logger.debug(
                    "Fitting data encoder {} on columns {} and {} rows of training data with parameters {}".format(
                        encoder_type, ", ".join(encoder.input_columns), len(train_df), encoder.__dict__))

                encoder.fit(train_df)

        logger.debug("Building Train Iterator with {} elements".format(len(train_df)))
        iter_train = ImputerIterDf(
            data_frame=train_df,
            data_columns=self.data_encoders,
            label_columns=self.label_encoders,
            batch_size=self.batch_size
        )

        logger.debug("Building Test Iterator with {} elements".format(len(test_df)))
        iter_test = ImputerIterDf(
            data_frame=test_df,
            data_columns=iter_train.data_columns,
            label_columns=iter_train.label_columns,
            batch_size=self.batch_size
        )

        return iter_train, iter_test

    def __transform_mxnet_iter(self, mxnet_iter: ImputerIterDf) -> dict:
        """
        Imputes values given an mxnet iterator (see iterators)

        :param mxnet_iter:  iterator, see ImputerIter in iterators.py
        :return: dict of {'column_name': list} where list contains the string predictions

        """
        labels, model_outputs = zip(*self.__predict_mxnet_iter(mxnet_iter).items())
        predictions = {}
        for col_enc, label, model_output in zip(mxnet_iter.label_columns, labels, model_outputs):
            if isinstance(col_enc, CategoricalEncoder):
                predictions[label] = col_enc.decode(pd.Series(model_output.argmax(axis=1)))
            elif isinstance(col_enc, NumericalEncoder):
                predictions[label] = model_output
        return predictions



    def __predict_above_precision_mxnet_iter(self,
                                             mxnet_iter: ImputerIterDf,
                                             precision_threshold: float = 0.95) -> dict:
        """
        Imputes values only if predictions are above certain precision threshold,
            determined on test set during fit

        :param mxnet_iter: iterator, see ImputerIter in iterators.py
        :param precision_threshold: don't predict if predicted class probability is below
            this precision threshold
        :return: dict of {'column_name': array}, array is a numpy array of shape samples-by-labels

        """
        predictions = self.__predict_top_k_mxnet_iter(mxnet_iter, top_k=1)
        for col_enc, att in zip(self.label_encoders, predictions.keys()):
            if isinstance(col_enc, CategoricalEncoder):
                predictions[att] = self.__filter_predictions(predictions[att], precision_threshold)
            else:
                logger.debug("Precision filtering only for CategoricalEncoder returning \
                            {} unfiltered".format(att))
                predictions[att] = predictions[att]

        return predictions

    ###############################################
    ####### predict numerical data ################
    ###############################################
    def __predict_mxnet_iter(self, mxnet_iter):
        """
        Returns the probabilities for each class
        :param mxnet_iter:  iterator, see ImputerIter in iterators.py
        :return: dict of {'column_name': array}, array is a numpy array of shape samples-by-labels
        """
        if not self.module.for_training:
            self.module.bind(data_shapes=mxnet_iter.provide_data,
                             label_shapes=mxnet_iter.provide_label)
        # FIXME: truncation to [:mxnet_iter.start_padding_idx] because of having to set last_batch_handle to discard.
        # truncating bottom rows for each output module while preserving all the columns
        mod_output = [o.asnumpy()[:mxnet_iter.start_padding_idx, :] for o in
                      self.module.predict(mxnet_iter)[1:]]
        output = {}
        
        for label_encoder, pred in zip(mxnet_iter.label_columns, mod_output):
            # print(" === pred === ", pred)
            if isinstance(label_encoder, NumericalEncoder):
                output[label_encoder.output_column] = label_encoder.decode(pred)
            else:
                # apply temperature scaling calibration if a temperature was fit.
                if self.calibration_temperature is not None:
                    pred = calibration.calibrate(pred, self.calibration_temperature)
                output[label_encoder.output_column] = pred
        return output

    def __predict_top_k_mxnet_iter(self, mxnet_iter, top_k=5):
        """
        For categorical outputs, returns tuples of (label, probability) for the top_k most likely
        predicted classes
        For numerical outputs, returns just prediction

        :param mxnet_iter:  iterator, see ImputerIter in iterators.py
        :return: dict of {'column_name': list} where list is a list of (label, probability)
                    tuples or numerical values

        """
        
        col_enc_att_probas = zip(mxnet_iter.label_columns,
                                 *zip(*self.__predict_mxnet_iter(mxnet_iter).items()))
        
        top_k_predictions = {}
        for col_enc, att, probas in col_enc_att_probas:
            
            if isinstance(col_enc, CategoricalEncoder):
                top_k_predictions[att] = []
                for pred in probas:
                    top_k_pred_idx = pred.argsort()[-top_k:][::-1]
                    label_proba_tuples = [(col_enc.decode_token(idx), pred[idx]) for idx in
                                          top_k_pred_idx]
                    top_k_predictions[att].append(label_proba_tuples)
                    
            else:
                logger.debug(
                    "Top-k only for CategoricalEncoder, dropping {}, {}".format(att, type(col_enc)))
                top_k_predictions[att] = probas

        return top_k_predictions

    def __transform_and_compute_metrics_mxnet_iter(self,
                                                   mxnet_iter: ImputerIterDf,
                                                   metrics_path: str = None) -> Tuple[dict, dict]:
        """

        Returns predictions and metrics (average and per class)

        :param mxnet_iter:
        :param metrics_path: if not None and exists, metrics are serialized as json to this path.
        :return: predictions and metrics

        """
        # get thresholded predictions for predictions and standard metrics
        mxnet_iter.reset()
        all_predictions = self.__transform_mxnet_iter(mxnet_iter)
        # reset iterator, compute probabilistic outputs for all classes for precision/recall curves
        mxnet_iter.reset()
        all_predictions_proba = self.__predict_mxnet_iter(mxnet_iter)
        mxnet_iter.reset()
        true_labels_idx_array = mx.nd.concat(
            *[mx.nd.concat(*[l for l in b.label], dim=1) for b in mxnet_iter],
            dim=0
        )
        true_labels_string = {}
        true_labels_idx = {}
        predictions_categorical = {}
        predictions_categorical_proba = {}

        predictions_numerical = {}
        true_labels_numerical = {}

        for l_idx, col_enc in enumerate(mxnet_iter.label_columns):
             # pylint: disable=invalid-sequence-index
            n_predictions = len(all_predictions[col_enc.output_column])
            if isinstance(col_enc, CategoricalEncoder):
                predictions_categorical[col_enc.output_column] = all_predictions[
                    col_enc.output_column]
                predictions_categorical_proba[col_enc.output_column] = all_predictions_proba[
                    col_enc.output_column]
                attribute = col_enc.output_column
                true_labels_idx[attribute] = true_labels_idx_array[:n_predictions, l_idx].asnumpy()
                true_labels_string[attribute] = [col_enc.decode_token(idx) for idx in
                                                 true_labels_idx[attribute]]
            elif isinstance(col_enc, NumericalEncoder):
                true_labels_numerical[col_enc.output_column] = \
                    true_labels_idx_array[:n_predictions, l_idx].asnumpy()
                predictions_numerical[col_enc.output_column] = \
                    all_predictions[col_enc.output_column]

        metrics = evaluate_and_persist_metrics(true_labels_string,
                                               true_labels_idx,
                                               predictions_categorical,
                                               predictions_categorical_proba,
                                               metrics_path,
                                               None,
                                               true_labels_numerical,
                                               predictions_numerical)

        return merge_dicts(predictions_categorical, predictions_numerical), metrics


    def predict(self,
                data_frame: pd.DataFrame,
                precision_threshold: float = 0.0,
                imputation_suffix: str = "_imputed",
                score_suffix: str = "_imputed_proba",
                inplace: bool = False) -> pd.DataFrame:
        """
        Computes imputations for numerical or categorical values

        For categorical imputations, most likely values are imputed if values are above a certain
        precision threshold computed on the validation set
        Precision is calculated as part of the `datawig.evaluate_and_persist_metrics` function.

        For numerical imputations, no thresholding is applied.

        Returns original dataframe with imputations and respective likelihoods as estimated by
        imputation model in additional columns; names of imputation columns are that of the label
        suffixed with `imputation_suffix`, names of respective likelihood columns are suffixed with
        `score_suffix`

        :param data_frame:   pandas data_frame
        :param precision_threshold: double between 0 and 1 indicating precision threshold for each
                                    imputation
        :param imputation_suffix: suffix for imputation columns
        :param score_suffix: suffix for imputation score columns
        :param inplace: add column with imputed values and column with confidence scores to data_frame, returns the
            modified object (True). Create copy of data_frame with additional columns, leave input unmodified (False).
        :return: dataframe with imputations and their likelihoods in additional columns
        """

        if not inplace:
            data_frame = data_frame.copy()

        numerical_outputs = list(
            itertools.chain(
                *[c.input_columns for c in self.label_encoders if isinstance(c, NumericalEncoder)]))
        # print(" ==================================================== ")
        # print(" === numerical_outputs == ", numerical_outputs)
        # print(" ==================================================== ")

        predictions = self.predict_above_precision(data_frame, precision_threshold).items()
        ###############################
        ###### imputation_suffix ######
        ###############################
        for label, imputations in predictions:
            imputation_col = label + imputation_suffix
            if imputation_col in data_frame.columns:
                raise ColumnOverwriteException(
                    "DataFrame contains column {}; remove column and try again".format(
                        imputation_col))

            if label not in numerical_outputs:
                imputation_proba_col = label + score_suffix
                if imputation_proba_col in data_frame.columns:
                    raise ColumnOverwriteException(
                        "DataFrame contains column {}; remove column and try again".format(
                            imputation_proba_col))

                imputed_values, imputed_value_scores = [], []
                for imputation_above_precision_threshold in imputations:
                    if not imputation_above_precision_threshold:
                        imputed_value, imputed_value_score = "", np.nan
                    else:
                        imputed_value, imputed_value_score = imputation_above_precision_threshold
                    imputed_values.append(imputed_value)
                    imputed_value_scores.append(imputed_value_score)

                data_frame[imputation_col] = imputed_values
                data_frame[imputation_proba_col] = imputed_value_scores

            elif label in numerical_outputs:
                data_frame[imputation_col] = imputations

        return data_frame

    def predict_above_precision(self, data_frame: pd.DataFrame, precision_threshold=0.95) -> dict:
        """
        Returns the probabilities for each class, filtering out predictions below the precision threshold.

        :param data_frame:  data frame
        :param precision_threshold: don't predict if predicted class probability is below this
                                        precision threshold
        :return: dict of {'column_name': array}, array is a numpy array of shape samples-by-labels

        """
        mxnet_iter = self.__mxnet_iter_from_df(data_frame)
        # print(" ==================================================== ")
        # print(" === predict_above_precision mxnet_iter == ", mxnet_iter)
        # print(" ==================================================== ")
        return self.__predict_above_precision_mxnet_iter(mxnet_iter,
                                                         precision_threshold=precision_threshold)


    def __drop_missing_labels(self, data_frame: pd.DataFrame, how='all') -> pd.DataFrame:
        """

        Drops rows of data frame that contain missing labels

        :param data_frame: pandas data frame
        :param how: ['all', 'any'] whether to drop rows if all labels are missing or if just
                    any label is missing
        :return: pandas DataFrame

        """
        n_samples = len(data_frame)
        missing_idx = -1
        for col_enc in self.label_encoders:
            if isinstance(col_enc, CategoricalEncoder):
                # for CategoricalEncoders, exclude rows that are either nan or not in the
                # token_to_idx mapping
                col_missing_idx = data_frame[col_enc.input_columns[0]].isna() | \
                                  ~data_frame[col_enc.input_columns[0]].isin(
                                      col_enc.token_to_idx.keys())
            elif isinstance(col_enc, NumericalEncoder):
                # for NumericalEncoders, exclude rows that are nan
                col_missing_idx = data_frame[col_enc.input_columns[0]].isna()

            logger.debug("Detected {} rows with missing labels \
                        for column {}".format(col_missing_idx.sum(), col_enc.input_columns[0]))

            if missing_idx == -1:
                missing_idx = col_missing_idx
            elif how == 'all':
                missing_idx = missing_idx & col_missing_idx
            elif how == 'any':
                missing_idx = missing_idx | col_missing_idx

        logger.debug("Dropping {}/{} rows".format(missing_idx.sum(), n_samples))

        return data_frame.loc[~missing_idx, :]

    def __prune_models(self):
        """

        Removes all suboptimal models from output directory

        """
        best_model = glob.glob(self.module_path + "*{}.params".format(self.__get_best_epoch()))
        logger.debug("Keeping {}".format(best_model[0]))
        worse_models = set(glob.glob(self.module_path + "*.params")) - set(best_model)
        # remove worse models
        for worse_epoch in worse_models:
            logger.debug("Deleting {}".format(worse_epoch))
            os.remove(worse_epoch)

    def __get_best_epoch(self):
        """

        Retrieves the best epoch, i.e. the minimum of the test_losses

        :return: best epoch
        """
        return sorted(enumerate(self.test_losses), key=lambda x: x[1])[0][0]

    def save(self):
        """

        Saves model to disk, except mxnet module which is stored separately during fit

        """
        # save all params but the mxnet module
        params = {k: v for k, v in self.__dict__.items() if k != 'module'}
        pickle.dump(params, open(os.path.join(self.output_path, "imputer.pickle"), "wb"))


    def __mxnet_iter_from_df(self, data_frame: pd.DataFrame) -> ImputerIterDf:
        """

        Transforms dataframe into imputer iterator for mxnet

        :param data_frame: pandas DataFrame
        :return: ImputerIterDf
        """
        return ImputerIterDf(
            data_frame=data_frame,
            data_columns=self.data_encoders,
            label_columns=self.label_encoders,
            batch_size=self.batch_size
        )

    def calibrate(self, test_iter: ImputerIterDf):
        """
        Cecks model calibration and fits temperature scaling.
        If the fit improves model calibration, the temperature parameter is assigned
        as property to self and used for all further predictions in self.predict_mxnet_iter().
        Saves calibration information to dictionary.

        :param test_iter: iterator, see ImputerIter in iterators.py
        :return: None
        """

        test_iter.reset()
        proba = self.__predict_mxnet_iter(test_iter)

        test_iter.reset()
        labels = mx.nd.concat(*[mx.nd.concat(*[l for l in b.label], dim=1) for b in test_iter], dim=0)

        if len(test_iter.label_columns) != 1:
            logger.warning('Aborting calibration. Can only calibrate one output column.')
            return

        output_label = test_iter.label_columns[0].output_column
        n_labels = proba[output_label].shape[0]

        scores = proba[output_label]
        labels = labels.asnumpy().squeeze()[:n_labels]

        ece_pre = calibration.compute_ece(scores, labels)
        self.calibration_info['ece_pre'] = ece_pre
        self.calibration_info['reliability_pre'] = calibration.reliability(scores, labels)
        logger.debug('Expected calibration error: {:.1f}%'.format(100*ece_pre))

        temperature = calibration.fit_temperature(scores, labels)
        ece_post = calibration.compute_ece(scores, labels, temperature)
        self.calibration_info['ece_post'] = ece_post
        logger.debug('Expected calibration error after calibration: {:.1f}%'.format(100*ece_post))

        # check whether calibration improves at all and apply
        if ece_pre - ece_post > 0:
            self.calibration_info['reliability_post'] = calibration.reliability(
                calibration.calibrate(scores, temperature), labels)
            self.calibration_info['ece_post'] = calibration.compute_ece(scores, labels, temperature)
            self.calibration_temperature = temperature


class _MXNetModule:
    def __init__(
            self,
            ctx: mx.context,
            label_encoders: List[ColumnEncoder],
            data_featurizers: List[Featurizer],
            final_fc_hidden_units: List[int]
    ):
        """
        Wrapper of internal DataWig MXNet module

        :param ctx: MXNet execution context
        :param label_encoders: list of label column encoders
        :param data_featurizers: list of data featurizers
        :param final_fc_hidden_units: list of number of hidden parameters
        """
        self.ctx = ctx
        self.data_featurizers = data_featurizers
        self.label_encoders = label_encoders
        self.final_fc_hidden_units = final_fc_hidden_units

    def __call__(self,
                 iter_train: ImputerIterDf) -> mx.mod.Module:
        """
        Given a training iterator, build MXNet module and return it

        :param iter_train: Training data iterator
        :return: mx.mod.Module
        """

        predictions, loss = self.__make_loss()

        logger.debug("Building output symbols")
        output_symbols = []
        for col_enc, output in zip(self.label_encoders, predictions):
            output_symbols.append(
                mx.sym.BlockGrad(output, name="pred-{}".format(col_enc.output_column)))

        mod = mx.mod.Module(
            mx.sym.Group([loss] + output_symbols),
            context=self.ctx,
            # [name for name, dim in iter_train.provide_data],
            data_names=[name for name, dim in iter_train.provide_data if name in loss.list_arguments()],
            label_names=[name for name, dim in iter_train.provide_label]
        )

        if mod.binded is False:
            mod.bind(data_shapes=[d for d in iter_train.provide_data if d.name in loss.list_arguments()],  # iter_train.provide_data,
                     label_shapes=iter_train.provide_label)

        return mod

    @staticmethod
    def __make_categorical_loss(latents: mx.symbol,
                                label_field_name: str,
                                num_labels: int,
                                final_fc_hidden_units: List[int] = None) -> Tuple[Any, Any]:
        """
        Generate output symbol for categorical loss

        :param latents: MxNet symbol containing the concantenated latents from all featurizers
        :param label_field_name: name of the label column
        :param num_labels: number of labels contained in the label column (for prediction)
        :param final_fc_hidden_units: list of dimensions for the final fully connected layer.
                                The length of this list corresponds to the number of FC
                                layers, and the contents of the list are integers with
                                corresponding hidden layer size.
        :return: mxnet symbols for predictions and loss
        """

        fully_connected = None
        if len(final_fc_hidden_units) == 0:
            # generate prediction symbol
            fully_connected = mx.sym.FullyConnected(
                data=latents,
                num_hidden=num_labels,
                name="label_{}".format(label_field_name))
        else:
            layer_size = final_fc_hidden_units
            with mx.name.Prefix("label_{}".format(label_field_name)):
                for i, layer in enumerate(layer_size):
                    if i == len(layer_size) - 1:
                        fully_connected = mx.sym.FullyConnected(
                            data=latents,
                            num_hidden=layer)
                    else:
                        latents = mx.sym.FullyConnected(
                            data=latents,
                            num_hidden=layer)

        instance_weight = mx.sym.Variable(INSTANCE_WEIGHT_COLUMN)
        pred = mx.sym.softmax(fully_connected)
        label = mx.sym.Variable(label_field_name)

        # assign to 0.0 the label values larger than number of classes so that they
        # do not contribute to the loss

        logger.debug("Building output of label {} with {} classes \
                     (including missing class)".format(label, num_labels))

        num_labels_vec = label * 0.0 + num_labels
        indices = mx.sym.broadcast_lesser(label, num_labels_vec)
        label = label * indices

        # goes from (batch, 1) to (batch,) as is required for softmax output
        label = mx.sym.split(label, axis=1, num_outputs=1, squeeze_axis=1)

        # mask entries when label is 0 (missing value)
        missing_labels = mx.sym.zeros_like(label)
        positive_mask = mx.sym.broadcast_greater(label, missing_labels)

        # compute the cross entropy only when labels are positive
        cross_entropy = mx.sym.pick(mx.sym.log_softmax(fully_connected), label) * -1 * positive_mask
        # multiply loss by class weighting
        cross_entropy = cross_entropy * mx.sym.pick(instance_weight, label)

        # normalize the cross entropy by the number of positive label
        num_positive_indices = mx.sym.sum(positive_mask)
        cross_entropy = mx.sym.broadcast_div(cross_entropy, num_positive_indices + 1.0)

        # todo because MakeLoss normalize even with normalization='null' argument is used,
        # we have to multiply by batch_size here
        batch_size = mx.sym.sum(mx.sym.ones_like(label))
        cross_entropy = mx.sym.broadcast_mul(cross_entropy, batch_size)

        return pred, cross_entropy

    @staticmethod
    def __make_numerical_loss(latents: mx.symbol,
                              label_field_name: str) -> Tuple[Any, Any]:
        """
        Generate output symbol for univariate numeric loss

        :param latents:
        :param label_field_name:
        :return: mxnet symbols for predictions and loss
        """

        # generate prediction symbol
        pred = mx.sym.FullyConnected(
            data=latents,
            num_hidden=1,
            name="label_{}".format(label_field_name))

        target = mx.sym.Variable(label_field_name)

        # squared loss
        loss = mx.sym.sum((pred - target) ** 2.0)

        return pred, loss

    def __make_loss(self, eps: float = 1e-5) -> Tuple[Any, Any]:

        logger.debug("Concatenating all {} latent symbols".format(len(self.data_featurizers)))

        unique_input_field_names = set([feat.field_name for feat in self.data_featurizers])
        if len(unique_input_field_names) < len(self.data_featurizers):
            raise ValueError("Input fields of Featurizers outputs of ColumnEncoders must be unique but \
                there were duplicates in {}, consider \
                explicitly providing output column names to ColumnEncoders".format(", ".join(unique_input_field_names)))

        # construct mxnet symbols for the data columns
        latents = mx.sym.concat(*[f.latent_symbol() for f in self.data_featurizers], dim=1)

        # build predictions and loss for each single output
        outputs = []
        for output_col in self.label_encoders:
            if isinstance(output_col, CategoricalEncoder):
                logger.debug("Constructing categorical loss for column {} and {} labels".format(
                    output_col.output_column, output_col.max_tokens))
                outputs.append(
                    self.__make_categorical_loss(
                        latents,
                        output_col.output_column,
                        output_col.max_tokens + 1,
                        self.final_fc_hidden_units
                    )
                )
            elif isinstance(output_col, NumericalEncoder):
                logger.debug(
                    "Constructing numerical loss for column {}".format(output_col.output_column))
                outputs.append(self.__make_numerical_loss(latents, output_col.output_column))

        predictions, losses = zip(*outputs)

        # compute mean loss for each output
        mean_batch_losses = [mx.sym.mean(l) + eps for l in losses]

        # normalize the loss contribution of each label by the mean over the batch
        normalized_losses = [mx.sym.broadcast_div(l, mean_loss) for l, mean_loss in zip(losses, mean_batch_losses)]

        # multiply the loss by the mean of all losses of all labels to preserve the gradient norm
        mean_label_batch_loss = mx.sym.ElementWiseSum(*mean_batch_losses) / float(len(mean_batch_losses))

        # normalize batch
        loss = mx.sym.broadcast_mul(
            mx.sym.ElementWiseSum(*normalized_losses) / float(len(mean_batch_losses)),
            mean_label_batch_loss
        )
        loss = mx.sym.MakeLoss(loss, normalization='valid', valid_thresh=1e-6)

        return predictions, loss