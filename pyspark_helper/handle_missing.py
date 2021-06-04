from pyspark_helper.common import get_column_mean, get_column_mode, get_column_median, generate_seq_idx, col_tolist
from pyspark_helper.exceptions import *
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import numpy as np
import pandas as pd
import random
import sys


def fill_timegap(sparksession, spark_df, ts_column_name, timestep):
    """
    Fill time gaps in time series dataframe based on a given timestep.

    :param sparksession: spark session object
    :param spark_df: spark dataframe
    :param ts_column_name: name of the column with timestamp dtype
    :param timestep: timestep to fill the time gap. E.g. 15 * 60 means 15 minutes

    :returns: filled timegap spark dataframe
    """
    # step = 15 * 60  # determine the timestep - in this case = 15 minutes

    minp, maxp = spark_df.select(
        F.min(ts_column_name).cast("long"), F.max(ts_column_name).cast("long")
    ).first()

    reference = sparksession.range(
        (minp / timestep) * timestep,
        ((maxp / timestep) + 1) * timestep,
        timestep).select(
        F.col("id").cast("timestamp").alias(ts_column_name))

    return reference.join(
        spark_df,
        [ts_column_name],
        "leftouter").orderBy(ts_column_name)

def fill_na(spark_df, column_name, fill_with=None, strategy='mean'):
    """
    Get the mode for the specified column. Default behaviour is to fill in the na values with mean values of the specified column.

    :param spark_df: spark dataframe
    :param column_name: name of the column to fill in the na values
    :param fill_with: value to replace the na values. Default is None
    :param strategy: 'mean', 'mode' or 'median'. Automatically set to None if user provides a value to `fill_with`. Default is 'mean'

    :returns: spark dataframe with the specified column that has the na values filled with mean/mode/median of the specified column or with a custom value
    """

    if fill_with is not None:  # if user provides custom value then set strategy to None and return filled df
        strategy = None
    else:
        if strategy == 'mean':
            fill_with = get_column_mean(spark_df, column_name)
        elif strategy == 'median':
            fill_with = get_column_median(spark_df, column_name)
        elif strategy == 'mode':
            fill_with = get_column_mode(spark_df, column_name)
        else:
            raise InvalidFillNAError(
                "Argument `fill_with` is None and `strategy` is not one of 'mean', 'mode' or 'median'")

    return spark_df.fillna({column_name: fill_with})


def fill_forward(spark_df, key_column, fill_column, id_column=None):
    """
    Fill the na values of the specified column with forward fill method.

    :param spark_df: spark dataframe
    :param key_column: name of the column to sort by
    :param fill_column: name of the column to fill in the na values
    :param id_column: name of the column to partition the spark dataframe. Default is None

    :returns: spark dataframe with the specified column that has the na values filled using forward fill method
    """

    if id_column is not None:  # if id_column is provided, partition by the id_column
        window_spec = Window.partitionBy(id_column) \
            .orderBy(key_column)                     \
            .rowsBetween(-sys.maxsize, 0)
    else:                  # else just do forward fill as is
        window_spec = Window.orderBy(key_column) \
            .rowsBetween(-sys.maxsize, 0)

    # Fill null's with last *non null* value in the window
    ff = spark_df.withColumn(
        'fill_fwd',
        F.last(fill_column, True)  # True: fill with last non-null
        .over(window_spec))

    # Drop the old column and rename the new column
    ff_out = ff.drop(fill_column).withColumnRenamed('fill_fwd', fill_column)

    return ff_out


def fill_backward(spark_df, key_column, fill_column, id_column=None):
    """
    Fill the na values of the specified column with backward fill method.

    :param spark_df: spark dataframe
    :param key_column: name of the column to sort by
    :param fill_column: name of the column to fill in the na values
    :param id_column: name of the column to partition the spark dataframe. Default is None

    :returns: spark dataframe with the specified column that has the na values filled using backward fill method
    """

    if id_column is not None:  # if id_column is provided, partition by the id_column
        window_spec = Window.partitionBy(id_column) \
            .orderBy(key_column)                     \
            .rowsBetween(0, sys.maxsize)
    else:                  # else just do forward fill as is
        window_spec = Window.orderBy(key_column) \
            .rowsBetween(0, sys.maxsize)

    # Fill null's with last *non null* value in the window
    bf = spark_df.withColumn(
        'fill_backwd',
        F.first(fill_column, True)  # True: fill with last non-null
        .over(window_spec))

    # Drop the old column and rename the new column
    bf_out = bf.drop(fill_column).withColumnRenamed('fill_backwd', fill_column)

    return bf_out


def fill_linear_interpolation(spark_df, order_col, value_col, id_cols=None):
    """
    Apply linear interpolation to dataframe to fill gaps.

    :param spark_df: spark dataframe
    :param order_col: column to use to order by the window function
    :param value_col: column to be filled
    :param id_cols: string or list of column names to partition by the window function

    :returns: spark dataframe updated with interpolated values
    """
    # create row number over window and a column with row number only for non
    # missing values

    if id_cols is not None:
        w = Window.partitionBy(id_cols).orderBy(order_col)
    else:
        w = Window.orderBy(order_col)
    new_df = spark_df.withColumn('rn', F.row_number().over(w))
    new_df = new_df.withColumn(
        'rn_not_null',
        F.when(
            F.col(value_col).isNotNull(),
            F.col('rn')))

    # create relative references to the start value (last value not missing)
    if id_cols is not None:
        w_start = Window.partitionBy(id_cols).orderBy(
            order_col).rowsBetween(Window.unboundedPreceding, -1)
    else:
        w_start = Window.orderBy(order_col).rowsBetween(
            Window.unboundedPreceding, -1)
    new_df = new_df.withColumn(
        'start_val', F.last(
            value_col, True).over(w_start))
    new_df = new_df.withColumn(
        'start_rn', F.last(
            'rn_not_null', True).over(w_start))

    # create relative references to the end value (first value not missing)
    if id_cols is not None:
        w_end = Window.partitionBy(id_cols).orderBy(
            order_col).rowsBetween(0, Window.unboundedFollowing)
    else:
        w_end = Window.orderBy(order_col).rowsBetween(
            0, Window.unboundedFollowing)
    new_df = new_df.withColumn('end_val', F.first(value_col, True).over(w_end))
    new_df = new_df.withColumn(
        'end_rn', F.first(
            'rn_not_null', True).over(w_end))

    if id_cols is not None:
        if not isinstance(
                id_cols,
                list):  # make id_cols as list if the user didn't parse in as list
            id_cols = [id_cols]

    # create references to gap length and current gap position
    new_df = new_df.withColumn('diff_rn', F.col('end_rn') - F.col('start_rn'))
    new_df = new_df.withColumn('curr_rn', F.col(
        'diff_rn') - (F.col('end_rn') - F.col('rn')))

    # calculate linear interpolation value
    lin_interp_func = (F.col('start_val') +
                       (F.col('end_val') -
                        F.col('start_val')) /
                       F.col('diff_rn') *
                       F.col('curr_rn'))
    new_df = new_df.withColumn(
        value_col,
        F.when(
            F.col(value_col).isNull(),
            lin_interp_func).otherwise(
            F.col(value_col)))

    new_df = new_df.drop(
        'rn',
        'rn_not_null',
        'start_val',
        'end_val',
        'start_rn',
        'end_rn',
        'diff_rn',
        'curr_rn')
    return new_df

def iterative_imputation(
    sparkcontext,
    spark_df,
    columns_to_drop,
    cols_to_orderBy,
    orderBy_strategy='asc',
    initial_strategy='mean',
    imputation_order='ascending',
    max_iter=10,
    random_state=0,
    add_indicator=False
):
    """
    Fill in missing values using sklearn multivariate imputer that estimates each feature from all the others.

    :param sparkcontext: spark context derived from the spark session
    :param spark_df: spark dataframe
    :param columns_to_drop: list - name of the columns to drop
    :param initial_strategy: the strategy used to initialize the missing values.
        Possible values are 'mean', 'median', 'most_frequent', 'constant'.
        Default value is 'mean'
    :param imputation_order: the order in which the features will be imputed.
        Possible values are 'ascending', 'descending', 'roman' (left to right), 'arabic' (right to left), 'random'
    :param max_iter: maximum number of imputation rounds to perform
    :param random_state: the seed of the pseudo random number generator to use. Used for determinism
    :param add_indicator: if True, will create additional columns to indicate missingness.
        This allows a predictive estimator to account for missingness despite imputation

    :returns: spark dataframe with missing values filled using iterative imputer
    """

    np.set_printoptions(suppress=True)
    pd.options.display.float_format = '{:.1f}'.format

    if not isinstance(
            columns_to_drop,
            list):  # make columns_to_drop as list if the user didn't parse in as list
        columns_to_drop = [columns_to_drop]

    # the part of the df after the specified columns are dropped
    df_after_drop = spark_df.drop(*columns_to_drop)
    # the part of the df that is dropped. kept for concatenation later
    dropped_df = spark_df.select(*columns_to_drop)
    # cast the dropped_df into float dtype for all columns for imputation
    df_imputer = df_after_drop.select(
        *(F.col(c).cast("float").alias(c) for c in df_after_drop.columns))

    # intialize sklearn iterative imputer
    imp = IterativeImputer(
        initial_strategy=initial_strategy,
        imputation_order=imputation_order,
        max_iter=max_iter,
        random_state=random_state,
        add_indicator=add_indicator
    )
    # returns ndarray. not sure if it will still work when the df is large
    # though...
    res = imp.fit_transform(df_imputer.toPandas().copy(deep=True))

    # convert the returned ndarray into spark df
    rdd_res1 = sparkcontext.parallelize(res)
    rdd_res2 = rdd_res1.map(lambda x: [float(i) for i in x])
    df = rdd_res2.toDF(df_after_drop.columns)

    # generate sequentially increasing index for the two dfs for concatenation
    # assumption is that the order of the two dfs have no been altered in any
    # way up to this point
    df1 = generate_seq_idx(df)
    df2 = generate_seq_idx(dropped_df)

    # concatenate the imputed df with the dropped df
    concat_df = df2.join(df1, "idx", "outer").drop("idx")

    if orderBy_strategy == 'asc':
        return concat_df.orderBy(*cols_to_orderBy)
    elif orderBy_strategy == 'desc':
        return concat_df.orderBy(*cols_to_orderBy).desc()
    else:
        raise InvalidOrderByStrategy("Invalid value of orderBy_strategy.")    

def _fill_with_random_choices(category_list, discrete_prob_list):
    """
    Private function for fill_na_weighted_probability() to fill missing values with weighted probability. 
    Assumption is that the `category_list` must consist of a value 'None' of dtype string.

    :param category_list: a list of the values of the categorical data column
    :param discrete_prob_list: a list of probabilities of the associating categorical data column

    :returns: one of the values in the category_list. Will never be 'None' due to the recursive algorithm
    """

    res = random.choices(category_list, discrete_prob_list, k=1)[0]
    while res == 'None':  # recursion until I don't get 'None'
        res = _fill_with_random_choices(category_list, discrete_prob_list)
    return res

def fill_na_weighted_probability(spark_df, column_name):
    """
    An algorithm to fill the specified column's missing values with weighted probability. 

    :param spark_df: spark dataframe
    :param column_name: name of the column to perform this algorithm (categorical data)

    :returns: spark dataframe with the specified column's missing values filled using weighted probability 
    """    
    # group by and count the frequencies of each value in the column
    res = spark_df.groupBy(column_name).count()

    # get the total number of rows (sum of frequencies)
    sum_of_count = res.groupBy().agg(F.sum('count')).collect()[0][0]  # returns an int

    # calculate the discrete probability of each value in that column
    res = res.withColumn('discrete_prob', F.col('count')/sum_of_count)    

    # convert the input column and the discrete probability column into a list for making a random choice when filling in null values
    category_list = col_tolist(res, column_name)
    discrete_prob_list = col_tolist(res, 'discrete_prob')

    # fill in null values using random choices
    df = spark_df.na.fill({'category':random.choices(category_list, discrete_prob_list, k=1)[0]})

    return df
