from pyspark_helper.exceptions import *
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql.window import Window
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from datetime import datetime
from functools import reduce
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import re
import os


# pd.set_option('max_colwidth', None) # to prevent truncating of columns
# in jupyter

def header_replace_special_char(spark_df):
    """
    Replace special characters in column headers

    :param spark_df: spark dataframe

    :returns: spark dataframe with special characters replace in the column headers
    """

    return spark_df.toDF(*[re.sub('[\\)|\\(|\\s|,|%.]', '_', x)
                         for x in spark_df.columns])


def count_null(spark_df):
    """
    Counts all occurences of null values in a spark dataframe.

    :param spark_df: spark dataframe

    :returns: spark dataframe with two columns - 'Column_Name' and 'NULL_Count'
    """

    df_agg = spark_df.agg(
        *[F.count(F.when(F.isnull(c), c)).alias(c) for c in spark_df.columns])

    return reduce(
        lambda a,
        b: a.union(b),
        (df_agg.select(
            F.lit(c).alias("Column_Name"),
            F.col(c).alias("NULL_Count")) for c in df_agg.columns))


def df_summary_tofile(
    spark_df,
    output_filepath=os.path.join(os.getcwd(), 'summary'),
    output_filename=f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}-df_summary.txt'
):
    """
    Write summary of the spark dataframe into a file.
    The summary consists of the pyspark summary() output, aggregation of columns corresponding dtypes and spark dataframe schema.

    :param spark_df: spark dataframe
    :param output_filepath: path of the directory to store the file
    :param output_filename: filename of the spark dataframe summary. Default value is datetime.now().strftime(%Y-%m-%d %H:%M:%S)-df_summary.txt

    :returns: None
    """

    count = pd.DataFrame(spark_df.dtypes).groupby(1, as_index=False)[0].agg(
        {'count': 'count', 'names': lambda x: " | ".join(set(x))}).rename(columns={1: "type"})

    if output_filename is not None:
        # make directory including its parent directories if not exists
        Path(output_filepath).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_filepath, output_filename)     

        # write the output to file
        original_stdout = sys.stdout  # saving the reference of the std output
        with open(output_file, 'w') as f:
            sys.stdout = f
            print("Dataframe summary:")
            spark_df.summary().show(truncate=False)
            print("Number of missing values for each column:")
            count_null(spark_df).show(spark_df.count(), truncate=False)
            print("Aggregated count of columns and respective data types:")
            print(count.to_markdown(index=False) + '\n')
            print("Dataframe schema:")
            spark_df.printSchema()
            sys.stdout = original_stdout  # reset the std output
    else:
        raise InvalidFilenameError("Output filename is invalid.")


def cast_column(spark_df, column_name, cast_type):
    """
    Cast the dtype of the specified column into another dtype.

    :param spark_df: spark dataframe
    :param column_name: name of the column to do dtype casting
    :param cast_type: dtype to cast into. Possible values are for example: 'integer', 'float', 'double', 'string', etc.

    :returns: spark dataframe with the specified column casted into another dtype
    """
    return spark_df.withColumn(
        column_name,
        spark_df[column_name].cast(cast_type).alias(column_name))


def plot_histogram(
        spark_df,
        column_name,
        buckets=20,
        output_filepath=os.path.join(
            os.getcwd(),
            'summary'),
        output_filename='hist.png'):
    """
    Plot histogram for the specified column. Save histogram as .png if filename is provided. Otherwise open a window to show plot.

    :param spark_df: spark dataframe
    :param column_name: name of the column to plot histogram
    :param buckets: number of bins for the histogram. Higher number means more fine-grained
    :param output_filepath: path of the directory to store the file
    :param output_filename: the filename of the histogram plot output. Default is 'hist.png'

    :returns: None
    """

    bins, counts = spark_df.select(column_name).rdd.flatMap(
        lambda x: x).histogram(buckets)
    plt.hist(bins[:-1], bins=bins, weights=counts)
    plt.xlabel(column_name)
    plt.ylabel('frequency')

    if output_filename is not None:
        # make directory including its parent directories if not exists
        Path(output_filepath).mkdir(parents=True, exist_ok=True)        
        plt.savefig(output_filename)
    else:
        plt.show()


def detect_outliers_IQR(spark_df, column_name, relative_error=0.1):
    """
    Detects outliers based on the IQR method. Uses the Greenwald-Khanna algorithm to approximate the first and third-quartile range.

    :param spark_df: spark dataframe
    :param column_name: name of the column to detect outliers
    :param relative_error: relative error of the Greenwald-Khanna algorithm. Default value is 0.01

    :returns: spark dataframe of the specified column with outliers and outlier count
    """

    Q1 = spark_df.approxQuantile(column_name, [0.25], relative_error)[0]
    Q3 = spark_df.approxQuantile(column_name, [0.75], relative_error)[0]
    IQR = Q3 - Q1

    # is outlier if a data point is > 1.5 * IQR + Q3 or < Q1 - 1.5 * IQR
    outliers = spark_df.select(column_name).filter(
        (spark_df[column_name] > Q3 +
         1.5 *
         IQR) | (
            spark_df[column_name] < Q1 -
            1.5 *
            IQR))
    outlier_count = outliers.count()

    return outliers, outlier_count


def get_skewness(spark_df, column_name):
    """
    Get the skewness of the specified column.

    :param spark_df: spark dataframe
    :param column_name: name of the column to calculate skewness from

    :returns: skewness value of the specified column
    """

    return spark_df.select(F.skewness(spark_df[column_name]))


def map_categorical_data(spark_df, input_column, output_column):
    """
    Map categorical data into integer labels using pyspark.ml.feature StringIndexer.

    :param spark_df: spark dataframe
    :param input_column: name of the column with categorical data
    :param output_column: name of the column which holds the labels of the `input` column

    :returns: spark dataframe with additional `output` column which holds the labels of the `input` column
    """

    stringIndexer = StringIndexer(
        inputCol=input_column,
        outputCol=output_column)
    mapped_df = stringIndexer.fit(spark_df).transform(spark_df)

    return cast_column(mapped_df, output_column, "integer")

def one_hot_encoding(spark_df, input_column):
    pass

def convert_timestamp_format(
        spark_df,
        ts_column_name,
        original_format,
        converted_format="yyyy-MM-dd"):
    """
    Map categorical data into integer labels using pyspark.ml.feature StringIndexer.

    :param spark_df: spark dataframe
    :param ts_column_name: name of the column with timestamp dtype
    :param original_format: original format of the timestamp column
    :param converted_format: datetime format for the timestamp column to convert into. Default is 'yyyy-MM-dd'

    :returns: spark dataframe with the specified timestamp column converted into another datetime format
    """
    # convert string datetime into datetime object, then change into different
    # datetime representation format
    return spark_df.withColumn(
        ts_column_name,
        F.from_unixtime(
            F.unix_timestamp(
                ts_column_name,
                original_format),
            converted_format)  # convert from yyyy-MM-dd to dd/MM/yyyy
    )


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


def get_column_mean(spark_df, column_name):
    """
    Get the mean for the specified column.

    :param spark_df: spark dataframe
    :param column_name: name of the column to calculate median

    :returns: mean of the specified column
    """

    df_mean = spark_df.select(
        F.mean(F.col(column_name)).alias('mean'),
    ).collect()
    return df_mean[0]['mean']


def get_column_median(spark_df, column_name, relative_error=0.1):
    """
    Get the approximation of median with a certain value of relative error for the specified column using the Greenwald-Khanna algorithm.

    :param spark_df: spark dataframe
    :param column_name: name of the column to calculate median
    :param relative_error: relative error of the Greenwald-Khanna algorithm. Default value is 0.01

    :returns: median of the specified column
    """

    return spark_df.approxQuantile(column_name, [0.5], relative_error)[0]


def get_column_mode(spark_df, column_name):
    """
    Get the mode for the specified column.

    :param spark_df: spark dataframe
    :param column_name: name of the column to calculate mode

    :returns: mode of the specified column
    """

    # count the number of occurences of each value in the specified column
    df = spark_df.groupBy(column_name).count()
    # sort by descending and get the value from the first row
    mode = df.orderBy(df['count'].desc()).collect()[0][0]
    if mode is None:  # if mode is None then return the next value with the most occurence
        mode = df.orderBy(df['count'].desc()).collect()[1][0]

    return mode


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


def replace_value(
        spark_df,
        column_name,
        value_to_replace,
        replace_with,
        regex=False):
    """
    Replace all rows of the specified column with a certain value with another value.

    :param spark_df: spark dataframe
    :param column_name: name of the column to do the value replacement
    :param value_to_replace: original value to be replaced with new value
    :param replace_with: new value to replace the original value
    :param regex: whether to use regex pattern matching to replace value. Default is False

    :returns: spark dataframe with the original values of the specified column replaced with new values
    """

    if regex:
        return spark_df.withColumn(
            column_name,
            F.regexp_replace(
                column_name,
                value_to_replace,
                replace_with)) .otherwise(
            spark_df[column_name])
    else:
        return spark_df.withColumn(
            column_name,
            F.when(
                spark_df.usage == value_to_replace,
                replace_with).otherwise(
                spark_df[column_name]))


def generate_seq_idx(spark_df):
    """
    Generate sequentially increasing id.

    :param sparkcontext: spark dataframe

    :returns: spark dataframe with sequentially increasing index additional column
    """

    rdd_df = spark_df.rdd.zipWithIndex()
    df = rdd_df.toDF()  # column '_1' is the zipped column, '_2' is the index

    for col in spark_df.columns:
        # unpack the zipped column and add each column as new column to the df
        df = df.withColumn(col, df['_1'].getItem(col))

    # drop the column with zipped values and rename '_2' column into idx
    df = df.drop('_1').withColumnRenamed('_2', 'idx')

    return df


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

def identify_timegaps(spark_df, ts_column_name, plot_hist=False, hist_buckets=10, hist_output_filename=None):
    """
    Identify timegaps in the spark dataframe. Provides the option to plot a histogram to visualize the distribution.

    :param spark_df: spark dataframe
    :param ts_column_name: name of the timestamp column
    :param plot_hist: whether to plot histogram. Default is False
    :param hist_buckets: histogram bins
    :param hist_output_filename: filename that the histogram should output to. Default is None - means show histogram in a plot window

    :returns: two spark dataframes. One of the df shows the three columns (ts, ts_next, diff_in_seconds). Another shows the df grouped by diff_in_seconds
    """

    ts_next_column_name = "ts_next"
    ts_diff_column_name = "diff_in_seconds"

    w = Window().partitionBy().orderBy(F.col(ts_column_name))
    df = spark_df.select("*", F.lead(ts_column_name).over(w).alias(ts_next_column_name))

    #Calculate Time difference in Seconds
    ts_diff_df = df.withColumn(ts_column_name, F.to_timestamp(F.col(ts_column_name))) \
        .withColumn(ts_next_column_name, F.to_timestamp(F.col(ts_next_column_name))) \
        .withColumn(ts_diff_column_name, F.col(ts_next_column_name).cast("long") - F.col(ts_column_name).cast("long"))

    if plot_hist is True:
        plot_histogram(ts_diff_df, ts_diff_column_name, buckets=hist_buckets, output_filename=hist_output_filename)

    ts_diff_df_raw = ts_diff_df.select(ts_column_name, ts_next_column_name, ts_diff_column_name)

    # show how many occurences each diff_in_seconds value has
    ts_diff_df_grouped = ts_diff_df.groupBy(ts_diff_column_name).count()

    return ts_diff_df_raw, ts_diff_df_grouped
