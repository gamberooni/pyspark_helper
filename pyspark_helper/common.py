from pyspark_helper.exceptions import *
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime
from functools import reduce
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import sys


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


def df_summary_tofile(
    spark_df,
    ts_column_name=None,
    output_filepath=os.path.join(os.getcwd(), 'summary'),
    output_filename=f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}-df_summary.txt'
):
    """
    Write summary of the spark dataframe into a file. Provide the ts_column_name if the df is time-series data.
    The summary consists of the pyspark summary() output, number of missing values for each column, 
        time gaps, aggregation of columns corresponding dtypes and spark dataframe schema.

    :param spark_df: spark dataframe
    :param ts_column_name: timestamp column name (if the dataframe is time-series data)
    :param output_filepath: path of the directory to store the file
    :param output_filename: filename of the spark dataframe summary. Default value is datetime.now().strftime(%Y-%m-%d %H:%M:%S)-df_summary.txt

    :returns: None
    """

    line_break = '#' * 100 + '\n'

    count = pd.DataFrame(spark_df.dtypes).groupby(1, as_index=False)[0].agg(
        {'count': 'count', 'column_names': lambda x: " | ".join(set(x))}).rename(columns={1: "type"})

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
            print(line_break)

            print("Number of missing values for each column:")
            count_null(spark_df).show(spark_df.count(), truncate=False)
            print(line_break)

            if ts_column_name != None:
                ts_diff_raw, ts_diff_grouped = identify_timegaps(spark_df, ts_column_name)
                print("Time gaps summary:")
                ts_diff_raw.show()
                ts_diff_grouped.show()
                print(line_break)

            print("Aggregated count of columns and respective data types:")
            print(count.to_markdown(index=False) + '\n')
            print(line_break)

            print("Dataframe schema:")
            spark_df.printSchema()

            sys.stdout = original_stdout  # reset the std output
    else:
        raise InvalidFilenameError("Output filename is invalid.")

def col_tolist(spark_df, column_name):
    """
    Converts a spark dataframe into a list of distinct values from the df.

    :param spark_df: spark dataframe
    :param column_name: name of the column to convert into the list

    :returns: a list that consists of the distinct values of the spark dataframe column
    """

    return list(
        spark_df.select(column_name).toPandas()[column_name]
    )    

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
