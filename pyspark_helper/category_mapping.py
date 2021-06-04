from pyspark_helper.common import cast_column
from pyspark.ml.feature import OneHotEncoder, StringIndexer


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

def _ohe_extract(row):
    """
    Private function for one_hot_encoding() to extract the OHE vector into multiple columns.

    :param row: one of the rows of the original data frame that only has one additional column (OHE vector)

    :returns: a new row with additional columns after mapping the OHE vector into multiple columns
    """
    return tuple(map(lambda x: row[x], row.__fields__)) + tuple(row['col_idx_vec'].toArray().tolist())

def one_hot_encoding(spark_df, input_column, input_column_indexed):
    """
    Performs one-hot encoding on a categorical data column after numerical mapping.

    :param spark_df: spark dataframe
    :param input_column: name of the column before numerical mapping
    :param input_column_indexed: name of the column after numerical mapping

    :returns: a spark dataframe with additional columns after one-hot encoding
    """
    output_column = "col_idx_vec"

    # rename columns to ensure output deterministic
    spark_df = spark_df.withColumnRenamed(input_column, "col") \
        .withColumnRenamed(input_column_indexed, "col_idx")

    # create OneHotEncoder instance
    ohe = OneHotEncoder(inputCol="col_idx", outputCol=output_column)
    ohe.setDropLast(False)

    ohe_df = ohe.fit(spark_df).transform(spark_df)

    # get the name of the additional columns for one hot encoding
    colIdx = ohe_df.select("col", "col_idx").distinct().rdd.collectAsMap()
    colIdx =  sorted((value, "ls_" + key) for (key, value) in colIdx.items())
    newCols = list(map(lambda x: x[1], colIdx))

    # get the original columns
    actualCol = ohe_df.columns

    # combined all the columns together 
    allColNames = actualCol + newCols

    # extract the OHE vector into multiple columns
    result_df = ohe_df.rdd.map(_ohe_extract).toDF(allColNames)    
    for col in newCols:
        result_df = result_df.withColumn(col, result_df[col].cast("int"))

    # rename columns back to original names
    result_df = result_df.withColumnRenamed("col", input_column) \
        .withColumnRenamed("col_idx", input_column_indexed) \
        .withColumnRenamed("col_idx_vec", f"{input_column}_vec")

    return result_df
