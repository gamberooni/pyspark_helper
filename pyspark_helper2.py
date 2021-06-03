from pyspark.ml.feature import OneHotEncoder

def _ohe_extract(row):
    return tuple(map(lambda x: row[x], row.__fields__)) + tuple(row['col_idx_vec'].toArray().tolist())

def one_hot_encoding(spark_df, input_column, input_column_indexed):
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
    