from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from argparse import ArgumentParser
import sys
import pandas as pd

# parse the argument
argp = ArgumentParser()

argp.add_argument('--data_uri', type=str, required=True, help='mongodb uri containing data')
argp.add_argument('--result_uri', type=str, required=True, help='mongodb uri containing data')

if __name__ == '__main__':
    args = argp.parse_args(sys.argv[1:])

COL_TYPES = ['int', 'string', 'float', 'double']

def read_table(uri):
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.format('mongo').option('uri', uri).option('sampleSize', 10000).load()
    return df

def get_column_stats(df, c, dtype):
    count = df.count()
    res = df.select(F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c),
                    F.min(c), F.max(c), F.mean(c), F.countDistinct(c),
                    F.stddev(c)).collect()[0]

    null_nan_count, min_val, max_val, mean_val, distict_count, std_dev_val, total_count = res[0], res[1], res[2], \
                                                                                          res[3], res[4], res[5], \
                                                                                          count - res[4]
    if dtype == 'string':
        return [c, dtype, count, null_nan_count, float('NaN'), float('NaN'), float('NaN'), distict_count, float('NaN'), total_count]

    return [c, dtype, count, null_nan_count, min_val, max_val, mean_val, distict_count, std_dev_val, total_count]


def run_profiler(uri, output_uri):
    spark = SparkSession.builder.getOrCreate()
    # read the table from the uri and cache it as we are going to do computation on this
    df = read_table(uri).cache()

    # get column stats
    res_list = []
    col_types = df.dtypes
    for rec in col_types:
        if rec[1] in COL_TYPES:
            col_stats = get_column_stats(df, rec[0], rec[1])
            res_list.append(col_stats)

    # convert to pandas dataFrame
    cols = ['col_name', 'col_type', 'count', 'null_nan_count', 'min_val', 'max_val', 'mean_val', 'distict_count',
            'std_dev', 'total_count']
    pdf = pd.DataFrame(res_list, columns=cols)

    # writing back to spark
    result_df = spark.createDataFrame(pdf)
    df.write.mode('overwrite').format('mongo').option('uri', output_uri).save()
    print('Result written to MongoDB')

def main(args):
    run_profiler(args.data_uri, args.result_uri)


if __name__ == '__main__':
    main(args)
