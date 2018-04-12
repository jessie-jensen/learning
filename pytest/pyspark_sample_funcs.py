from operator import add
from pyspark import SparkContext, SparkConf



def word_counts(lines):
    """ count of words in an rdd of lines """

    counts = lines.flatMap(lambda x: x.split())\
                  .map(lambda x: (x, 1))\
                  .reduceByKey(add)\
                  .collect()

    results = {word: count for word, count in counts}

    return results



def json_name_filter_counts(df, target_name):
    return df.filter(df.name == target_name).count()