from operator import add
import pyspark

from pyspark import SparkContext, SparkConf

conf = (SparkConf().setMaster("local[2]").setAppName("wordcount"))
sc = SparkContext(conf=conf)


def word_counts(lines):
    counts = (lines.flatMap(lambda x: x.split()).map(lambda x: (x, 1)).reduceByKey(add))
    results = counts.collect()
    #results = {word: count for word, count in counts.collect()}
    return results



#######################


test_input = [' hello spark ',' hello again spark spark']
print(test_input)

input_rdd = sc.parallelize(test_input, 1)
results = word_counts(input_rdd)

print(results)

lines = spark.read.text(test_input).rdd.map(lambda r: r[0])

spark = SparkSession\
    .builder\
    .appName("PythonWordCount")\
    .getOrCreate()