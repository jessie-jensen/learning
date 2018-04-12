from pyspark import SparkContext, SparkConf
import logging

import pytest
from pyspark_wordcount import *



@pytest.fixture(scope='session')
def spark_context(request):
    '''fixture creating local spark context'''

    conf = (SparkConf().setMaster("local[2]").setAppName("pytest-pyspark-local-testing"))
    sc = SparkContext(conf=conf)
    request.addfinalizer(lambda: sc.stop())
    
    logger = logging.getLogger('py4j')
    logger.setLevel(logging.WARN)

    return sc



#@pytest.mark.use
def test_do_word_counts(spark_context):
    test_input = [
        ' hello spark ',
        ' hello again spark spark'
    ]

    input_rdd = spark_context.parallelize(test_input, 1)
    results = word_counts(input_rdd)

    expected_results = {'hello':2, 'spark':3, 'again':1}  
    assert results == expected_results