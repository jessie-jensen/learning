from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import logging

import pytest
import pyspark_sample_funcs as psf



#
### fixtures
#

@pytest.fixture(scope='session')
def sc(request):
    '''fixture creating local spark context'''

    conf = (SparkConf().setMaster("local[2]").setAppName("pytest-pyspark-local-testing"))
    sc = SparkContext(conf=conf)
    request.addfinalizer(lambda: sc.stop())
    
    logger = logging.getLogger('py4j')
    logger.setLevel(logging.WARN)

    return sc



@pytest.fixture(scope='session')
def spark():
    '''fixture creating local spark session'''
    spark = SparkSession.builder.master('local[2]').appName('pytests').getOrCreate()

    return spark



#
### tests
#

def test_word_counts(sc):
    test_input = [
        ' hello spark ',
        ' hello again spark spark'
    ]

    input_rdd = sc.parallelize(test_input, 1)
    results = psf.word_counts(input_rdd)

    expected_results = {'hello':2, 'spark':3, 'again':1}  
    assert results == expected_results



@pytest.mark.parametrize('name,expected',[
    ('vikas',2),
    ('john',1),
    ('jane',1),
])
def test_json_name_filter_counts(sc, spark, name, expected):
    test_input = [
        {'name': 'vikas'},
        {'name': 'vikas'},
        {'name': 'john'},
        {'name': 'jane'},
    ]

    input_rdd = sc.parallelize(test_input, 1)
    df = spark.read.json(input_rdd)
    results = psf.json_name_filter_counts(df, name)

    assert results == expected