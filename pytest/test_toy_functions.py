import pytest
import toy_functions as tf


def test_calc_sum():
    total = tf.calc_sum(4,5)
    assert total == 9
    


def test_calc_product():
    total = tf.calc_product(10,3)
    assert total == 30


def test_capital_case():
    assert tf.capital_case('semaphore') == 'Semaphore'



def test_raises_exception_on_non_string_arguments():
    with pytest.raises(TypeError):
        tf.capital_case(9)