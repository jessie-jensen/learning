def calc_sum(a,b):
    return a+b



def calc_product(a,b):
    return a*b



def capital_case(x):
    if not isinstance(x, str):
        raise TypeError('Requires string')
    return x.capitalize()