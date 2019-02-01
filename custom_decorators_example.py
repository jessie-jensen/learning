# https://realpython.com/primer-on-python-decorators/
import datetime as dt
import functools

def do_twice(func):
    @functools.wraps(func)
    def wrapper_do_twice(*args, **kwargs):
        func(*args, **kwargs)
        func(*args, **kwargs)

    return wrapper_do_twice


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = dt.datetime.now()
        func(*args, **kwargs)
        t1 = dt.datetime.now()

        print('EXECUTION TIME:', t1 - t0)
        return t1-t0

    return wrapper


def debug(func):
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]                      # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)           # 3
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")           # 4
        return value
    return wrapper_debug




@do_twice
def say_hello(name):
    print(f'hello {name}!')


@timer
@debug
def sum_of_sequential_squares(n):
    l = [i**2 for i in range(n)]
    return sum(l)


if __name__ == "__main__":
    say_hello('world')
    sum_of_sequential_squares(1000*1000)