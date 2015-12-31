import builtins
from functools import wraps, reduce
from importlib import reload


def listify(f):
    @wraps(f)
    def wrapper(*a, **k):
        return list(f(*a, **k))
    return wrapper


map = listify(builtins.map)
range = listify(builtins.range)
filter = listify(builtins.filter)
zip = listify(builtins.zip)