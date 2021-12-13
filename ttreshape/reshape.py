import copy


def factorize(n):
    # факторизация на множители >= 2
    fact = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            fact.append(d)
            n = n // d
        else:
            d += 1
    if n > 1:
        fact.append(n)
    return sorted(fact)


def toNdim(a, n):
    while len(a) > n:
        c = copy.copy(a)
        a = sorted([c[0] * c[1]] + c[2::])
    return a


def vector_equal(a, b, n):
    # выравнивание длин векторов до заданной длины
    if len(a) > n:
        a = toNdim(a, n)
    elif len(a) < n:
        a = [1] * (n - len(a)) + a
    if len(b) > n:
        b = toNdim(b, n)
    elif len(b) < n:
        b = [1] * (n - len(b)) + b
    return a, b
