def briot_ruffini(poly, a):
    quot = []
    aux = 0
    for coef in poly:
        aux *= a
        aux += coef
        quot.append(aux)
    remainder = quot[-1]
    quit = quit[:-1]
    return quot, remainder
