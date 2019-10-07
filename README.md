# calculo-numerico

Para instalar, simplesmente use:

```bash
sudo python3 setup.py install
```

E então, dentro do seu código, use `import numerico`.

## linalg

Para usar os recursos de solução e decomposição de sistemas lineares, você pode usar o seguinte código:
```python
import numpy as np
from numerico import linalg

a = np.array([
    [1, 5],
    [2, 3]
])
lu = linalg.LU(a)
print(lu.det)
print(lu.inv())
print(lu.solve([2,-1]))
```
Além da decomposição LU, estão disponíveis Cholesky e LDLt. Use Cholesky e LDLt para matrizes simétricas; Cholesky se você não se importa de obter resultados complexos, e LDLt para evitar o cálculo de raízes.

## interp

O submódulo interp possui as interpolações usando a matriz de Vandermonde, a interpolação de Lagrange e os polinômios interpoladores de Newton e Gregory-Newton. Vandermonde é útil quando você deseja encontrar um polinômio interpolador e utilizá-lo diversas vezes para calcular vários pontos distintos, ou simplesmente saber seus coeficientes; Lagrange é útil se você deseja calcular apenas um único ponto após a interpolação; use Newton para todos os outros casos, onde os pontos x não são igualmente espaçados, ou Gregory-Newton quando o são. Gregory-Newton é especialmente útil se você está calculando a integral de uma função que não possui uma primitiva, e você pode estimar analiticamente os valores de y dados valores de x arbitrários. Newton é mais útil quando x não pode ser medido analiticamente e apresenta algum erro de medida, sendo impossível manter os pontos equidistantes entre si.

```python
import numpy as np
from numerico import interp

# interpolando uma parábola (y = x**2)
x = np.array([-1,0,1])
y = np.array([1,0,1])
vd = interp.Vandermonde(x, y)
print(vd.coefs)     # mostra os coeficientes 1, 0, 0
estimate = vd(2)    # retorna 2**2 == 4
```

Classes disponíveis: Vandermonde, Lagrange, Newton, GregoryNewton
Métodos disponíveis: choose_points

## regression
O submódulo regression possui as classes Linear e Polynomial. A classe Linear define regressões múltiplas ou simples, enquando a classe Polynomial define regressão polinomial. Para usar regressões não-lineares nem polinomiais, linearize o sistema primeiro. Você pode linearizar um polinômio se quiser e usar a classe Linear, mas Polynomial é mais eficiente para regressões polinomiais.

```python
import numpy as np
from numerico import regression

# regressão do tipo y = c0 + c1x_1 + c2x_2 (linear)
x = np.array([[0, 1, 2, 3, 4],
              [-2, -1, 0, 1, 2]]).T    # atenção ao T que chama a transposta
y = np.array([1.8, 1.9, 3.1, 3.9, 3.3])
reg = regression.Linear(x, y)
pred = round(reg([0.5, 1]), 4)     # pred == 3.3 aproximadamente.
```
