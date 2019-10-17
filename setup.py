from distutils.core import setup
import setuptools


setup(
      name='numerico',
      description='Biblioteca de cálculo numérico',
      author='Felipe Ribas Muniz',
      author_email='muniz.r.felipe@gmail.com',
      url='https://github.com/fmuniz351987/calculo-numerico',
      package_dir={'': 'src'},
      packages=['numerico', 'numerico.linalg', 'numerico.interpolation', 
                'numerico.regression', 'numerico.integral', 'numerico.eqroot'],
      install_requires=['numpy>=1.17.2']
    )
