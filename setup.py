from setuptools import setup, Extension
from distutils.command.install import install as DistutilsInstall
from setuptools.command.install import install
from distutils.command.build import build
import os
from multiprocessing import cpu_count
from subprocess import call

BASEPATH = os.path.dirname(os.path.abspath(__file__))
BTM_PATH = os.path.join(BASEPATH, 'btm')

btm_cpp = Extension('btm_cpp',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '0')],
                    libraries = ['boost_python3', 'boost_numpy3'],
                    sources = ['btm/model.cpp','btm/infer.cpp'])




setup(
    name='btm',
    version='0.1.0',
    packages=['btm'],
    ext_modules = [btm_cpp]
)

