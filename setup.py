from distutils.unixccompiler import UnixCCompiler
from numpy.distutils.exec_command import find_executable
from distutils.core import setup, Extension
import os
import numpy
import subprocess

libs = [
    'cblas',
    'gomp',
]

flags = [
    '-fopenmp',
    '-ffast-math',
    '-Wno-write-strings',
    '-DDEBUG',
]

library_dirs = [
    '/usr/lib/atlas-base',
]

include_dirs = [
    '/usr/include/atlas',
]


regionproposals_regionproposals = Extension(
    'regionproposals._regionproposals',

    sources=[
        'src/regionproposals/_regionproposals.c'
    ],

    library_dirs=library_dirs,

    libraries=[
        'boost_system',
    ]+libs,

    extra_compile_args=[
        '-g', 
        '-O3', 
        '-Wall', 
        '-Wno-long-long',
        '-funroll-loops',
        subprocess.check_output(['pkg-config', '--cflags', 'eigen3']).strip(),
    ]+flags+os.environ.get('CXXFLAGS','').split(),

    extra_link_args=[
        subprocess.check_output(['pkg-config', '--libs', 'eigen3']).strip(),
    ]+os.environ.get('LDFLAGS','').split(),

    include_dirs=[
        numpy.get_include(),
    ]+include_dirs,
)


setup ( 
    name='regionproposals',
    version='0.0',
    description='',
    author='Kevin Matzen',
    author_email='kmatzen@gmail.com',
    classifiers=[],
    ext_modules=[
        regionproposals_regionproposals,
    ],
    packages=['regionproposals'],
    package_dir={'regionproposals':'src/regionproposals'},

    scripts=[
    ],
    requires=[
        'numpy',
    ],
)
