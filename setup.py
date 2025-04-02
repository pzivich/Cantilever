from setuptools import setup

exec(compile(open('cantilever/version.py').read(),
             'cantilever/version.py', 'exec'))

with open("README.md") as f:
    descript = f.read()


setup(name='cantilever',
      version=__version__,
      description='Bridged comparisons',
      keywords='bridge',
      packages=['cantilever',
                'cantilever.estimators',
                'cantilever.estimators.point',
                ],
      include_package_data=True,
      license='MIT',
      author='Paul Zivich',
      author_email='zivich.5@gmail.com',
      # url='https://github.com/pzivich/Deli',
      classifiers=['Programming Language :: Python :: 3.8',
                   'Programming Language :: Python :: 3.9',
                   'Programming Language :: Python :: 3.10',
                   'Programming Language :: Python :: 3.11',
                   ],
      install_requires=['numpy>=1.18.5',
                        'scipy>=1.9.0',
                        'pandas',
                        'delicatessen',
                        'matplotlib',
                        ],
      long_description=descript,
      long_description_content_type="text/markdown",
      )
