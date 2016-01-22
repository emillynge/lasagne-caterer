from distutils.core import setup
import sys
if sys.version_info < (3, 4):
    print("The use of metaclasses in this package is not supported below 3.4\nPlease upgrade")
    sys.exit(1)

setup(
        name='lasagnecaterer',
        version='0.1',
        packages=['lasagnecaterer'],
        url='https://github.com/emillynge/lasagne-caterer',
        license='GNU GENERAL PUBLIC LICENSE  Version 3',
        author='Emil Sauer Lynge',
        author_email='',
        description='plug n play lasagne models',
        classifiers=[
                        'Intended Audience :: Developers',
                        'License :: OSI Approved :: GPL v3',
                        'Programming Language :: Python :: 3.4',
                        'Programming Language :: Python :: 3.5',
                ], requires=['lasagne', 'theano']
)

