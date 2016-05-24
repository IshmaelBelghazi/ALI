from setuptools import find_packages, setup

setup(
    name='ali',
    version='0.1.0',
    description='Code for the "Adversarially Learned Inference" paper',
    long_description='Code for the "Adversarially Learned Inference" paper',
    url='https://github.com/IshmaelBelghazi/ALI',
    author='Vincent Dumoulin, Ishmael Belghazi',
    license='MIT',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
    ],
    keywords='theano blocks machine learning neural networks deep learning',
    packages=find_packages(exclude=['scripts', 'experiments']),
    install_requires=['numpy', 'theano', 'blocks', 'fuel'],
    zip_safe=False)
