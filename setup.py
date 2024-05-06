from setuptools import setup, find_packages

setup(
  name = 'flowmatching-bdt',
  packages = find_packages(exclude=['assets']),
  version = '0.1.0',
  license='MIT',
  description = 'Flow Matching with BDTs',
  long_description_content_type = 'text/markdown',
  author = 'Radi Radev',
  author_email = 'radi.radev.uk@gmail.com',
  url = 'https://github.com/radiradev/flowmatching-bdt',
  keywords = [
    'artificial intelligence',
    'flow matching',
    'xgboost',
  ],
  install_requires=[
    'xgboost>=2.0.0',
    'scikit-learn>=1.3',
    'tqdm>=4.6',
    'tqdm_joblib>=0.0.3',
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    "joblib==1.3.0",
    "scikit-learn==1.3.2",
    "tqdm==4.66.2",
    "tqdm_joblib==0.0.3",
    "xgboost==2.0.0"
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.11',
  ],
)