from __future__ import print_function
from setuptools import setup,find_packages
setup(
    name='njuseg',
    version='2.0',
    author='Jiahuan Li, Tong Pu, Huiyun Yang, Yu Bao',
    author_email='lijh@nlp.nju.edu.cn',
    description='Chinese Word Segmenter developed by Nanjing University NLP Group',
    long_description=open("README.rst").read(),
    license="MIT",
    url="",
    packages=find_packages(),
    install_requires=["torchtext >= 0.3.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",]
)
