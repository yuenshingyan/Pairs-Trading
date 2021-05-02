#!/usr/bin/env python
# coding: utf-8

# In[1]:


import setuptools

with open('C:/Users/Hindy/Desktop/Container_pairs_trading/README.md', 'r') as fh:
    long_descripion = fh.read()
    
setuptools.setup(
    name='Simeple_PairsTrading', 
    version='1.0.0',
    author='Hindy Yuen', 
    author_email='hindy888@hotmail.com', 
    description='Pairs Trading', 
    long_description=long_descripion, 
    long_description_content_type='text/markdown', 
    url='https://github.com/HindyDS/Pairs-Trading', 
    packages=setuptools.find_packages(), 
    classifiers=[
        'Programming Language :: Python :: 3', 
        'License :: OSI Approved :: MIT License', 
        'Operating System :: OS Independent', 
    ], 
    python_requires='>=3.6', 
)

