from __future__ import division
try:
    import setuptools
except:
    print ('''
setuptools not found.

On linux, the package is often called python-setuptools''')
    from sys import exit
    exit(1)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name='LexiClean',
    version='1.0.6',
    author='Mohamed Edderyouch',
    author_email='mohamededderyouch5@gmail.com',
    maintainer= 'Zineb Boughedda',
    license='Academic Free License',
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    description='Text Preprocessing package for NLP projects',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['nltk', 'pandas', 'tqdm','SpellChecker','IPython', 'textblob'],
    url='https://github.com/simoderyouch',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

)