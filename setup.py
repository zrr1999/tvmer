from setuptools import setup, find_packages
import sys

# with open("README.md", "r", encoding='UTF-8') as fh:
#     long_description = fh.read()

setup(
    name="tvmer",
    version="0.1.0",
    keywords=[],
    description="",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    license="GPL-3.0 Licence",
    # url="https://github.com/zrr1999/tvmer",
    author="zrr1999",
    author_email="2742392377@qq.com",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    platforms="any",
    install_requires=['tvm'],
    scripts=[]
)
