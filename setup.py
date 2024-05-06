# *----------------------------------------------------------------------------*
# * Copyright (C) 2024 IBM Inc. All rights reserved                            *
# * SPDX-License-Identifier: GPL-3.0-only                                      *
# *----------------------------------------------------------------------------*

from setuptools import setup, find_packages

setup(
    name="arlc",
    version="1.0.0",
    description="Abductive Rule Learner with Context-awareness",
    url="https://research.ibm.com/people/giacomo-camposampiero--1",
    author="Giacomo Camposampiero",
    author_email="giacomo.camposampiero1@ibm.com",
    license="GPL-3.0",
    packages=find_packages(
        where="arlc",
    ),
    include_package_data=True,
    zip_safe=False,
)
