#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 08:35:20 2022

@author: maksim, nick
"""

default_stop_phrases = [
    'All rights reserved.',
    r'(?=Published by).*\.',  # remove statements with Published by
    r'(?=Approved for unlimited, public release).*\.',
    r'(?=Prepared by LLNL under).*\.',
    r'(?=Copyright (19|20)\d{2} by).*\.',  # remove 'Copyright YEAR by' statements
]
