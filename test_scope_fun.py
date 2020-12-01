# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 22:59:38 2020

@author: dolor
"""

def test_scope_fun(zata, othervar):
    zata[0][0] = 'Hello'
    zata[0][1] = 'there'
    othervar[0] = 6
    return othervar