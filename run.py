#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:52:12 2021

@author: Bonfils Anthony
"""

from api import app

if __name__ == "__main__":
    app.run(debug=True)
    # app.run(debug=True, host='127.0.0.1', port=5001)
