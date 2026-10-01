#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 08:33:15 2026

@author: kathrin

Port for definitions - centralized to avoid dual (ore more) defintions

"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures"