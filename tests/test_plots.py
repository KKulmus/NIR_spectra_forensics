#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:48:48 2026

@author: kathrin
"""

import unittest

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from nir_forensics.plots import plot_confusion_matrix_heatmap

class TestSplit(unittest.TestCase):
    
    def test_plot_confusion_matrix_heatmap(self):
        """Test whether plot_confusion_matrix_heatmap returns a Figure,
            reuses an existing Figure when ax is provided, and does not
            create additional Axes objects (total 2 axes, since colorbar ax is created 
                                            by sns.heatmap).
        """
        
        conf_matrix = np.eye(8)
        fig = plot_confusion_matrix_heatmap(conf_matrix, "dummy instrument")
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        plt.close(fig)
        fig, ax = plt.subplots()
        fig_plot = plot_confusion_matrix_heatmap(conf_matrix, "dummy instrument", ax = ax)
        self.assertIs(fig,fig_plot)
        self.assertEqual(len(fig.axes), 2)
        plt.close(fig)
        
        