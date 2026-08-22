# Copyright 2018 Christoph Heindl.
# 
# Licensed under MIT License
# ============================================================

from datetime import datetime
import tensorflow as tf
import numpy as np
import os

import tf_matplotlib as tfmpl

if __name__ == '__main__':
    @tfmpl.figure_tensor
    def draw_scatter(scaled, colors): 
        '''Draw scatter plots. One for each color.'''  
        figs = tfmpl.create_figures(len(colors), figsize=(4,4))
        for idx, f in enumerate(figs):
            ax = f.add_subplot(111)
            ax.axis('off')
            ax.scatter(scaled[:, 0], scaled[:, 1], c=colors[idx])
            f.tight_layout()

        return figs  

    points = tf.random.normal((100, 2), dtype=tf.float32)
    scale = tf.constant(2., dtype=tf.float32)        
    scaled = points*scale
   
    os.makedirs('log', exist_ok=True)
    now = datetime.now()
    logdir = "log/" + now.strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.create_file_writer(logdir)
    with writer.as_default():
        image_tensor = draw_scatter(scaled, ['r', 'g'])
        image_summary = tf.summary.image('scatter', image_tensor, step=0)
        writer.flush()
