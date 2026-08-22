# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================

from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import cm
from datetime import datetime
import tensorflow as tf
import sonnet as snt
import numpy as np
import os

import tf_matplotlib as tfmpl

if __name__ == '__main__':

    def beale(x, y):
        '''Beale surface for optimization tests.'''
        with tf.name_scope('beale'):
            return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
            
    # List of optimizers to compare
    optimizers = [
        (tf.keras.optimizers.SGD(1e-3), 'SGD'),
        (tf.keras.optimizers.Adagrad(1e-1), 'Adagrad'),
        (tf.keras.optimizers.Adadelta(1e2), 'Adadelta'),
        (tf.keras.optimizers.Adam(1e-1), 'Adam'),            
    ]

    paths = []        
    history = []

    def init_fig(*args, **kwargs):
        '''Initialize figures.'''
        fig = tfmpl.create_figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d', elev=50, azim=-30)
        ax.xaxis.set_pane_color((1.0,1.0,1.0,1.0))
        ax.yaxis.set_pane_color((1.0,1.0,1.0,1.0))
        ax.zaxis.set_pane_color((1.0,1.0,1.0,1.0))
        ax.set_title('Gradient descent on Beale surface')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('beale($x$,$y$)')
    
        xx, yy = np.meshgrid(np.linspace(-4.5, 4.5, 40), np.linspace(-4.5, 4.5, 40))
        zz = beale(xx, yy)
        ax.plot_surface(xx, yy, zz, norm=LogNorm(), rstride=1, cstride=1, edgecolor='none', alpha=.8, cmap=cm.jet)
        ax.plot([3], [.5], [beale(3, .5)], 'k*', markersize=5)
        
        for o in optimizers:
            path, = ax.plot([],[],[], label=o[1])
            paths.append(path)

        ax.legend(loc='upper left')
        fig.tight_layout()

        return fig, paths
        
    @tfmpl.blittable_figure_tensor(init_func=init_fig)
    def draw(xy, z):
        '''Updates paths for each optimizer.'''
        history.append(np.c_[xy, z])
        xyz = np.stack(history) #NxMx3

        for path_index in range(xyz.shape[1]):
            path = paths[path_index]
            path.set_data(xyz[:, path_index, 0], xyz[:, path_index, 1])
            path.set_3d_properties(xyz[:, path_index, 2])
    
        return paths

    # Create variables for each optimizer
    
    xys = [tf.Variable([3., 4.], dtype=tf.float32, name=f'x_{o[1]}') for o in optimizers]        
    train = []

    # Alloc summary writer
    os.makedirs('log', exist_ok=True)
    now = datetime.now()
    logdir = "log/" + now.strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.create_file_writer(logdir)

    # Run optimization, write summary every now and then.
    for i in range(200):
        with tf.GradientTape(persistent=True) as tape:
            zs = [beale(xy[0], xy[1]) for xy in xys]
            
        grads = [tape.gradient(z, xy) for z, xy in zip(zs, xys)]
        clipped = [tf.clip_by_value(g, -10, 10) for g in grads]
        for (opt, name), g, xy in zip(optimizers, clipped, xys):        
            train.append(opt.apply_gradients([(g, xy)]))

        if i % 10 == 0:
            # Generate summary
            image_tensor = draw(tf.stack(xys), tf.stack(zs))
            with writer.as_default():
                tf.summary.image('optimization', image_tensor, step=i)        

    print(len(f"len(train) => {len(train)}"))