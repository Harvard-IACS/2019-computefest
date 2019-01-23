import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
import pandas as pd
import imageio
import skimage

def display_manifold(decoder, height, width, base_vec, bound_x=15, bound_y=15, axis_x=0, axis_y=1, n=15,
                     desc_x = 'x', desc_y = 'y', file_out=None):
    '''Varies up to two dimensions of the latent representation, and visualizes its effect.

    This function can be used in one or two dimensions. To just vary a single dimension, set
    either bound_x or bound_y to zero.

    Args:
        decoder: The keras decoder model to use.
        height: The image height the decoder produces.
        width: The image width the decoder produces.
        base_vec: The basic latent representation to which changes should be applied.
            Per convention, the first entries in base_vec correspond to the latent variables,
            followed by variables we condition on (if any). Therefore, dimension is the sum of
            the latent dimension and the conditioning dimension.
        bound_x: The range that the values on axis_x will be modified to.
        bound_y: The range that the values on axis_y will be modified to.
        axis_x: The first axis to modify. Must be 0 <= axis_x <= len(base_vec).
        axis_y: The first axis to modify. Must be 0 <= axis_y <= len(base_vec).
        n: The number of columns/rows to generate. Thus, in total, n**2 images will be generated
            if two dimensions are modified. Otherwise, just n images will be generated.
        desc_x: The caption of the x-axis shown on the plot.
        desc_y: The caption of the y-axis shown on the plot.
        file_out: File path if the resulting plot should be saved. Can be None.

    Returns:
        Results will be plotted. In addition, a tuple is returned, containing both the grid as
        color image, as well as a list of the individual images generated (row-wise).
    '''
    figure = np.zeros((height * (n if bound_y > 0 else 1), width * (n if bound_x > 0 else 1), 3))
    grid_x = np.linspace(-bound_x, bound_x, n) if bound_x > 0 else [0]
    grid_y = np.linspace(-bound_y, bound_y, n) if bound_y > 0 else [0]
    individual_outputs = []

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = base_vec.copy()
            z_sample[axis_x] = xi # SD is 1
            z_sample[axis_y] = yi # SD is 1

            x_decoded = decoder.predict(np.expand_dims(z_sample, axis=0))
            sample = np.clip(x_decoded[0], 0, 1)
            figure[i * height: (i + 1) * height, j * width: (j + 1) * width] = sample
            individual_outputs.append(sample)

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.xlabel(desc_x)
    plt.ylabel(desc_y)
    if file_out is not None:
        plt.savefig(file_out, dpi=200, bbox_inches='tight')
    return figure, individual_outputs

def generate_gif(decoder, height, width, base_vec, axis, total_frames, degree, file_out):
    """Generates an animated GIF, showing impact of perturbing a single axis.

    Args:
        decoder: The keras decoder model to use.
        height: The image height the decoder produces.
        width: The image width the decoder produces.
        base_vec: The basic latent representation to which changes should be applied.
            Per convention, the first entries in base_vec correspond to the latent variables,
            followed by variables we condition on (if any). Therefore, dimension is the sum of
            the latent dimension and the conditioning dimension.
        axis: The axis to modify. Must be 0 <= axis <= len(base_vec).
        total_frames: The total number of frames to generate.
        degree: The extent to which the values in the given axis should be perturbed.
        file_out: File path to indicate where the GIF should be stored.
    """
    if base_vec.ndim > 1:
        base_vec = base_vec[0]
    _, image_seq = display_manifold(decoder, height, width, base_vec,
                                    bound_x=degree,
                                    bound_y=0,
                                    axis_x=axis,
                                    n=total_frames)
    image_seq = [(img * 255.).astype(np.uint8) for img in image_seq]
    imageio.mimsave(file_out, image_seq)

def load_mnist(target_height=64, target_width=64):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    def preprocess_mnist(df, target_height, target_width):
        '''Preprocesses the MNIST data by downsampling and scaling.

        Args:
            df: The original input data for MNIST.
            height: The target height.
            width: The target width.

        Returns:
            The preprocessed MNIST data.
        '''
        df = [skimage.transform.resize(x, (target_height, target_width)).astype(np.float32) for x in df]
        df = np.array(df)
        df = np.expand_dims(df, axis=3)
        df = np.repeat(df, axis=3, repeats=3)
        print('details: shape', df.shape, 'min', df.min(), 'max', df.max())
        return df

    x_train = preprocess_mnist(x_train, target_height, target_width)
    x_test = preprocess_mnist(x_test, target_height, target_width)
    return x_train, y_train, x_test, y_test

def load_celeba(file):
    # Open annotation.
    with open(file, 'r') as f:
        lines = f.readlines()
    columns = lines[1].split(' ')[:-1]
    columns.insert(0, 'Filename')

    matrix = []
    for i in range(2, len(lines)):
        data = lines[i].split(' ')
        data = [d.replace('\n', '') for d in data if len(d) > 0]
        matrix.append(data)

    df = pd.DataFrame(np.array(matrix), columns=columns)
    return df
