import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def matrix_downsizer(matrix, block_size = (20,20,20)):

    # Calculate the padding needed to make each dimension a multiple of 20
    pad_x = (block_size[0] - matrix.shape[0] % block_size[0]) % block_size[0]
    pad_y = (block_size[1] - matrix.shape[1] % block_size[1]) % block_size[1]
    pad_z = (block_size[2] - matrix.shape[2] % block_size[2]) % block_size[2]

    # Pad the matrix with zeros to make its dimensions multiples of 20
    padded_matrix = np.pad(matrix, ((0, pad_x), (0, pad_y), (0, pad_z)), mode='constant', constant_values=0)

    # Reshape the padded matrix into blocks of 20x20x20
    reshaped_matrix = padded_matrix.reshape(
        padded_matrix.shape[0] // block_size[0], block_size[0],
        padded_matrix.shape[1] // block_size[1], block_size[1],
        padded_matrix.shape[2] // block_size[2], block_size[2]
    )

    # Take the mean across the 20x20x20 blocks
    downsized_matrix = reshaped_matrix.mean(axis=(1, 3, 5))

    return downsized_matrix


def plane_mask(matrix_shape = (100,100,100), points=None, chosen_axis = 0):
    #point1 = np.array([44, 26, 35])*20//block_size[0]
    #point2 = np.array([25, 30, 10])*20//block_size[1]
    #point3 = np.array([1, 23, 20])*20//block_size[2]
    #points = (point1,point2,point3)
    point1, point2, point3 = points


    # Get the dimensions for the axes perpendicular to the chosen one
    if chosen_axis == 0:
        shape_other_axes = (matrix_shape[1], matrix_shape[2])
        axis1, axis2 = 1, 2
    elif chosen_axis == 1:
        shape_other_axes = (matrix_shape[0], matrix_shape[2])
        axis1, axis2 = 0, 2
    else:
        shape_other_axes = (matrix_shape[0], matrix_shape[1])
        axis1, axis2 = 0, 1

    # Create a meshgrid over the indices of the two perpendicular axes
    idx1, idx2 = np.meshgrid(np.arange(shape_other_axes[0]), np.arange(shape_other_axes[1]), indexing='ij')

    # Define the plane equation based on the three points
    # Plane equation: ax + by + cz = d
    v1 = point2 - point1
    v2 = point3 - point1
    normal = np.cross(v1, v2)
    a, b, c = normal
    d = np.dot(normal, point1)

    # Solve for the chosen axis coordinate as a function of the other two coordinates
    # chosen_axis_coord = (d - a*idx1 - b*idx2) / c
    if chosen_axis == 0:
        coord_plane = (d - b * idx1 - c * idx2) / a
    elif chosen_axis == 1:
        coord_plane = (d - a * idx1 - c * idx2) / b
    else:
        coord_plane = (d - a * idx1 - b * idx2) / c

    # Round to get the nearest integer index along the chosen axis
    coord_plane = np.round(coord_plane).astype(int)

    # Ensure the calculated indices are within bounds
    coord_plane = np.clip(coord_plane, 0, matrix_shape[chosen_axis] - 1)

    # Construct the mask
    mask = np.zeros(matrix_shape, dtype=bool)
    if chosen_axis == 0:
        mask[coord_plane, idx1, idx2] = True
    elif chosen_axis == 1:
        mask[idx1, coord_plane, idx2] = True
    else:
        mask[idx1, idx2, coord_plane] = True

    return mask, shape_other_axes


def array_normalizer(array):
    array = array - np.min(array)
    array = array/np.max(array)
    return array

def render_3d(matrix_d, isomin = 0.5, isomax = 1.0, opacity = 0.1,plot_plane = False, plane = 0,opacity_plane = 0.05):
    matrix_d = matrix_d + np.min(matrix_d)
    matrix_d = matrix_d/np.max(matrix_d)


    if plot_plane:
        data = [go.Volume(
        x=np.repeat(np.arange(matrix_d.shape[0]), matrix_d.shape[1] * matrix_d.shape[2]),
        y=np.tile(np.repeat(np.arange(matrix_d.shape[1]), matrix_d.shape[2]), matrix_d.shape[0]),
        z=np.tile(np.arange(matrix_d.shape[2]), matrix_d.shape[0] * matrix_d.shape[1]),
        value=matrix_d.flatten(),  # Flatten the matrix to get values
        opacity=opacity,  # Lower opacity for a better 3D effect
        isomin=isomin,   # Minimum threshold for volume rendering
        isomax=isomax,   # Maximum threshold for volume rendering
        surface_count=15,  # Number of surfaces in the volume rendering
        colorscale="Viridis"),
        go.Volume(
            x=np.repeat(np.arange(plane.shape[0]), plane.shape[1] * plane.shape[2]),
            y=np.tile(np.repeat(np.arange(plane.shape[1]), plane.shape[2]), plane.shape[0]),
            z=np.tile(np.arange(plane.shape[2]), plane.shape[0] * plane.shape[1]),
            value=plane.flatten(),  # Flatten the matrix to get values
            opacity=opacity_plane,  # Lower opacity for a better 3D effect
            isomin=isomin,   # Minimum threshold for volume rendering
            isomax=isomax,   # Maximum threshold for volume rendering
            surface_count=15,  # Number of surfaces in the volume rendering
            colorscale="thermal"
        )
    ]
    else: 
        data = go.Volume(
        x=np.repeat(np.arange(matrix_d.shape[0]), matrix_d.shape[1] * matrix_d.shape[2]),
        y=np.tile(np.repeat(np.arange(matrix_d.shape[1]), matrix_d.shape[2]), matrix_d.shape[0]),
        z=np.tile(np.arange(matrix_d.shape[2]), matrix_d.shape[0] * matrix_d.shape[1]),
        value=matrix_d.flatten(),  # Flatten the matrix to get values
        opacity=opacity,  # Lower opacity for a better 3D effect
        isomin=isomin,   # Minimum threshold for volume rendering
        isomax=isomax,   # Maximum threshold for volume rendering
        surface_count=15,  # Number of surfaces in the volume rendering
        colorscale="Viridis")

    fig = go.Figure(data=data)

    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[0, matrix_d.shape[0]], title='X Axis'),
            yaxis=dict(nticks=4, range=[0, matrix_d.shape[1]], title='Y Axis'),
            zaxis=dict(nticks=4, range=[0, matrix_d.shape[2]], title='Z Axis'),
            aspectmode='manual',  # Set manual aspect ratio
            aspectratio=dict(
                x=matrix_d.shape[0] / matrix_d.shape[2],
                y=matrix_d.shape[1] / matrix_d.shape[2],
                z=1  # Use 1 as the reference dimension for scaling
            )
        ),
        title="3D rendering"
    )

    fig.show()


def draw_circle(image, center, radius):
    """Draw a filled circle on the image."""
    y, x = np.ogrid[-center[1]:image.shape[0]-center[1], -center[0]:image.shape[1]-center[0]]
    mask = x**2 + y**2 <= radius**2
    image[mask] = 1  # Set pixel values to 1 for the circle

def draw_square(image, center, size, angle):
    """Draw a filled square on the image, rotated by a given angle."""
    half_size = size / 2
    # Create a grid of points
    y, x = np.mgrid[-half_size:half_size:1, -half_size:half_size:1]
    
    # Rotate the points
    cos_angle = np.cos(np.radians(angle))
    sin_angle = np.sin(np.radians(angle))
    x_rotated = cos_angle * x - sin_angle * y
    y_rotated = sin_angle * x + cos_angle * y

    # Translate the square to the center
    x_translated = x_rotated + center[0]
    y_translated = y_rotated + center[1]

    # Set pixel values to 1 for the square
    for xi, yi in zip(x_translated.flatten(), y_translated.flatten()):
        if 0 <= xi < image.shape[1] and 0 <= yi < image.shape[0]:
            image[int(yi), int(xi)] = 1

def create_phantom_image(image_size, circles, squares):
    """
    Create a 2D phantom image with specified circles and squares.

    :param image_size: Tuple (width, height) of the image.
    :param circles: List of tuples (x, y, radius) for each circle.
    :param squares: List of tuples (x, y, size, angle) for each square.
    :return: A matrix representing the phantom image.
    """
    # Create an empty image
    image = np.zeros(image_size, dtype=np.float32)

    # Draw circles on the image
    for (x, y, radius) in circles:
        draw_circle(image, (x, y), radius)

    # Draw squares on the image
    for (x, y, size, angle) in squares:
        draw_square(image, (x, y), size, angle)

    return image


def pad_matrix(matrix, target_shape):
    """
    Pad a matrix to the specified target shape with a specified padding value.

    :param matrix: The input 2D matrix (numpy array).
    :param target_shape: A tuple (target_height, target_width) specifying the desired dimensions.
    :param pad_value: The value to use for padding.
    :return: The padded matrix.
    """
    # Get the current shape of the matrix
    current_shape = matrix.shape
    pad_height = target_shape[0] - current_shape[0]
    pad_width = target_shape[1] - current_shape[1]

    # Check for negative padding sizes
    if pad_height < 0 or pad_width < 0:
        raise ValueError("Target shape must be greater than or equal to the current shape.")

    # Calculate padding sizes for each dimension
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Pad the matrix using numpy's pad function
    padded_matrix = np.pad(matrix, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')

    return padded_matrix


def transform_grid(N, M, d, angle, translation):
    """
    Transform a grid of coordinates by squishing, rotating, and translating.

    :param N: Maximum x-coordinate.
    :param M: Maximum y-coordinate.
    :param d: Squishing factor.
    :param angle: Rotation angle in degrees.
    :param translation: A tuple (tx, ty) for translation.
    :return: Transformed x and y coordinates.
    """
    # Generate a grid of coordinates
    x = np.linspace(0, N-1, num=N)  # x-coordinates from 0 to N
    y = np.linspace(0, M-1, num=M)  # y-coordinates from 0 to M
    X, Y = np.meshgrid(x, y)  # Create a grid

    # Flatten the grid for transformation
    coordinates = np.vstack((X.flatten(), Y.flatten()))

    # 1. Squish the coordinates by a factor of d
    coordinates[0] *= d  # Apply squish factor to x-coordinates
    coordinates[1] *= d  # Apply squish factor to y-coordinates

    # 2. Rotate the coordinates
    theta = np.radians(angle)  # Convert angle to radians
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
    
    rotated_coordinates = rotation_matrix @ coordinates

    # 3. Translate the coordinates
    tx, ty = translation
    translated_coordinates = rotated_coordinates + np.array([[tx], [ty]])

    # Reshape back to grid format
    transformed_X = translated_coordinates[0].reshape(X.shape)
    transformed_Y = translated_coordinates[1].reshape(Y.shape)

    return transformed_X, transformed_Y