import vtk
import numpy as np
import imageio

def generate_volume_data(shape=(1000, 1000, 1000)):
    # Create meshgrid (indices for x, y, and z)
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    
    # Create a spherical mask where the center of the sphere is at (shape[0]/2, shape[1]/2, shape[2]/2)
    sphere = (x - shape[0] / 2) ** 2 + 1.5 * (y - shape[1] / 2) ** 2 + 0.9 * (z - shape[2] / 2) ** 2 < (shape[0] / 3) ** 2
    
    # Convert the boolean mask to uint8 (0 or 255)
    return (sphere * 255).astype(np.uint8)

def numpy_to_vtk(volume_data):
    depth, height, width = volume_data.shape
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(width, height, depth)
    vtk_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    flat_data = volume_data.ravel(order='C')
    vtk_array = vtk.vtkUnsignedCharArray()
    vtk_array.SetNumberOfComponents(1)
    vtk_array.SetArray(flat_data, flat_data.size, 1)

    vtk_data.GetPointData().SetScalars(vtk_array)
    return vtk_data

def create_volume_renderer(volume_data):
    vtk_data = numpy_to_vtk(volume_data)

    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(vtk_data)


    clip_plane = vtk.vtkPlane()
    clip_plane.SetOrigin(200, 0, 0)  # Cut at X = 250
    clip_plane.SetNormal(-1, 0, 0)  # Clip the negative side (keep X >= 250)
    volume_mapper.AddClippingPlane(clip_plane)

    volume_property = vtk.vtkVolumeProperty()
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()

    opacity_func = vtk.vtkPiecewiseFunction()
    opacity_func.AddPoint(0, 0.0)   # Completely transparent
    opacity_func.AddPoint(0, 0.0)   # Transparent for low values
    opacity_func.AddPoint(255, 0.5) # Fully opaque for high values
    volume_property.SetScalarOpacity(opacity_func)

    color_func = vtk.vtkColorTransferFunction()
    color_func.AddRGBPoint(0, 0, 0, 0)  # Black for background
    color_func.AddRGBPoint(128, 1, 0, 0)  # Red for mid intensity
    color_func.AddRGBPoint(255, 0, 1, 0)  # Green for highest intensity
    volume_property.SetColor(color_func)

    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    return volume

def animate_camera(volume, output_file="vtk_animation.mp4", frames=100):
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1808, 1200)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    renderer.AddVolume(volume)
    renderer.SetBackground(0.3, 0.3, 0.3)

    # Adjust camera position and focal point to ensure the whole volume is visible
    camera = renderer.GetActiveCamera()
    camera.SetPosition(1000, 1000, 1000)  # Adjust to view from a better angle
    camera.SetFocalPoint(300, 300, 300)  # Center focal point in the volume
    camera.SetViewUp(0,0 , 1)  # Make sure the "up" direction is aligned
    render_window.Render()  # Ensure VTK rendering pipeline is set up

    # Set near and far clipping to include full volume
    camera.SetClippingRange(1, 2000)

    # Get the volume mapper and retrieve the existing clipping plane
    volume_mapper = volume.GetMapper()
    planes = volume_mapper.GetClippingPlanes()  # Get existing planes
    if planes.GetNumberOfItems() > 0:
        clipping_plane = planes.GetItem(0)  # Use the first clipping plane
    else:
        print("Warning: No clipping plane found!")

    render_window.SetOffScreenRendering(1)

    # Create the image filter once before the loop
    window_to_image = vtk.vtkWindowToImageFilter()
    window_to_image.SetInput(render_window)
    window_to_image.SetScale(1)  # Adjust scale if needed
    window_to_image.SetReadFrontBuffer(False)  # Important for some systems
    window_to_image.SetInputBufferTypeToRGB()
    window_to_image.Update()

    # Create the video writer
    writer = imageio.get_writer(output_file, format='ffmpeg', fps=50, codec='mpeg4', quality=10)

    for i in range(frames):
        if clipping_plane:
            new_position = 200 + (i * 1.5)  # Move from slice 250 onward
            clipping_plane.SetOrigin(new_position, 0, 0)  # Shift the plane
            clipping_plane.Modified()  # Notify VTK of the change

        # Rotate the camera for a smooth animation
        camera.Azimuth(1.3)  # Rotate camera slightly every frame
        camera.Dolly(0.999)  # Gradual zoom effect

        render_window.Render()  # Update the render

        # Capture frame
        window_to_image.Modified()  # Ensure the filter updates
        window_to_image.Update()

        vtk_image = window_to_image.GetOutput()
        width, height, _ = vtk_image.GetDimensions()
        vtk_array = vtk.vtkUnsignedCharArray.SafeDownCast(vtk_image.GetPointData().GetScalars())

        if vtk_array:
            img_data = np.frombuffer(vtk_array, dtype=np.uint8).reshape((height, width, 3))
            writer.append_data(img_data)
        else:
            print("Warning: Could not extract image data")

    writer.close()
    print(f"Animation saved to {output_file}")

volume_data = generate_volume_data(shape=(1000, 1000, 1000))

volume = create_volume_renderer(volume_data)
animate_camera(volume, output_file='/dtu-compute/msaca/visualizations/example2.mp4', frames=1000)
