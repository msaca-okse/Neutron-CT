import vtk
import numpy as np
import imageio



def generate_volume_data(shape=(50, 50, 50)):
    x, y, z = np.indices(shape)
    sphere = (x - 25) ** 2 + (y - 25) ** 2 + (z - 25) ** 2 < 20**2
    return (sphere * 255).astype(np.uint8)

def numpy_to_vtk(volume_data):
    depth, height, width = volume_data.shape
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(width, height, depth)
    vtk_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    flat_data = volume_data.ravel(order='F')
    vtk_array = vtk.vtkUnsignedCharArray()
    vtk_array.SetNumberOfComponents(1)
    vtk_array.SetArray(flat_data, flat_data.size, 1)

    vtk_data.GetPointData().SetScalars(vtk_array)
    return vtk_data

def create_volume_renderer(volume_data):
    vtk_data = numpy_to_vtk(volume_data)

    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(vtk_data)

    volume_property = vtk.vtkVolumeProperty()
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()

    opacity_func = vtk.vtkPiecewiseFunction()
    opacity_func.AddPoint(0, 0.0)
    opacity_func.AddPoint(255, 1.0)
    volume_property.SetScalarOpacity(opacity_func)

    color_func = vtk.vtkColorTransferFunction()
    color_func.AddRGBPoint(0, 0, 0, 0)
    color_func.AddRGBPoint(255, 1, 1, 1)
    volume_property.SetColor(color_func)

    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    return volume

def animate_camera(volume, output_file="vtk_animation.mp4", frames=100):
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    print(1)
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    print(2)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.1, 0.1)
    camera = renderer.GetActiveCamera()
    camera.SetPosition(100, 100, 100)
    camera.SetFocalPoint(25, 25, 25)
    print(3)
    render_window.Render()  # Ensure VTK rendering pipeline is set up
    render_window.SetOffScreenRendering(1)
    print(4)
    # Create the image filter once before the loop
    window_to_image = vtk.vtkWindowToImageFilter()
    window_to_image.SetInput(render_window)
    window_to_image.SetScale(1)  # Adjust scale if needed
    window_to_image.SetReadFrontBuffer(False)  # Important for some systems
    window_to_image.SetInputBufferTypeToRGB()
    window_to_image.Update()
    print(5)
    writer = imageio.get_writer(output_file, fps=30)

    for i in range(frames):
        print(6, i)
        camera.Azimuth(3.6)  # Rotate camera
        renderer.ResetCameraClippingRange()
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

print('hello')
volume_data = generate_volume_data()
print('hi')
volume = create_volume_renderer(volume_data)
print('med dig ')
animate_camera(volume, output_file = '/dtu-compute/msaca/visualizations/example1.mp4')