# trace generated using paraview version 5.9.0

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'Xdmf3ReaderS'
c_file_name = input("Absolute path of c.xdmf: ")
cxdmf = Xdmf3ReaderS(registrationName='c.xdmf', FileName=[c_file_name])
cxdmf.PointArrays = ['c']

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# create a new 'Xdmf3ReaderS'
phi_file_name = input("Absolute path of phi.xdmf: ")
phixdmf = Xdmf3ReaderS(registrationName='phi.xdmf', FileName=[phi_file_name])
phixdmf.PointArrays = ['phi']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
phixdmfDisplay = Show(phixdmf, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'phi'
phiLUT = GetColorTransferFunction('phi')
phiLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'phi'
phiPWF = GetOpacityTransferFunction('phi')
phiPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
phixdmfDisplay.Representation = 'Surface'
phixdmfDisplay.ColorArrayName = ['POINTS', 'phi']
phixdmfDisplay.LookupTable = phiLUT
phixdmfDisplay.SelectTCoordArray = 'None'
phixdmfDisplay.SelectNormalArray = 'None'
phixdmfDisplay.SelectTangentArray = 'None'
phixdmfDisplay.OSPRayScaleArray = 'phi'
phixdmfDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
phixdmfDisplay.SelectOrientationVectors = 'None'
phixdmfDisplay.ScaleFactor = 0.1700000047683716
phixdmfDisplay.SelectScaleArray = 'phi'
phixdmfDisplay.GlyphType = 'Arrow'
phixdmfDisplay.GlyphTableIndexArray = 'phi'
phixdmfDisplay.GaussianRadius = 0.008500000238418579
phixdmfDisplay.SetScaleArray = ['POINTS', 'phi']
phixdmfDisplay.ScaleTransferFunction = 'PiecewiseFunction'
phixdmfDisplay.OpacityArray = ['POINTS', 'phi']
phixdmfDisplay.OpacityTransferFunction = 'PiecewiseFunction'
phixdmfDisplay.DataAxesGrid = 'GridAxesRepresentation'
phixdmfDisplay.PolarAxes = 'PolarAxesRepresentation'
phixdmfDisplay.ScalarOpacityFunction = phiPWF
phixdmfDisplay.ScalarOpacityUnitDistance = 0.008383798682624784
phixdmfDisplay.OpacityArrayName = ['POINTS', 'phi']

# reset view to fit data
renderView1.ResetCamera()

# get the material library
materialLibrary1 = GetMaterialLibrary()

# show color bar/color legend
phixdmfDisplay.SetScalarBarVisibility(renderView1, True)

# show data in view
cxdmfDisplay = Show(cxdmf, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'c'
cLUT = GetColorTransferFunction('c')
cLUT.RGBPoints = [-1.0, 0.231373, 0.298039, 0.752941, -0.0048223137855529785, 0.865003, 0.865003, 0.865003, 0.990355372428894, 0.705882, 0.0156863, 0.14902]
cLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'c'
cPWF = GetOpacityTransferFunction('c')
cPWF.Points = [-1.0, 0.0, 0.5, 0.0, 0.990355372428894, 1.0, 0.5, 0.0]
cPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
cxdmfDisplay.Representation = 'Surface'
cxdmfDisplay.ColorArrayName = ['POINTS', 'c']
cxdmfDisplay.LookupTable = cLUT
cxdmfDisplay.SelectTCoordArray = 'None'
cxdmfDisplay.SelectNormalArray = 'None'
cxdmfDisplay.SelectTangentArray = 'None'
cxdmfDisplay.OSPRayScaleArray = 'c'
cxdmfDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
cxdmfDisplay.SelectOrientationVectors = 'None'
cxdmfDisplay.ScaleFactor = 0.1700000047683716
cxdmfDisplay.SelectScaleArray = 'c'
cxdmfDisplay.GlyphType = 'Arrow'
cxdmfDisplay.GlyphTableIndexArray = 'c'
cxdmfDisplay.GaussianRadius = 0.008500000238418579
cxdmfDisplay.SetScaleArray = ['POINTS', 'c']
cxdmfDisplay.ScaleTransferFunction = 'PiecewiseFunction'
cxdmfDisplay.OpacityArray = ['POINTS', 'c']
cxdmfDisplay.OpacityTransferFunction = 'PiecewiseFunction'
cxdmfDisplay.DataAxesGrid = 'GridAxesRepresentation'
cxdmfDisplay.PolarAxes = 'PolarAxesRepresentation'
cxdmfDisplay.ScalarOpacityFunction = cPWF
cxdmfDisplay.ScalarOpacityUnitDistance = 0.008383798682624784
cxdmfDisplay.OpacityArrayName = ['POINTS', 'c']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
cxdmfDisplay.ScaleTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 0.990355372428894, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
cxdmfDisplay.OpacityTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 0.990355372428894, 1.0, 0.5, 0.0]

# show color bar/color legend
cxdmfDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=phixdmf)
contour1.ContourBy = ['POINTS', 'phi']
contour1.Isosurfaces = [0.5]
contour1.PointMergeMethod = 'Uniform Binning'

# show data in view
contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
contour1Display.Representation = 'Surface'
contour1Display.ColorArrayName = ['POINTS', 'phi']
contour1Display.LookupTable = phiLUT
contour1Display.SelectTCoordArray = 'None'
contour1Display.SelectNormalArray = 'Normals'
contour1Display.SelectTangentArray = 'None'
contour1Display.OSPRayScaleArray = 'phi'
contour1Display.OSPRayScaleFunction = 'PiecewiseFunction'
contour1Display.SelectOrientationVectors = 'None'
contour1Display.ScaleFactor = 0.10043859481811523
contour1Display.SelectScaleArray = 'phi'
contour1Display.GlyphType = 'Arrow'
contour1Display.GlyphTableIndexArray = 'phi'
contour1Display.GaussianRadius = 0.005021929740905762
contour1Display.SetScaleArray = ['POINTS', 'phi']
contour1Display.ScaleTransferFunction = 'PiecewiseFunction'
contour1Display.OpacityArray = ['POINTS', 'phi']
contour1Display.OpacityTransferFunction = 'PiecewiseFunction'
contour1Display.DataAxesGrid = 'GridAxesRepresentation'
contour1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
contour1Display.ScaleTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
contour1Display.OpacityTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]

# hide data in view
Hide(phixdmf, renderView1)

# show color bar/color legend
contour1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# turn off scalar coloring
ColorBy(contour1Display, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(phiLUT, renderView1)

# Properties modified on contour1Display
contour1Display.Opacity = 0.5

# change solid color
contour1Display.AmbientColor = [1.0, 1.0, 0.4980392156862745]
contour1Display.DiffuseColor = [1.0, 1.0, 0.4980392156862745]

# set active source
SetActiveSource(cxdmf)

# create a new 'Contour'
contour2 = Contour(registrationName='Contour2', Input=cxdmf)
contour2.ContourBy = ['POINTS', 'c']
contour2.Isosurfaces = [-0.0048223137855529785]
contour2.PointMergeMethod = 'Uniform Binning'

# Properties modified on contour2
contour2.Isosurfaces = [0.0]

# show data in view
contour2Display = Show(contour2, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
contour2Display.Representation = 'Surface'
contour2Display.ColorArrayName = ['POINTS', 'c']
contour2Display.LookupTable = cLUT
contour2Display.SelectTCoordArray = 'None'
contour2Display.SelectNormalArray = 'Normals'
contour2Display.SelectTangentArray = 'None'
contour2Display.OSPRayScaleArray = 'c'
contour2Display.OSPRayScaleFunction = 'PiecewiseFunction'
contour2Display.SelectOrientationVectors = 'None'
contour2Display.ScaleFactor = 0.1700000047683716
contour2Display.SelectScaleArray = 'c'
contour2Display.GlyphType = 'Arrow'
contour2Display.GlyphTableIndexArray = 'c'
contour2Display.GaussianRadius = 0.008500000238418579
contour2Display.SetScaleArray = ['POINTS', 'c']
contour2Display.ScaleTransferFunction = 'PiecewiseFunction'
contour2Display.OpacityArray = ['POINTS', 'c']
contour2Display.OpacityTransferFunction = 'PiecewiseFunction'
contour2Display.DataAxesGrid = 'GridAxesRepresentation'
contour2Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
contour2Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
contour2Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# hide data in view
Hide(cxdmf, renderView1)

# show color bar/color legend
contour2Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# turn off scalar coloring
ColorBy(contour2Display, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(cLUT, renderView1)

# set active source
SetActiveSource(cxdmf)

# show data in view
cxdmfDisplay = Show(cxdmf, renderView1, 'UnstructuredGridRepresentation')

# show color bar/color legend
cxdmfDisplay.SetScalarBarVisibility(renderView1, True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
cLUT.ApplyPreset('X Ray', True)

# invert the transfer function
cLUT.InvertTransferFunction()

# Properties modified on cxdmfDisplay
cxdmfDisplay.Opacity = 0.1

# hide color bar/color legend
cxdmfDisplay.SetScalarBarVisibility(renderView1, False)

# Properties modified on renderView1.AxesGrid
renderView1.AxesGrid.Visibility = 1

# get layout
layout1 = GetLayout()

# split cell
layout1.SplitHorizontal(0, 0.5)

# set active view
SetActiveView(None)

# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.AxesGrid = 'GridAxes3DActor'
renderView2.StereoType = 'Crystal Eyes'
renderView2.CameraFocalDisk = 1.0
renderView2.BackEnd = 'OSPRay raycaster'
renderView2.OSPRayMaterialLibrary = materialLibrary1

# assign view to a particular cell in the layout
AssignViewToLayout(view=renderView2, layout=layout1, hint=2)

# set active view
SetActiveView(renderView1)

# reset view to fit data bounds
renderView1.ResetCamera(0.0, 1.7000000476837158, 0.0, 1.7000000476837158, 0.0, 1.5)

# set active view
SetActiveView(renderView2)

# split cell
layout1.SplitVertical(2, 0.5)

# set active view
SetActiveView(None)

# Create a new 'Render View'
renderView3 = CreateView('RenderView')
renderView3.AxesGrid = 'GridAxes3DActor'
renderView3.StereoType = 'Crystal Eyes'
renderView3.CameraFocalDisk = 1.0
renderView3.BackEnd = 'OSPRay raycaster'
renderView3.OSPRayMaterialLibrary = materialLibrary1

# assign view to a particular cell in the layout
AssignViewToLayout(view=renderView3, layout=layout1, hint=6)

# set active view
SetActiveView(renderView2)

# show data in view
cxdmfDisplay_1 = Show(cxdmf, renderView2, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
cxdmfDisplay_1.Representation = 'Surface'
cxdmfDisplay_1.ColorArrayName = ['POINTS', 'c']
cxdmfDisplay_1.LookupTable = cLUT
cxdmfDisplay_1.SelectTCoordArray = 'None'
cxdmfDisplay_1.SelectNormalArray = 'None'
cxdmfDisplay_1.SelectTangentArray = 'None'
cxdmfDisplay_1.OSPRayScaleArray = 'c'
cxdmfDisplay_1.OSPRayScaleFunction = 'PiecewiseFunction'
cxdmfDisplay_1.SelectOrientationVectors = 'None'
cxdmfDisplay_1.ScaleFactor = 0.1700000047683716
cxdmfDisplay_1.SelectScaleArray = 'c'
cxdmfDisplay_1.GlyphType = 'Arrow'
cxdmfDisplay_1.GlyphTableIndexArray = 'c'
cxdmfDisplay_1.GaussianRadius = 0.008500000238418579
cxdmfDisplay_1.SetScaleArray = ['POINTS', 'c']
cxdmfDisplay_1.ScaleTransferFunction = 'PiecewiseFunction'
cxdmfDisplay_1.OpacityArray = ['POINTS', 'c']
cxdmfDisplay_1.OpacityTransferFunction = 'PiecewiseFunction'
cxdmfDisplay_1.DataAxesGrid = 'GridAxesRepresentation'
cxdmfDisplay_1.PolarAxes = 'PolarAxesRepresentation'
cxdmfDisplay_1.ScalarOpacityFunction = cPWF
cxdmfDisplay_1.ScalarOpacityUnitDistance = 0.008383798682624784
cxdmfDisplay_1.OpacityArrayName = ['POINTS', 'c']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
cxdmfDisplay_1.ScaleTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 0.990355372428894, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
cxdmfDisplay_1.OpacityTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 0.990355372428894, 1.0, 0.5, 0.0]

# show color bar/color legend
cxdmfDisplay_1.SetScalarBarVisibility(renderView2, True)

# reset view to fit data
renderView2.ResetCamera()

# create a new 'Clip'
clip1 = Clip(registrationName='Clip1', Input=cxdmf)
clip1.ClipType = 'Plane'
clip1.HyperTreeGridClipper = 'Plane'
clip1.Scalars = ['POINTS', 'c']
clip1.Value = -0.0048223137855529785

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Origin = [0.8500000238418579, 0.8500000238418579, 0.75]

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip1.HyperTreeGridClipper.Origin = [0.8500000238418579, 0.8500000238418579, 0.75]

# Properties modified on clip1.ClipType
clip1.ClipType.Origin = [0.8500000238418579, 0.8500000238418579, 1.4]
clip1.ClipType.Normal = [0.0, 0.0, 1.0]

# show data in view
clip1Display = Show(clip1, renderView2, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
clip1Display.Representation = 'Surface'
clip1Display.ColorArrayName = ['POINTS', 'c']
clip1Display.LookupTable = cLUT
clip1Display.SelectTCoordArray = 'None'
clip1Display.SelectNormalArray = 'None'
clip1Display.SelectTangentArray = 'None'
clip1Display.OSPRayScaleArray = 'c'
clip1Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display.SelectOrientationVectors = 'None'
clip1Display.ScaleFactor = 0.1700000047683716
clip1Display.SelectScaleArray = 'c'
clip1Display.GlyphType = 'Arrow'
clip1Display.GlyphTableIndexArray = 'c'
clip1Display.GaussianRadius = 0.008500000238418579
clip1Display.SetScaleArray = ['POINTS', 'c']
clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display.OpacityArray = ['POINTS', 'c']
clip1Display.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display.DataAxesGrid = 'GridAxesRepresentation'
clip1Display.PolarAxes = 'PolarAxesRepresentation'
clip1Display.ScalarOpacityFunction = cPWF
clip1Display.ScalarOpacityUnitDistance = 0.008415480602041376
clip1Display.OpacityArrayName = ['POINTS', 'c']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
clip1Display.ScaleTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 0.9898058176040649, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
clip1Display.OpacityTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 0.9898058176040649, 1.0, 0.5, 0.0]

# hide data in view
Hide(cxdmf, renderView2)

# show color bar/color legend
clip1Display.SetScalarBarVisibility(renderView2, True)

# update the view to ensure updated data information
renderView2.Update()

# set active source
SetActiveSource(cxdmf)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip1.ClipType)

# show data in view
cxdmfDisplay_1 = Show(cxdmf, renderView2, 'UnstructuredGridRepresentation')

# show color bar/color legend
cxdmfDisplay_1.SetScalarBarVisibility(renderView2, True)

# hide data in view
Hide(cxdmf, renderView2)

# set active source
SetActiveSource(contour2)

# show data in view
contour2Display_1 = Show(contour2, renderView2, 'GeometryRepresentation')

# trace defaults for the display properties.
contour2Display_1.Representation = 'Surface'
contour2Display_1.ColorArrayName = ['POINTS', 'c']
contour2Display_1.LookupTable = cLUT
contour2Display_1.SelectTCoordArray = 'None'
contour2Display_1.SelectNormalArray = 'Normals'
contour2Display_1.SelectTangentArray = 'None'
contour2Display_1.OSPRayScaleArray = 'c'
contour2Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
contour2Display_1.SelectOrientationVectors = 'None'
contour2Display_1.ScaleFactor = 0.1700000047683716
contour2Display_1.SelectScaleArray = 'c'
contour2Display_1.GlyphType = 'Arrow'
contour2Display_1.GlyphTableIndexArray = 'c'
contour2Display_1.GaussianRadius = 0.008500000238418579
contour2Display_1.SetScaleArray = ['POINTS', 'c']
contour2Display_1.ScaleTransferFunction = 'PiecewiseFunction'
contour2Display_1.OpacityArray = ['POINTS', 'c']
contour2Display_1.OpacityTransferFunction = 'PiecewiseFunction'
contour2Display_1.DataAxesGrid = 'GridAxesRepresentation'
contour2Display_1.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
contour2Display_1.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
contour2Display_1.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# show color bar/color legend
contour2Display_1.SetScalarBarVisibility(renderView2, True)

# turn off scalar coloring
ColorBy(contour2Display_1, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(cLUT, renderView2)

animationScene1.GoToLast()

# set active source
SetActiveSource(clip1)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip1.ClipType)

# Properties modified on clip1Display
clip1Display.Opacity = 0.6

# hide color bar/color legend
clip1Display.SetScalarBarVisibility(renderView2, False)

# set active view
SetActiveView(renderView3)

# set active source
SetActiveSource(cxdmf)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip1.ClipType)

# show data in view
cxdmfDisplay_2 = Show(cxdmf, renderView3, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
cxdmfDisplay_2.Representation = 'Surface'
cxdmfDisplay_2.ColorArrayName = ['POINTS', 'c']
cxdmfDisplay_2.LookupTable = cLUT
cxdmfDisplay_2.SelectTCoordArray = 'None'
cxdmfDisplay_2.SelectNormalArray = 'None'
cxdmfDisplay_2.SelectTangentArray = 'None'
cxdmfDisplay_2.OSPRayScaleArray = 'c'
cxdmfDisplay_2.OSPRayScaleFunction = 'PiecewiseFunction'
cxdmfDisplay_2.SelectOrientationVectors = 'None'
cxdmfDisplay_2.ScaleFactor = 0.1700000047683716
cxdmfDisplay_2.SelectScaleArray = 'c'
cxdmfDisplay_2.GlyphType = 'Arrow'
cxdmfDisplay_2.GlyphTableIndexArray = 'c'
cxdmfDisplay_2.GaussianRadius = 0.008500000238418579
cxdmfDisplay_2.SetScaleArray = ['POINTS', 'c']
cxdmfDisplay_2.ScaleTransferFunction = 'PiecewiseFunction'
cxdmfDisplay_2.OpacityArray = ['POINTS', 'c']
cxdmfDisplay_2.OpacityTransferFunction = 'PiecewiseFunction'
cxdmfDisplay_2.DataAxesGrid = 'GridAxesRepresentation'
cxdmfDisplay_2.PolarAxes = 'PolarAxesRepresentation'
cxdmfDisplay_2.ScalarOpacityFunction = cPWF
cxdmfDisplay_2.ScalarOpacityUnitDistance = 0.008383798682624784
cxdmfDisplay_2.OpacityArrayName = ['POINTS', 'c']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
cxdmfDisplay_2.ScaleTransferFunction.Points = [-1.3017044067382812, 0.0, 0.5, 0.0, 1.613266110420227, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
cxdmfDisplay_2.OpacityTransferFunction.Points = [-1.3017044067382812, 0.0, 0.5, 0.0, 1.613266110420227, 1.0, 0.5, 0.0]

# show color bar/color legend
cxdmfDisplay_2.SetScalarBarVisibility(renderView3, True)

# reset view to fit data
renderView3.ResetCamera()

# create a new 'Clip'
clip2 = Clip(registrationName='Clip2', Input=cxdmf)
clip2.ClipType = 'Plane'
clip2.HyperTreeGridClipper = 'Plane'
clip2.Scalars = ['POINTS', 'c']
clip2.Value = 0.1557808518409729

# init the 'Plane' selected for 'ClipType'
clip2.ClipType.Origin = [0.8500000238418579, 0.8500000238418579, 0.75]

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip2.HyperTreeGridClipper.Origin = [0.8500000238418579, 0.8500000238418579, 0.75]

# Properties modified on clip2.ClipType
clip2.ClipType.Origin = [0.8500000238418579, 0.8500000238418579, 1.2]
clip2.ClipType.Normal = [0.0, 0.0, 1.0]

# show data in view
clip2Display = Show(clip2, renderView3, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
clip2Display.Representation = 'Surface'
clip2Display.ColorArrayName = ['POINTS', 'c']
clip2Display.LookupTable = cLUT
clip2Display.SelectTCoordArray = 'None'
clip2Display.SelectNormalArray = 'None'
clip2Display.SelectTangentArray = 'None'
clip2Display.OSPRayScaleArray = 'c'
clip2Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip2Display.SelectOrientationVectors = 'None'
clip2Display.ScaleFactor = 0.1700000047683716
clip2Display.SelectScaleArray = 'c'
clip2Display.GlyphType = 'Arrow'
clip2Display.GlyphTableIndexArray = 'c'
clip2Display.GaussianRadius = 0.008500000238418579
clip2Display.SetScaleArray = ['POINTS', 'c']
clip2Display.ScaleTransferFunction = 'PiecewiseFunction'
clip2Display.OpacityArray = ['POINTS', 'c']
clip2Display.OpacityTransferFunction = 'PiecewiseFunction'
clip2Display.DataAxesGrid = 'GridAxesRepresentation'
clip2Display.PolarAxes = 'PolarAxesRepresentation'
clip2Display.ScalarOpacityFunction = cPWF
clip2Display.ScalarOpacityUnitDistance = 0.008559403884667909
clip2Display.OpacityArrayName = ['POINTS', 'c']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
clip2Display.ScaleTransferFunction.Points = [-1.2949007749557495, 0.0, 0.5, 0.0, 1.613266110420227, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
clip2Display.OpacityTransferFunction.Points = [-1.2949007749557495, 0.0, 0.5, 0.0, 1.613266110420227, 1.0, 0.5, 0.0]

# hide data in view
Hide(cxdmf, renderView3)

# show color bar/color legend
clip2Display.SetScalarBarVisibility(renderView3, True)

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
renderView2.Update()

# update the view to ensure updated data information
renderView3.Update()

# set active source
SetActiveSource(contour2)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip2.ClipType)

# create a new 'Clip'
clip3 = Clip(registrationName='Clip3', Input=contour2)
clip3.ClipType = 'Plane'
clip3.HyperTreeGridClipper = 'Plane'
clip3.Scalars = ['POINTS', 'c']

# init the 'Plane' selected for 'ClipType'
clip3.ClipType.Origin = [0.8500000238418579, 0.8500000238418579, 1.0353071689605713]

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip3.HyperTreeGridClipper.Origin = [0.8500000238418579, 0.8500000238418579, 1.0353071689605713]

# Properties modified on clip3.ClipType
clip3.ClipType.Origin = [0.8500000238418579, 0.8500000238418579, 1.25]
clip3.ClipType.Normal = [0.0, 0.0, 1.0]

# show data in view
clip3Display = Show(clip3, renderView3, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
clip3Display.Representation = 'Surface'
clip3Display.ColorArrayName = ['POINTS', 'c']
clip3Display.LookupTable = cLUT
clip3Display.SelectTCoordArray = 'None'
clip3Display.SelectNormalArray = 'Normals'
clip3Display.SelectTangentArray = 'None'
clip3Display.OSPRayScaleArray = 'c'
clip3Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip3Display.SelectOrientationVectors = 'None'
clip3Display.ScaleFactor = 0.06670869588851928
clip3Display.SelectScaleArray = 'c'
clip3Display.GlyphType = 'Arrow'
clip3Display.GlyphTableIndexArray = 'c'
clip3Display.GaussianRadius = 0.003335434794425964
clip3Display.SetScaleArray = ['POINTS', 'c']
clip3Display.ScaleTransferFunction = 'PiecewiseFunction'
clip3Display.OpacityArray = ['POINTS', 'c']
clip3Display.OpacityTransferFunction = 'PiecewiseFunction'
clip3Display.DataAxesGrid = 'GridAxesRepresentation'
clip3Display.PolarAxes = 'PolarAxesRepresentation'
clip3Display.ScalarOpacityFunction = cPWF
clip3Display.ScalarOpacityUnitDistance = 0.01561462210486338
clip3Display.OpacityArrayName = ['POINTS', 'c']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
clip3Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
clip3Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# show color bar/color legend
clip3Display.SetScalarBarVisibility(renderView3, True)

# update the view to ensure updated data information
renderView3.Update()

# turn off scalar coloring
ColorBy(clip3Display, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(cLUT, renderView3)

# set active source
SetActiveSource(clip2)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip3.ClipType)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip2.ClipType)

# Properties modified on clip2Display
clip2Display.Opacity = 0.6

# hide color bar/color legend
clip2Display.SetScalarBarVisibility(renderView3, False)

# set active source
SetActiveSource(cxdmf)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip2.ClipType)

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1515, 830)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [-1.3316287921395205, -2.757763585696212, 4.974410317376616]
renderView1.CameraFocalPoint = [0.8500000238418579, 0.8500000238418579, 0.75]
renderView1.CameraViewUp = [-0.172162226452288, 0.7912126855992522, 0.5868071692897165]
renderView1.CameraParallelScale = 1.5534954965832222

# current camera placement for renderView2
renderView2.CameraPosition = [0.8500000238418579, 0.8500000238418579, 6.224337329293185]
renderView2.CameraFocalPoint = [0.8500000238418579, 0.8500000238418579, 0.75]
renderView2.CameraParallelScale = 1.4168627601367458

# current camera placement for renderView3
renderView3.CameraPosition = [0.8500000238418579, 0.8500000238418579, 6.224337329293185]
renderView3.CameraFocalPoint = [0.8500000238418579, 0.8500000238418579, 0.75]
renderView3.CameraParallelScale = 1.4168627601367458

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).