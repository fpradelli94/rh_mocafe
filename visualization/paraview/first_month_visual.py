"""
NOTICE: This script is meant to be executed from Paraview
"""

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

folder = input("Paste folder: ")

# create a new 'Xdmf3ReaderS'
cxdmf = Xdmf3ReaderS(registrationName='c.xdmf', FileName=[f'{folder}/xdmf/c.xdmf'])
cxdmf.PointArrays = ['c', 'vtkGhostType', 'vtkOriginalPointIds']

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
cxdmfDisplay = Show(cxdmf, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
cxdmfDisplay.Representation = 'Surface'
cxdmfDisplay.ColorArrayName = [None, '']
cxdmfDisplay.SelectTCoordArray = 'None'
cxdmfDisplay.SelectNormalArray = 'None'
cxdmfDisplay.SelectTangentArray = 'None'
cxdmfDisplay.OSPRayScaleArray = 'c'
cxdmfDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
cxdmfDisplay.SelectOrientationVectors = 'None'
cxdmfDisplay.ScaleFactor = 0.16662500000000002
cxdmfDisplay.SelectScaleArray = 'None'
cxdmfDisplay.GlyphType = 'Arrow'
cxdmfDisplay.GlyphTableIndexArray = 'None'
cxdmfDisplay.GaussianRadius = 0.00833125
cxdmfDisplay.SetScaleArray = ['POINTS', 'c']
cxdmfDisplay.ScaleTransferFunction = 'PiecewiseFunction'
cxdmfDisplay.OpacityArray = ['POINTS', 'c']
cxdmfDisplay.OpacityTransferFunction = 'PiecewiseFunction'
cxdmfDisplay.DataAxesGrid = 'GridAxesRepresentation'
cxdmfDisplay.PolarAxes = 'PolarAxesRepresentation'
cxdmfDisplay.ScalarOpacityUnitDistance = 0.008377880198659605
cxdmfDisplay.OpacityArrayName = ['POINTS', 'c']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
cxdmfDisplay.ScaleTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
cxdmfDisplay.OpacityTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera()

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(cxdmfDisplay, ('POINTS', 'c'))

# rescale color and/or opacity maps used to include current data range
cxdmfDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
cxdmfDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'c'
cLUT = GetColorTransferFunction('c')

# get opacity transfer function/opacity map for 'c'
cPWF = GetOpacityTransferFunction('c')

# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=cxdmf)
contour1.ContourBy = ['POINTS', 'c']
contour1.Isosurfaces = [0.0]
contour1.PointMergeMethod = 'Uniform Binning'

# show data in view
contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
contour1Display.Representation = 'Surface'
contour1Display.ColorArrayName = ['POINTS', 'c']
contour1Display.LookupTable = cLUT
contour1Display.SelectTCoordArray = 'None'
contour1Display.SelectNormalArray = 'Normals'
contour1Display.SelectTangentArray = 'None'
contour1Display.OSPRayScaleArray = 'c'
contour1Display.OSPRayScaleFunction = 'PiecewiseFunction'
contour1Display.SelectOrientationVectors = 'None'
contour1Display.ScaleFactor = 0.16662500000000002
contour1Display.SelectScaleArray = 'c'
contour1Display.GlyphType = 'Arrow'
contour1Display.GlyphTableIndexArray = 'c'
contour1Display.GaussianRadius = 0.00833125
contour1Display.SetScaleArray = ['POINTS', 'c']
contour1Display.ScaleTransferFunction = 'PiecewiseFunction'
contour1Display.OpacityArray = ['POINTS', 'c']
contour1Display.OpacityTransferFunction = 'PiecewiseFunction'
contour1Display.DataAxesGrid = 'GridAxesRepresentation'
contour1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
contour1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
contour1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# hide data in view
Hide(cxdmf, renderView1)

# show color bar/color legend
contour1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# turn off scalar coloring
ColorBy(contour1Display, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(cLUT, renderView1)

# Properties modified on renderView1
renderView1.CameraParallelProjection = 1

# Properties modified on renderView1
renderView1.CameraParallelProjection = 0

# Properties modified on renderView1
renderView1.OrientationAxesVisibility = 0

# reset view to fit data
renderView1.ResetCamera()

# create a new 'Xdmf3ReaderS'
phixdmf = Xdmf3ReaderS(registrationName='phi.xdmf', FileName=[f'{folder}/xdmf/phi.xdmf'])
phixdmf.PointArrays = ['phi', 'vtkGhostType', 'vtkOriginalPointIds']

# show data in view
phixdmfDisplay = Show(phixdmf, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
phixdmfDisplay.Representation = 'Surface'
phixdmfDisplay.ColorArrayName = [None, '']
phixdmfDisplay.SelectTCoordArray = 'None'
phixdmfDisplay.SelectNormalArray = 'None'
phixdmfDisplay.SelectTangentArray = 'None'
phixdmfDisplay.OSPRayScaleArray = 'phi'
phixdmfDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
phixdmfDisplay.SelectOrientationVectors = 'None'
phixdmfDisplay.ScaleFactor = 0.16662500000000002
phixdmfDisplay.SelectScaleArray = 'None'
phixdmfDisplay.GlyphType = 'Arrow'
phixdmfDisplay.GlyphTableIndexArray = 'None'
phixdmfDisplay.GaussianRadius = 0.00833125
phixdmfDisplay.SetScaleArray = ['POINTS', 'phi']
phixdmfDisplay.ScaleTransferFunction = 'PiecewiseFunction'
phixdmfDisplay.OpacityArray = ['POINTS', 'phi']
phixdmfDisplay.OpacityTransferFunction = 'PiecewiseFunction'
phixdmfDisplay.DataAxesGrid = 'GridAxesRepresentation'
phixdmfDisplay.PolarAxes = 'PolarAxesRepresentation'
phixdmfDisplay.ScalarOpacityUnitDistance = 0.008377880198659605
phixdmfDisplay.OpacityArrayName = ['POINTS', 'phi']

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Contour'
contour2 = Contour(registrationName='Contour2', Input=phixdmf)
contour2.ContourBy = ['POINTS', 'phi']
contour2.Isosurfaces = [0.5]
contour2.PointMergeMethod = 'Uniform Binning'

# show data in view
contour2Display = Show(contour2, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'phi'
phiLUT = GetColorTransferFunction('phi')

# trace defaults for the display properties.
contour2Display.Representation = 'Surface'
contour2Display.ColorArrayName = ['POINTS', 'phi']
contour2Display.LookupTable = phiLUT
contour2Display.SelectTCoordArray = 'None'
contour2Display.SelectNormalArray = 'Normals'
contour2Display.SelectTangentArray = 'None'
contour2Display.OSPRayScaleArray = 'phi'
contour2Display.OSPRayScaleFunction = 'PiecewiseFunction'
contour2Display.SelectOrientationVectors = 'None'
contour2Display.ScaleFactor = 0.036821428571428574
contour2Display.SelectScaleArray = 'phi'
contour2Display.GlyphType = 'Arrow'
contour2Display.GlyphTableIndexArray = 'phi'
contour2Display.GaussianRadius = 0.0018410714285714287
contour2Display.SetScaleArray = ['POINTS', 'phi']
contour2Display.ScaleTransferFunction = 'PiecewiseFunction'
contour2Display.OpacityArray = ['POINTS', 'phi']
contour2Display.OpacityTransferFunction = 'PiecewiseFunction'
contour2Display.DataAxesGrid = 'GridAxesRepresentation'
contour2Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
contour2Display.ScaleTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
contour2Display.OpacityTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]

# hide data in view
Hide(phixdmf, renderView1)

# show color bar/color legend
contour2Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get opacity transfer function/opacity map for 'phi'
phiPWF = GetOpacityTransferFunction('phi')

# turn off scalar coloring
ColorBy(contour2Display, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(phiLUT, renderView1)

# change solid color
contour2Display.AmbientColor = [1.0, 1.0, 0.0]
contour2Display.DiffuseColor = [1.0, 1.0, 0.0]

# Properties modified on contour2Display
contour2Display.Opacity = 0.3

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1612, 830)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [0.833125, 0.833125, 5.744743664886641]
renderView1.CameraFocalPoint = [0.833125, 0.833125, 1.1879294217687075]
renderView1.CameraParallelScale = 1.1793903111133492

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
