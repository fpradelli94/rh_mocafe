"""
NOTICE: This script is meant to be executed from Paraview
"""

# trace generated using paraview version 5.9.0

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# set folder
folder = input("Paste folder: ")

# create a new 'Xdmf3ReaderS'
afxdmf = Xdmf3ReaderS(registrationName='af.xdmf', FileName=[f'{folder}xdmf/af.xdmf'])
afxdmf.PointArrays = ['af', 'vtkGhostType', 'vtkOriginalPointIds']

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
afxdmfDisplay = Show(afxdmf, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
afxdmfDisplay.Representation = 'Surface'
afxdmfDisplay.ColorArrayName = [None, '']
afxdmfDisplay.SelectTCoordArray = 'None'
afxdmfDisplay.SelectNormalArray = 'None'
afxdmfDisplay.SelectTangentArray = 'None'
afxdmfDisplay.OSPRayScaleArray = 'af'
afxdmfDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
afxdmfDisplay.SelectOrientationVectors = 'None'
afxdmfDisplay.ScaleFactor = 0.16662500000000002
afxdmfDisplay.SelectScaleArray = 'None'
afxdmfDisplay.GlyphType = 'Arrow'
afxdmfDisplay.GlyphTableIndexArray = 'None'
afxdmfDisplay.GaussianRadius = 0.00833125
afxdmfDisplay.SetScaleArray = ['POINTS', 'af']
afxdmfDisplay.ScaleTransferFunction = 'PiecewiseFunction'
afxdmfDisplay.OpacityArray = ['POINTS', 'af']
afxdmfDisplay.OpacityTransferFunction = 'PiecewiseFunction'
afxdmfDisplay.DataAxesGrid = 'GridAxesRepresentation'
afxdmfDisplay.PolarAxes = 'PolarAxesRepresentation'
afxdmfDisplay.ScalarOpacityUnitDistance = 0.008364995055153713
afxdmfDisplay.OpacityArrayName = ['POINTS', 'af']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
afxdmfDisplay.ScaleTransferFunction.Points = [0.8003977096907203, 0.0, 0.5, 0.0, 5.423451131640912, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
afxdmfDisplay.OpacityTransferFunction.Points = [0.8003977096907203, 0.0, 0.5, 0.0, 5.423451131640912, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera()

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Programmable Filter'
programmableFilter1 = ProgrammableFilter(registrationName='ProgrammableFilter1', Input=afxdmf)
programmableFilter1.Script = ''
programmableFilter1.RequestInformationScript = ''
programmableFilter1.RequestUpdateExtentScript = ''
programmableFilter1.PythonPath = ''

# Properties modified on programmableFilter1
programmableFilter1.Script = ''
programmableFilter1.RequestInformationScript = ''
programmableFilter1.RequestUpdateExtentScript = ''
programmableFilter1.PythonPath = ''

# show data in view
programmableFilter1Display = Show(programmableFilter1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
programmableFilter1Display.Representation = 'Surface'
programmableFilter1Display.ColorArrayName = [None, '']
programmableFilter1Display.SelectTCoordArray = 'None'
programmableFilter1Display.SelectNormalArray = 'None'
programmableFilter1Display.SelectTangentArray = 'None'
programmableFilter1Display.OSPRayScaleFunction = 'PiecewiseFunction'
programmableFilter1Display.SelectOrientationVectors = 'None'
programmableFilter1Display.ScaleFactor = 0.16662500000000002
programmableFilter1Display.SelectScaleArray = 'None'
programmableFilter1Display.GlyphType = 'Arrow'
programmableFilter1Display.GlyphTableIndexArray = 'None'
programmableFilter1Display.GaussianRadius = 0.00833125
programmableFilter1Display.SetScaleArray = [None, '']
programmableFilter1Display.ScaleTransferFunction = 'PiecewiseFunction'
programmableFilter1Display.OpacityArray = [None, '']
programmableFilter1Display.OpacityTransferFunction = 'PiecewiseFunction'
programmableFilter1Display.DataAxesGrid = 'GridAxesRepresentation'
programmableFilter1Display.PolarAxes = 'PolarAxesRepresentation'
programmableFilter1Display.ScalarOpacityUnitDistance = 0.008364995055153713
programmableFilter1Display.OpacityArrayName = [None, '']

# hide data in view
Hide(afxdmf, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on programmableFilter1
programmableFilter1.Script = """af = inputs[0].PointData[\'af\']
af_ng = 6 * af
output.PointData.append(af_ng, "af [ng / mL")"""

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(programmableFilter1Display, ('POINTS', 'af [ng / mL'))

# rescale color and/or opacity maps used to include current data range
programmableFilter1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
programmableFilter1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'afngmL'
afngmLLUT = GetColorTransferFunction('afngmL')

# get opacity transfer function/opacity map for 'afngmL'
afngmLPWF = GetOpacityTransferFunction('afngmL')

# create a new 'Clip'
clip1 = Clip(registrationName='Clip1', Input=programmableFilter1)
clip1.ClipType = 'Plane'
clip1.HyperTreeGridClipper = 'Plane'
clip1.Scalars = ['POINTS', 'af [ng / mL']
clip1.Value = 18.6715465239949

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Origin = [0.833125, 0.833125, 0.644375]

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip1.HyperTreeGridClipper.Origin = [0.833125, 0.833125, 0.644375]

# Properties modified on clip1.ClipType
clip1.ClipType.Origin = [0.833125, 0.833125, 0.8]
clip1.ClipType.Normal = [0.0, 0.0, 1.0]

# show data in view
clip1Display = Show(clip1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
clip1Display.Representation = 'Surface'
clip1Display.ColorArrayName = ['POINTS', 'af [ng / mL']
clip1Display.LookupTable = afngmLLUT
clip1Display.SelectTCoordArray = 'None'
clip1Display.SelectNormalArray = 'None'
clip1Display.SelectTangentArray = 'None'
clip1Display.OSPRayScaleArray = 'af [ng / mL'
clip1Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display.SelectOrientationVectors = 'None'
clip1Display.ScaleFactor = 0.16662500000000002
clip1Display.SelectScaleArray = 'None'
clip1Display.GlyphType = 'Arrow'
clip1Display.GlyphTableIndexArray = 'None'
clip1Display.GaussianRadius = 0.008331250000000002
clip1Display.SetScaleArray = ['POINTS', 'af [ng / mL']
clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display.OpacityArray = ['POINTS', 'af [ng / mL']
clip1Display.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display.DataAxesGrid = 'GridAxesRepresentation'
clip1Display.PolarAxes = 'PolarAxesRepresentation'
clip1Display.ScalarOpacityFunction = afngmLPWF
clip1Display.ScalarOpacityUnitDistance = 0.00905750674946504
clip1Display.OpacityArrayName = ['POINTS', 'af [ng / mL']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
clip1Display.ScaleTransferFunction.Points = [4.802386258144322, 0.0, 0.5, 0.0, 32.25569831067518, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
clip1Display.OpacityTransferFunction.Points = [4.802386258144322, 0.0, 0.5, 0.0, 32.25569831067518, 1.0, 0.5, 0.0]

# hide data in view
Hide(programmableFilter1, renderView1)

# show color bar/color legend
clip1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Xdmf3ReaderS'
phixdmf = Xdmf3ReaderS(registrationName='phi.xdmf', FileName=[f'{folder}xdmf/phi.xdmf'])
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
phixdmfDisplay.ScalarOpacityUnitDistance = 0.008364995055153713
phixdmfDisplay.OpacityArrayName = ['POINTS', 'phi']

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=phixdmf)
contour1.ContourBy = ['POINTS', 'phi']
contour1.Isosurfaces = [0.5]
contour1.PointMergeMethod = 'Uniform Binning'

# show data in view
contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'phi'
phiLUT = GetColorTransferFunction('phi')

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
contour1Display.ScaleFactor = 0.09424532312925171
contour1Display.SelectScaleArray = 'phi'
contour1Display.GlyphType = 'Arrow'
contour1Display.GlyphTableIndexArray = 'phi'
contour1Display.GaussianRadius = 0.004712266156462585
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

# get opacity transfer function/opacity map for 'phi'
phiPWF = GetOpacityTransferFunction('phi')

# turn off scalar coloring
ColorBy(contour1Display, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(phiLUT, renderView1)

# change solid color
contour1Display.AmbientColor = [1.0, 1.0, 0.0]
contour1Display.DiffuseColor = [1.0, 1.0, 0.0]

# Properties modified on contour1Display
contour1Display.Opacity = 0.5

# create a new 'Clip'
clip2 = Clip(registrationName='Clip2', Input=contour1)
clip2.ClipType = 'Plane'
clip2.HyperTreeGridClipper = 'Plane'
clip2.Scalars = ['POINTS', 'phi']
clip2.Value = 0.5

# init the 'Plane' selected for 'ClipType'
clip2.ClipType.Origin = [0.8331249999999999, 0.8331249999999999, 0.8175233843537415]

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip2.HyperTreeGridClipper.Origin = [0.8331249999999999, 0.8331249999999999, 0.8175233843537415]

# Properties modified on clip2.ClipType
clip2.ClipType.Normal = [-1.0, -1.0, 0.0]

# show data in view
clip2Display = Show(clip2, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
clip2Display.Representation = 'Surface'
clip2Display.ColorArrayName = ['POINTS', 'phi']
clip2Display.LookupTable = phiLUT
clip2Display.SelectTCoordArray = 'None'
clip2Display.SelectNormalArray = 'Normals'
clip2Display.SelectTangentArray = 'None'
clip2Display.OSPRayScaleArray = 'phi'
clip2Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip2Display.SelectOrientationVectors = 'None'
clip2Display.ScaleFactor = 0.09424532312925171
clip2Display.SelectScaleArray = 'phi'
clip2Display.GlyphType = 'Arrow'
clip2Display.GlyphTableIndexArray = 'phi'
clip2Display.GaussianRadius = 0.004712266156462585
clip2Display.SetScaleArray = ['POINTS', 'phi']
clip2Display.ScaleTransferFunction = 'PiecewiseFunction'
clip2Display.OpacityArray = ['POINTS', 'phi']
clip2Display.OpacityTransferFunction = 'PiecewiseFunction'
clip2Display.DataAxesGrid = 'GridAxesRepresentation'
clip2Display.PolarAxes = 'PolarAxesRepresentation'
clip2Display.ScalarOpacityFunction = phiPWF
clip2Display.ScalarOpacityUnitDistance = 0.025801722241236353
clip2Display.OpacityArrayName = ['POINTS', 'phi']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
clip2Display.ScaleTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
clip2Display.OpacityTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]

# show color bar/color legend
clip2Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# hide data in view
Hide(contour1, renderView1)

# turn off scalar coloring
ColorBy(clip2Display, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(phiLUT, renderView1)

# change solid color
clip2Display.AmbientColor = [1.0, 1.0, 0.0]
clip2Display.DiffuseColor = [1.0, 1.0, 0.0]

# Properties modified on clip2Display
clip2Display.Opacity = 0.5

# set active source
SetActiveSource(programmableFilter1)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip2.ClipType)

# create a new 'Plot Data Over Time'
plotDataOverTime1 = PlotDataOverTime(registrationName='PlotDataOverTime1', Input=programmableFilter1)

# Create a new 'Quartile Chart View'
quartileChartView1 = CreateView('QuartileChartView')

# show data in view
plotDataOverTime1Display = Show(plotDataOverTime1, quartileChartView1, 'QuartileChartRepresentation')

# trace defaults for the display properties.
plotDataOverTime1Display.AttributeType = 'Row Data'
plotDataOverTime1Display.UseIndexForXAxis = 0
plotDataOverTime1Display.XArrayName = 'Time'
plotDataOverTime1Display.SeriesVisibility = ['af [ng / mL (stats)']
plotDataOverTime1Display.SeriesLabel = ['af [ng / mL (stats)', 'af [ng / mL (stats)', 'X (stats)', 'X (stats)', 'Y (stats)', 'Y (stats)', 'Z (stats)', 'Z (stats)', 'N (stats)', 'N (stats)', 'Time (stats)', 'Time (stats)', 'vtkValidPointMask (stats)', 'vtkValidPointMask (stats)']
plotDataOverTime1Display.SeriesColor = ['af [ng / mL (stats)', '0', '0', '0', 'X (stats)', '0.8899977111467154', '0.10000762951094835', '0.1100022888532845', 'Y (stats)', '0.220004577706569', '0.4899977111467155', '0.7199969481956207', 'Z (stats)', '0.30000762951094834', '0.6899977111467155', '0.2899977111467155', 'N (stats)', '0.6', '0.3100022888532845', '0.6399938963912413', 'Time (stats)', '1', '0.5000076295109483', '0', 'vtkValidPointMask (stats)', '0.6500038147554742', '0.3400015259021897', '0.16000610360875867']
plotDataOverTime1Display.SeriesPlotCorner = ['af [ng / mL (stats)', '0', 'X (stats)', '0', 'Y (stats)', '0', 'Z (stats)', '0', 'N (stats)', '0', 'Time (stats)', '0', 'vtkValidPointMask (stats)', '0']
plotDataOverTime1Display.SeriesLabelPrefix = ''
plotDataOverTime1Display.SeriesLineStyle = ['af [ng / mL (stats)', '1', 'X (stats)', '1', 'Y (stats)', '1', 'Z (stats)', '1', 'N (stats)', '1', 'Time (stats)', '1', 'vtkValidPointMask (stats)', '1']
plotDataOverTime1Display.SeriesLineThickness = ['af [ng / mL (stats)', '2', 'X (stats)', '2', 'Y (stats)', '2', 'Z (stats)', '2', 'N (stats)', '2', 'Time (stats)', '2', 'vtkValidPointMask (stats)', '2']
plotDataOverTime1Display.SeriesMarkerStyle = ['af [ng / mL (stats)', '0', 'X (stats)', '0', 'Y (stats)', '0', 'Z (stats)', '0', 'N (stats)', '0', 'Time (stats)', '0', 'vtkValidPointMask (stats)', '0']
plotDataOverTime1Display.SeriesMarkerSize = ['af [ng / mL (stats)', '4', 'X (stats)', '4', 'Y (stats)', '4', 'Z (stats)', '4', 'N (stats)', '4', 'Time (stats)', '4', 'vtkValidPointMask (stats)', '4']

# get layout
layout1 = GetLayoutByName("Layout #1")

# add view to a layout so it's visible in UI
AssignViewToLayout(view=quartileChartView1, layout=layout1, hint=0)

# Properties modified on plotDataOverTime1Display
plotDataOverTime1Display.SeriesPlotCorner = ['N (stats)', '0', 'Time (stats)', '0', 'X (stats)', '0', 'Y (stats)', '0', 'Z (stats)', '0', 'af [ng / mL (stats)', '0', 'vtkValidPointMask (stats)', '0']
plotDataOverTime1Display.SeriesLineStyle = ['N (stats)', '1', 'Time (stats)', '1', 'X (stats)', '1', 'Y (stats)', '1', 'Z (stats)', '1', 'af [ng / mL (stats)', '1', 'vtkValidPointMask (stats)', '1']
plotDataOverTime1Display.SeriesLineThickness = ['N (stats)', '2', 'Time (stats)', '2', 'X (stats)', '2', 'Y (stats)', '2', 'Z (stats)', '2', 'af [ng / mL (stats)', '2', 'vtkValidPointMask (stats)', '2']
plotDataOverTime1Display.SeriesMarkerStyle = ['N (stats)', '0', 'Time (stats)', '0', 'X (stats)', '0', 'Y (stats)', '0', 'Z (stats)', '0', 'af [ng / mL (stats)', '0', 'vtkValidPointMask (stats)', '0']
plotDataOverTime1Display.SeriesMarkerSize = ['N (stats)', '4', 'Time (stats)', '4', 'X (stats)', '4', 'Y (stats)', '4', 'Z (stats)', '4', 'af [ng / mL (stats)', '4', 'vtkValidPointMask (stats)', '4']

# update the view to ensure updated data information
quartileChartView1.Update()

# Properties modified on plotDataOverTime1Display
plotDataOverTime1Display.ShowRanges = 0

# Properties modified on plotDataOverTime1Display
plotDataOverTime1Display.SeriesColor = ['af [ng / mL (stats)', '0.2', '0.5333333333333333', '0.7254901960784313', 'X (stats)', '0.8899977111467154', '0.10000762951094835', '0.1100022888532845', 'Y (stats)', '0.220004577706569', '0.4899977111467155', '0.7199969481956207', 'Z (stats)', '0.30000762951094834', '0.6899977111467155', '0.2899977111467155', 'N (stats)', '0.6', '0.3100022888532845', '0.6399938963912413', 'Time (stats)', '1', '0.5000076295109483', '0', 'vtkValidPointMask (stats)', '0.6500038147554742', '0.3400015259021897', '0.16000610360875867']

# Properties modified on quartileChartView1
quartileChartView1.LeftAxisUseCustomRange = 1

# Properties modified on quartileChartView1
quartileChartView1.LeftAxisRangeMinimum = 0.0

# Properties modified on quartileChartView1
quartileChartView1.LeftAxisTitle = 'af [ng / mL]'

# Properties modified on quartileChartView1
quartileChartView1.BottomAxisUseCustomRange = 1

# Properties modified on quartileChartView1
quartileChartView1.BottomAxisUseCustomRange = 0

# Properties modified on quartileChartView1
quartileChartView1.BottomAxisTitle = 'steps'

# Properties modified on quartileChartView1
quartileChartView1.ShowLegend = 0

# set active source
SetActiveSource(clip1)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=clip1.ClipType)

# set active source
SetActiveSource(programmableFilter1)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=clip1.ClipType)

# set active source
SetActiveSource(afxdmf)

# set active source
SetActiveSource(clip1)

# set active source
SetActiveSource(programmableFilter1)

# set active source
SetActiveSource(plotDataOverTime1)

# set active view
SetActiveView(renderView1)

# set active source
SetActiveSource(clip1)

# get color legend/bar for afngmLLUT in view renderView1
afngmLLUTColorBar = GetScalarBar(afngmLLUT, renderView1)

# Properties modified on afngmLLUTColorBar
afngmLLUTColorBar.WindowLocation = 'UpperRightCorner'
afngmLLUTColorBar.Title = 'af [ng / mL]'
afngmLLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
afngmLLUTColorBar.LabelColor = [0.0, 0.0, 0.0]

# reset view to fit data bounds
renderView1.ResetCamera(0.0, 1.6662500000000002, 0.0, 1.6662500000000002, 0.0, 0.8)

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1603, 830)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [-1.2931499541206015, -2.5065187334964314, 3.40452479618454]
renderView1.CameraFocalPoint = [0.8331250000000001, 0.8331250000000001, 0.4]
renderView1.CameraViewUp = [-0.004993048680104638, 0.6705694545000819, 0.7418299509701269]
renderView1.CameraParallelScale = 1.2893129382095199

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
