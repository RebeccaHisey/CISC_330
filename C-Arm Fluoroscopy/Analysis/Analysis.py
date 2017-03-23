import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import math
import numpy


#
# Analysis
#

class Analysis(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Analysis" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["John Doe (AnyWare Corp.)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
    This is an example of scripted loadable module bundled in an extension.
    It performs a simple thresholding on the input volume and optionally captures a screenshot.
    """
    self.parent.acknowledgementText = """
    This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
    and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# AnalysisWidget
#

class AnalysisWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  def __init__(self, parent=None):
      ScriptedLoadableModuleWidget.__init__(self, parent)
      self.logic = AnalysisLogic()

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # input volume selector
    #
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputSelector.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("Input Volume: ", self.inputSelector)

    #
    # output volume selector
    #
    self.outputSelector = slicer.qMRMLNodeComboBox()
    self.outputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.outputSelector.selectNodeUponCreation = True
    self.outputSelector.addEnabled = True
    self.outputSelector.removeEnabled = True
    self.outputSelector.noneEnabled = True
    self.outputSelector.showHidden = False
    self.outputSelector.showChildNodeTypes = False
    self.outputSelector.setMRMLScene( slicer.mrmlScene )
    self.outputSelector.setToolTip( "Pick the output to the algorithm." )
    parametersFormLayout.addRow("Output Volume: ", self.outputSelector)

    #
    # threshold value
    #
    self.imageThresholdSliderWidget = ctk.ctkSliderWidget()
    self.imageThresholdSliderWidget.singleStep = 0.1
    self.imageThresholdSliderWidget.minimum = -100
    self.imageThresholdSliderWidget.maximum = 100
    self.imageThresholdSliderWidget.value = 0.5
    self.imageThresholdSliderWidget.setToolTip("Set threshold value for computing the output image. Voxels that have intensities lower than this value will set to zero.")
    parametersFormLayout.addRow("Image threshold", self.imageThresholdSliderWidget)

    #
    # check box to trigger taking screen shots for later use in tutorials
    #
    self.enableScreenshotsFlagCheckBox = qt.QCheckBox()
    self.enableScreenshotsFlagCheckBox.checked = 0
    self.enableScreenshotsFlagCheckBox.setToolTip("If checked, take screen shots for tutorials. Use Save Data to write them to disk.")
    parametersFormLayout.addRow("Enable Screenshots", self.enableScreenshotsFlagCheckBox)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.inputSelector.currentNode() and self.outputSelector.currentNode()

  def onApplyButton(self):
    logic = AnalysisLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    imageThreshold = self.imageThresholdSliderWidget.value
    logic.run(self.inputSelector.currentNode(), self.outputSelector.currentNode(), imageThreshold, enableScreenshotsFlag)

#
# AnalysisLogic
#

class AnalysisLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def hasImageData(self,volumeNode):
    """This is an example logic method that
    returns true if the passed in volume
    node has valid image data
    """
    if not volumeNode:
      logging.debug('hasImageData failed: no volume node')
      return False
    if volumeNode.GetImageData() is None:
      logging.debug('hasImageData failed: no image data in volume node')
      return False
    return True

  def isValidInputOutputData(self, inputVolumeNode, outputVolumeNode):
    """Validates if the output is not the same as input
    """
    if not inputVolumeNode:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not outputVolumeNode:
      logging.debug('isValidInputOutputData failed: no output volume node defined')
      return False
    if inputVolumeNode.GetID()==outputVolumeNode.GetID():
      logging.debug('isValidInputOutputData failed: input and output volume is the same. Create a new volume for output to avoid this error.')
      return False
    return True

  def takeScreenshot(self,name,description,type=-1):
    # show the message even if not taking a screen shot
    slicer.util.delayDisplay('Take screenshot: '+description+'.\nResult is available in the Annotations module.', 3000)

    lm = slicer.app.layoutManager()
    # switch on the type to get the requested window
    widget = 0
    if type == slicer.qMRMLScreenShotDialog.FullLayout:
      # full layout
      widget = lm.viewport()
    elif type == slicer.qMRMLScreenShotDialog.ThreeD:
      # just the 3D window
      widget = lm.threeDWidget(0).threeDView()
    elif type == slicer.qMRMLScreenShotDialog.Red:
      # red slice window
      widget = lm.sliceWidget("Red")
    elif type == slicer.qMRMLScreenShotDialog.Yellow:
      # yellow slice window
      widget = lm.sliceWidget("Yellow")
    elif type == slicer.qMRMLScreenShotDialog.Green:
      # green slice window
      widget = lm.sliceWidget("Green")
    else:
      # default to using the full window
      widget = slicer.util.mainWindow()
      # reset the type so that the node is set correctly
      type = slicer.qMRMLScreenShotDialog.FullLayout

    # grab and convert to vtk image data
    qpixMap = qt.QPixmap().grabWidget(widget)
    qimage = qpixMap.toImage()
    imageData = vtk.vtkImageData()
    slicer.qMRMLUtils().qImageToVtkImageData(qimage,imageData)

    annotationLogic = slicer.modules.annotations.logic()
    annotationLogic.CreateSnapShot(name, description, type, 1, imageData)

  # Generates a sphere model with 2.5cm radius. Used to check validity of reconstruction
  def GenerateSphere(self, center, radius):
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter([0, 0, 0])
    sphere.SetRadius(radius)
    sphere.SetPhiResolution(30)
    sphere.SetThetaResolution(30)
    sphere.Update()
    model = slicer.vtkMRMLModelNode()
    model.SetAndObservePolyData(sphere.GetOutput())
    modelDisplay = slicer.vtkMRMLModelDisplayNode()
    modelDisplay.SetVisibility(True)
    slicer.mrmlScene.AddNode(modelDisplay)
    model.SetAndObserveDisplayNodeID(modelDisplay.GetID())
    slicer.mrmlScene.AddNode(model)

  # Generates the best possible reconstruction, using images taken at all angles
  # Returns:
  #    the ratio of the surface areas and volumes compared to the ground truth sphere
  def BestReconstructionOfSphere(self):
      import SimulateAndReconstruct
      logic = SimulateAndReconstruct.SimulateAndReconstructLogic()

      angles = [(0, -180), (0, -135), (0, -90), (0, -45), (0, 0), (0, 45), (0, 90), (0, 135),
                (45, -180), (45, -135), (45, -90), (45, -45), (45, 0), (45, 45), (45, 90), (45, 135),
                (-45, -180), (-45, -135), (-45, -90), (-45, -45), (-45, 0), (-45, 45), (-45, 90), (-45, 135)]

      contours = []
      for i in range(0, 24):
        contour = logic.Simulator([0,0,0],25, 0)[0]
        contours.append(contour)

      (surfaceArea, Volume) = logic.ReconstructTumour(10, angles, contours)

      self.GenerateSphere([0,0,0],25)

      VolumeSphere = 65449.5
      surfaceAreaSphere = 7854
      volumeRatio = Volume / VolumeSphere
      surfaceAreaRatio = surfaceArea / surfaceAreaSphere

      return (volumeRatio,surfaceAreaRatio)

      # Generates the best possible reconstruction, using images taken at all angles
      # Returns:
      #    the ratio of the surface areas and volumes compared to the ground truth sphere

  def ReconstructionOfSphere(self, NumImages, center, radius, Emax):
    import SimulateAndReconstruct
    logic = SimulateAndReconstruct.SimulateAndReconstructLogic()

    angles = [(-180, 0), (-135, 0), (-90, 0), (-45, 0), (0, 0), (45, 0), (90, 0), (135, 0),
              (-180, 45), (-135, 45), (-90, 45), (-45, 45), (0, 45), (45, 45), (90, 45), (135, 45),
              (-180, -45), (-135, -45), (-90, -45), (-45, -45), (0, -45), (45, -45), (90, -45), (135, -45)]

    #angles = []
    #NumXAngles = int(math.pow(NumImages,0.5))
    #NumZAngles = int(math.ceil((float(NumImages))/NumXAngles))
    #zAngles = numpy.linspace(0, 360, NumZAngles, dtype = int).tolist()
    #xAngles = numpy.linspace(-45, 45, NumXAngles, dtype = int).tolist()
    #for i in range (0,NumXAngles):
    #  for j in range (0,NumZAngles):
    #      angles.append((zAngles[j],xAngles[i]))
    print angles

    contours = logic.Simulator(center,radius,Emax)

    (surfaceArea, Volume) = logic.ReconstructTumour(NumImages, angles, contours)

    self.GenerateSphere(center, radius)

    VolumeSphere = (4.0/3.0)*math.pi*math.pow(radius,3)
    surfaceAreaSphere = 4*math.pi*math.pow(radius,2)
    volumeRatio = Volume / VolumeSphere
    surfaceAreaRatio = surfaceArea / surfaceAreaSphere

    return (volumeRatio, surfaceAreaRatio)

  # Generates a reconstruction while only rotating about the z axis not the x axis
  # Returns:
  #    the ratio of the surface areas and volumes compared to the ground truth sphere
  def OnlyZRotation(self):
      import SimulateAndReconstruct
      logic = SimulateAndReconstruct.SimulateAndReconstructLogic()

      angles = [(0, 0), (45, 0), (90, 0), (135, 0), (180, 0), (225, 0), (270, 0), (315, 0)]

      contours = []
      for i in range(0, 8):
        contour = logic.Simulator([0, 0, 0], 25, 0)[0]
        contours.append(contour)

      (surfaceArea, Volume) = logic.ReconstructTumour(8, angles, contours)

      self.GenerateSphere([0,0,0],25)

      VolumeSphere = 65449.5
      surfaceAreaSphere = 7854
      volumeRatio = Volume / VolumeSphere
      surfaceAreaRatio = surfaceArea / surfaceAreaSphere

      return (volumeRatio,surfaceAreaRatio)

  # Compares how the surface area ratio and volume ratio change as Emax increases
  # Returns: the surface area and volume ratios for each value of Emax
  def ReconstructionWithContourErrors(self,reconstructionType):
      import SimulateAndReconstruct
      logic = SimulateAndReconstruct.SimulateAndReconstructLogic()

      angles = [(0, 0), (0, 45), (0, 90)]#, (0, 135), (0, 180), (0, 225), (0, 270), (0, 315),
                #(45, 0), (45, 45), (45, 90), (45, 135), (45, 180), (45, 225), (45, 270), (45, 315),
                #(-45, 0), (-45, 45), (-45, 90), (-45, 135), (-45, 180), (-45, 225), (-45, 270), (-45, 315)]

      dataTuples = []
      for j in [0,0.05]:#,0.1,0.15]:
          sumVolumeRatio = 0
          sumSurfaceAreaRatio = 0
          for n in range(0,5):
              slicer.mrmlScene.Clear(0)
              contours = []
              '''
              for i in range(0, 1):
              contour = logic.Simulator([0, 0, 0], 25, j)
              contours.append(contour)
              '''
              contours = logic.Simulator([0,0,0],25,j)
              if reconstructionType == "BottomUp":
                  (surfaceArea, Volume) = logic.ReconstructTumour(24, angles, contours)
              else:
                  imagePointsNode = slicer.mrmlScene.GetFirstNodeByName("contour")
                  if imagePointsNode == None:
                      imagePointsNode = slicer.vtkMRMLMarkupsFiducialNode()
                      imagePointsNode.SetName("contour")
                      slicer.mrmlScene.AddNode(imagePointsNode)
                  else:
                      imagePointsNode.RemoveAllMarkups()
                  for i in range(0, len(contours)):
                      imagePointsNode.AddFiducial(contours[i][0], contours[i][1], contours[i][2])
                  for k in range(0, imagePointsNode.GetNumberOfFiducials()):
                      imagePointsNode.SetNthFiducialVisibility(k, False)
                  (surfaceArea, Volume) = logic.ReconstructTumourFromSlabs(angles, imagePointsNode)

              VOLUME_SPHERE_MM3 = 65449.5
              SURFACE_AREA_SPHERE_MM2 = 7854
              sumVolumeRatio += Volume / VOLUME_SPHERE_MM3
              sumSurfaceAreaRatio += surfaceArea / SURFACE_AREA_SPHERE_MM2
          volumeRatio = sumVolumeRatio / 5
          surfaceAreaRatio = sumSurfaceAreaRatio / 5
          dataTuples.append((j,volumeRatio,surfaceAreaRatio))

      self.GenerateSphere([0, 0, 0], 25)

      self.CreateChart(dataTuples)

      return dataTuples

  # Creates a chart to visualize the data from ReconstructionWithContourErrors()
  def CreateChart(self,dataTuples):
    lns = slicer.mrmlScene.GetNodesByClass('vtkMRMLLayoutNode')
    lns.InitTraversal()
    ln = lns.GetNextItemAsObject()
    ln.SetViewArrangement(24)

    cvns = slicer.mrmlScene.GetNodesByClass('vtkMRMLChartViewNode')
    cvns.InitTraversal()
    cvn = cvns.GetNextItemAsObject()

    daNode = slicer.mrmlScene.AddNode(slicer.vtkMRMLDoubleArrayNode())
    emaxVsSurfaceArea = daNode.GetArray()
    emaxVsSurfaceArea.SetNumberOfTuples(4)
    x = range(0, 4)
    for i in range(0,2):
      emaxVsSurfaceArea.SetComponent(i, 0, dataTuples[i][0])
      emaxVsSurfaceArea.SetComponent(i, 1, dataTuples[i][2])
      emaxVsSurfaceArea.SetComponent(i, 2, 0)

    daNode2 = slicer.mrmlScene.AddNode(slicer.vtkMRMLDoubleArrayNode())
    emaxVsVolume = daNode2.GetArray()
    emaxVsVolume.SetNumberOfTuples(4)
    x = range(0, 4)
    for i in range(0, 2):
      emaxVsVolume.SetComponent(i, 0, dataTuples[i][0])
      emaxVsVolume.SetComponent(i, 1, dataTuples[i][1])
      emaxVsVolume.SetComponent(i, 2, 0)

    chartNode = slicer.mrmlScene.AddNode(slicer.vtkMRMLChartNode())

    chartNode.AddArray('Surface Area Ratio', daNode.GetID())
    chartNode.AddArray('Volume Ratio', daNode2.GetID())

    chartNode.SetProperty('default', 'title', 'Surface Area Ratio and Volume Ratio vs Emax')
    chartNode.SetProperty('default', 'xAxisLabel', 'Emax')
    chartNode.SetProperty('default', 'yAxisLabel', 'Reconstruction/Ground Truth (%)')

    cvn.SetChartNodeID(chartNode.GetID())

  def run(self, inputVolume, outputVolume, imageThreshold, enableScreenshots=0):
    """
    Run the actual algorithm
    """

    if not self.isValidInputOutputData(inputVolume, outputVolume):
      slicer.util.errorDisplay('Input volume is the same as output volume. Choose a different output volume.')
      return False

    logging.info('Processing started')

    # Compute the thresholded output volume using the Threshold Scalar Volume CLI module
    cliParams = {'InputVolume': inputVolume.GetID(), 'OutputVolume': outputVolume.GetID(), 'ThresholdValue' : imageThreshold, 'ThresholdType' : 'Above'}
    cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True)

    # Capture screenshot
    if enableScreenshots:
      self.takeScreenshot('AnalysisTest-Start','MyScreenshot',-1)

    logging.info('Processing completed')

    return True


class AnalysisTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    #self.test_Analysis1()
    self.testSlabAnalysis()

  # Tests the ReconstructionWithContourErrors function
  # displays the values for Emax, Surface area ratio and volume ratio
  def test_Analysis1(self):
    logic = AnalysisLogic()
    comparisonData = logic.ReconstructionWithContourErrors("BottomUp")
    for i in range(0,len(comparisonData)):
      print 'Emax: ' + str(comparisonData[i][0])
      print 'Volume Ratio: ' + str(comparisonData[i][1])
      print 'Surface Area Ratio: ' + str(comparisonData[i][2]) + '\n'

  def testSlabAnalysis(self):
    logic = AnalysisLogic()
    comparisonData = logic.ReconstructionWithContourErrors("Slab")
    for i in range(0,len(comparisonData)):
      print 'Emax: ' + str(comparisonData[i][0])
      print 'Volume Ratio: ' + str(comparisonData[i][1])
      print 'Surface Area Ratio: ' + str(comparisonData[i][2]) + '\n'
