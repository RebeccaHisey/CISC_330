import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy
import math
import random

#
# SimulateAndReconstruct
#

class SimulateAndReconstruct(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "SimulateAndReconstruct" # TODO make this more human readable by adding spaces
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
# SimulateAndReconstructWidget
#

class SimulateAndReconstructWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
      ScriptedLoadableModuleWidget.__init__(self, parent)
      self.logic = SimulateAndReconstructLogic()

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
    logic = SimulateAndReconstructLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    imageThreshold = self.imageThresholdSliderWidget.value
    logic.run(self.inputSelector.currentNode(), self.outputSelector.currentNode(), imageThreshold, enableScreenshotsFlag)

#
# SimulateAndReconstructLogic
#

class SimulateAndReconstructLogic(ScriptedLoadableModuleLogic):
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

  # Generates points on the upper half of a sphere, then randomly offsets them
  # Parameters:
  #    ctr: the center point of the sphere
  #    radius: the radius of the sphere
  #    numPoints: the number of points to be generated
  #    maxOff: the maximum offset
  # Returns:
  #    dataPoints: the simulated datapoints on the sphere
  def TumourContourSimulator(self, ctr, radius, numPoints, maxOff):

    dataPoints = []
    numPointsGenerated = 0
    
    #generate coordinates of points
    p = numpy.linspace(0,1,numPoints).tolist()
    for n in range(0,numPoints):
        xOffset = numpy.random.uniform((1-maxOff), (1+maxOff))
        yOffset = numpy.random.uniform((1-maxOff), (1+maxOff))
        zOffset = numpy.random.uniform((1-maxOff), (1+maxOff))
        phi = numpy.random.randint(-180,180) #angle displaced around the z axis
        psi = math.acos(2*p[n] - 1) #angle rotated down from the z axis
        x = (radius*math.cos(phi) + ctr[0]) * xOffset
        y = 0
        z = (radius*math.sin(phi) + ctr[1]) * yOffset
        if dataPoints == []:
            dataPoints.append(numpy.array([x,y,z]))
        else:
            i = 0
            while (i < numPointsGenerated and x > dataPoints[i][0]):
                i+= 1
            if i == numPointsGenerated:
                dataPoints.append(numpy.array([x, y, z]))
            else:
                dataPoints.insert(i, numpy.array([x,y,z]))
        numPointsGenerated += 1

    return dataPoints

  # finds the intersection between a line (specified by 2 points) and
  # a plane (specified by a point and a normal vector)
  #  Parameters:
  #    l0: an initial point on a line
  #    l1: a second point on the line
  #    p0: a point on the plane
  #    n: the normal vector of the plane
  # Returns:
  #    ci: the computed intersection point between the line and the plane
  def LineAndPlane(self, l0, l1, p0, n):
      # compute the direction vector of the line and normalizes it
      l = l1 - l0
      length = numpy.linalg.norm(l)
      if length == 0:
        print ('The line supplied has length 0')
        return
      l = l / length

      # checks to see if the line and the plane are parallel
      denom = numpy.dot(l, n)
      if denom == 0:
        return False
      numer = numpy.dot((p0 - l0), n)
      d = numer / denom

      ci = d * l + l0  # Computed intesection

      return ci

  def GenerateImage(self,tumour):
    logic = SimulateAndReconstructLogic()

    source = numpy.array([[0],[750],[0]])

    detectorP1 = numpy.array([[1],[-750],[0]])
    normal = numpy.array([[0],[1],[0]])

    source = numpy.array([source.item(0), source.item(1), source.item(2)])
    detectorP1 = numpy.array([detectorP1.item(0), detectorP1.item(1), detectorP1.item(2)])
    normal = numpy.array([normal.item(0), normal.item(1), normal.item(2)])

    numPoints = len(tumour)
    projectedPoints = []

    for i in range (0, numPoints):
        point = logic.LineAndPlane(source, tumour[i], detectorP1, normal)
        projectedPoints.append(point)

    projectedPoints = logic.xyzToUVW(projectedPoints)
    return projectedPoints

  def ReconstructTumour(self,numImages, angles, images):
      logic = SimulateAndReconstructLogic()
      reconstructionPoints = []
      for i in range(0,numImages):
          source = numpy.array([[0], [750], [0]])
          xRotationMatrix = numpy.matrix([[1, 0, 0],
                                          [0, numpy.cos(angles[i][1]), numpy.sin(angles[i][1])],
                                          [0, (-1) * numpy.sin(angles[i][1]), numpy.cos(angles[i][1])]])

          zRotationMatrix = numpy.matrix([[numpy.cos(angles[i][0]), numpy.sin(angles[i][0]), 0],
                                          [(-1) * numpy.sin(angles[i][0]), numpy.cos(angles[i][0]), 0],
                                          [0, 0, 1]])
          source = zRotationMatrix * xRotationMatrix * source
          source = numpy.array([source.item(0), source.item(1), source.item(2)])
          numPoints = len(images[i])
          imagePoints = logic.uvwToXYZ(angles[i],images[i])
          for j in range(0,numPoints):
              imagePointSourceLine = source - imagePoints[j]
              imagePointSourceDistance = numpy.linalg.norm(imagePointSourceLine)
              pointSourceLine = imagePointSourceLine/imagePointSourceDistance
              projectedPointSourceDistance = (750*1500)/imagePointSourceDistance
              imageProjectedPointDisance = imagePointSourceDistance - projectedPointSourceDistance
              projectedPoint = imagePoints[j] + imageProjectedPointDisance*pointSourceLine
              reconstructionPoints.append(projectedPoint)
      surfaceArea,Volume = logic.createSurface(numImages*numPoints,reconstructionPoints)

      return (surfaceArea,Volume)

  def createSurface(self,numPoints, datapoints):
      points = vtk.vtkPoints()
      cellArray = vtk.vtkCellArray()

      points.SetNumberOfPoints(numPoints)

      for i in range(numPoints):
          points.SetPoint(i, datapoints[i])

      cellArray.InsertNextCell(numPoints)
      for i in range(numPoints):
          cellArray.InsertCellPoint(i)

      pointPolyData = vtk.vtkPolyData()
      pointPolyData.SetLines(cellArray)
      pointPolyData.SetPoints(points)

      delaunay = vtk.vtkDelaunay3D()

      sphere = vtk.vtkCubeSource()
      glyph = vtk.vtkGlyph3D()
      glyph.SetInputData(pointPolyData)
      glyph.SetSourceConnection(sphere.GetOutputPort())
      delaunay.SetInputConnection(glyph.GetOutputPort())

      surfaceFilter = vtk.vtkDataSetSurfaceFilter()
      surfaceFilter.SetInputConnection(delaunay.GetOutputPort())

      normals = vtk.vtkPolyDataNormals()
      normals.SetInputConnection(surfaceFilter.GetOutputPort())
      normals.SetFeatureAngle(100.0)

      self.tumorModel = slicer.vtkMRMLModelNode()
      self.tumorModel.SetName("tumour")
      self.tumorModel.SetPolyDataConnection(normals.GetOutputPort())
      modelDisplayNode = slicer.vtkMRMLModelDisplayNode()
      modelDisplayNode.SetColor(0, 1, 0)
      slicer.mrmlScene.AddNode(modelDisplayNode)
      self.tumorModel.SetAndObserveDisplayNodeID(modelDisplayNode.GetID())
      self.tumorModel.Modified()
      slicer.mrmlScene.AddNode(self.tumorModel)
      properties = vtk.vtkMassProperties()
      properties.SetInputData(self.tumorModel.GetPolyData())
      volume = properties.GetVolume()
      surfaceArea = properties.GetSurfaceArea()

      return (surfaceArea,volume)

  def rotateContour(self, zRotation, xRotation, contour):
      xRotationMatrix = numpy.matrix([[1, 0, 0],
                                      [0, numpy.cos(xRotation), numpy.sin(xRotation)],
                                      [0, (-1) * numpy.sin(xRotation), numpy.cos(xRotation)]])
      zRotationMatrix = numpy.matrix([[numpy.cos(zRotation), numpy.sin(zRotation), 0],
                                      [(-1) * numpy.sin(zRotation), numpy.cos(zRotation), 0],
                                      [0, 0, 1]])
      matrix = numpy.matrix([contour[0]])
      for i in range(1,len(contour)):
          matrix = numpy.append(matrix,[contour[i]], axis = 0)
      matrixT = numpy.transpose(matrix)
      rotate = zRotationMatrix*xRotationMatrix*matrixT
      rotate = numpy.transpose(rotate)
      rotatedContour = []
      for i in range(0, len(contour)):
          point = [rotate.item(i*3),rotate.item(1+i*3),rotate.item(2+i*3)]
          rotatedContour.append(point)
      return rotatedContour

  def xyzToUVW(self,xyzPoints):
      scaleMatrix = numpy.matrix([[1,0,0],
                                  [0,0,1],
                                  [0,0,0]])
      transformedPoints = []
      for i in range(0,len(xyzPoints)):
          point = numpy.array([[xyzPoints[i][0]],
                               [xyzPoints[i][1]],
                               [xyzPoints[i][2]]])
          transformedPoint = scaleMatrix*point
          transformedPoint = [transformedPoint.item(0),transformedPoint.item(1),transformedPoint.item(2)]
          transformedPoints.append(transformedPoint)
      return transformedPoints

  def uvwToXYZ(self,angles, uvwPoints):
      scaleMatrix = numpy.matrix([[1, 0, 0],
                                  [0, 0, 1],
                                  [0, 1, 0]])
      translationVector = numpy.array([[0],[-750],[0]])
      zRotationMatrix = numpy.matrix([[numpy.cos(angles[0]), numpy.sin(angles[0]), 0],
                                      [(-1) * numpy.sin(angles[0]), numpy.cos(angles[0]), 0],
                                      [0, 0, 1]])
      xRotationMatrix = numpy.matrix([[1, 0, 0],
                                      [0, numpy.cos(angles[1]), numpy.sin(angles[1])],
                                      [0, (-1) * numpy.sin(angles[1]), numpy.cos(angles[1])]])
      transformedPoints = []
      for i in range(0, len(uvwPoints)):
          point = numpy.array([[uvwPoints[i][0]],
                               [uvwPoints[i][1]],
                               [uvwPoints[i][2]]])
          transformedPoint = zRotationMatrix*xRotationMatrix*((scaleMatrix * point) + translationVector)
          transformedPoint = [transformedPoint.item(0), transformedPoint.item(1), transformedPoint.item(2)]
          transformedPoints.append(transformedPoint)
      return transformedPoints

  def Simulator(self,N,Emax):
      logic = SimulateAndReconstructLogic()
      center = numpy.array([0, 0, 0])
      tumor = logic.TumourContourSimulator(center,25,25,Emax)
      images = []
      for i in range(0,N):
          image = logic.GenerateImage(tumor)
          images.append(image)
      return images

  def SimulatorWithOneImage(self):
      self.ImageFiducials = slicer.vtkMRMLMarkupsFiducialNode()
      self.ImageFiducials.SetName('Image')
      slicer.mrmlScene.AddNode(self.ImageFiducials)

      logic = SimulateAndReconstructLogic()
      images = logic.Simulator(1,0)

      for k in range(0, len(images[0])):
          x = images[0][k][0]
          y = images[0][k][1]
          z = images[0][k][2]
          self.ImageFiducials.AddFiducial(x, y, z)

  def SimulatorWithTwoImages(self):
      self.Image1Fiducials = slicer.vtkMRMLMarkupsFiducialNode()
      self.Image1Fiducials.SetName('Image1')
      slicer.mrmlScene.AddNode(self.Image1Fiducials)
      self.Image2Fiducials = slicer.vtkMRMLMarkupsFiducialNode()
      self.Image2Fiducials.SetName('Image2')
      #self.Image2Fiducials.colour(0,1,0)
      slicer.mrmlScene.AddNode(self.Image2Fiducials)

      logic = SimulateAndReconstructLogic()
      images = logic.Simulator(2, 0.15)
      image1 = images[0]
      image2 = images[1]

      for k in range(0, len(image1)):
          x = images[0][k][0]
          y = images[0][k][1]
          z = images[0][k][2]
          self.Image1Fiducials.AddFiducial(x,y,z)

      for k in range(0, len(image2)):
          x = images[0][k][0]
          y = images[0][k][1]
          z = images[0][k][2]
          self.Image2Fiducials.AddFiducial(x , y ,z)


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
      self.takeScreenshot('SimulateAndReconstructTest-Start','MyScreenshot',-1)

    logging.info('Processing completed')

    return True


class SimulateAndReconstructTest(ScriptedLoadableModuleTest):
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
    self.TestSimulatorWithOneImage()
    self.TestSimulatorWithTwoImages()
    #self.TestReconstructor()
    self.TestReconstructor2()

  def TestSimulatorWithOneImage(self):
      logic = SimulateAndReconstructLogic()
      logic.SimulatorWithOneImage()

  def TestSimulatorWithTwoImages(self):
      logic = SimulateAndReconstructLogic()
      logic.SimulatorWithTwoImages()

  def TestReconstructor(self):
          logic = SimulateAndReconstructLogic()

          angles = [(0, 0), (0, 45), (0, 90), (0, 135), (0, 180), (0, 225), (0, 270), (0, 315), (-45, 0), (0, 45)]

          contours = []
          for i in range(0, 10):
              contour = logic.Simulator(1, 0)[0]
              contours.append(contour)

          for i in range(0, 10):
              contours[i] = logic.rotateContour(angles[i][0], angles[i][1], contours[i])

          (surfaceArea, Volume) = logic.ReconstructTumour(10, angles, contours)
          VolumeSphere = 65449.5
          surfaceAreaSphere = 7854
          volumeRatio = Volume/VolumeSphere
          surfaceAreaRatio = surfaceArea/surfaceAreaSphere
          print 'Surface Area ratio: ' + str(surfaceAreaRatio)
          print 'Volume Ratio: ' + str(volumeRatio)


