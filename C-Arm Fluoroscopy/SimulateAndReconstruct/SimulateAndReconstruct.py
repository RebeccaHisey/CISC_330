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

    self.ImagetoDetectorTransform = slicer.qMRMLNodeComboBox()
    self.ImagetoDetectorTransform.nodeTypes = ['vtkMRMLLinearTransformNode']
    self.ImagetoDetectorTransform.setMRMLScene(slicer.mrmlScene)
    parametersFormLayout.addRow("Image to Detector Transform: ", self.ImagetoDetectorTransform)

    self.DetectortoRASTransform = slicer.qMRMLNodeComboBox()
    self.DetectortoRASTransform.nodeTypes = ['vtkMRMLLinearTransformNode']
    self.DetectortoRASTransform.setMRMLScene(slicer.mrmlScene)
    parametersFormLayout.addRow("Detector to RAS Transform: ", self.DetectortoRASTransform)

    self.tumourContour = slicer.qMRMLNodeComboBox()
    self.tumourContour.nodeTypes = ['vtkMRMLMarkupsFiducialNode']
    self.tumourContour.setMRMLScene(slicer.mrmlScene)
    parametersFormLayout.addRow("Tumour Contour: ", self.tumourContour)

    self.ModelFiducials = slicer.vtkMRMLMarkupsFiducialNode()
    self.ModelFiducials.SetName('Model')
    slicer.mrmlScene.AddNode(self.ModelFiducials)

    self.outputModel = slicer.vtkMRMLModelNode()
    self.outputModel.SetName('tumour')
    slicer.mrmlScene.AddNode(self.outputModel)

    self.outputDisplayModel = slicer.vtkMRMLModelDisplayNode()
    slicer.mrmlScene.AddNode(self.outputDisplayModel)

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
    # Add Button
    #
    self.applyButton = qt.QPushButton("Add")
    self.applyButton.toolTip = "Add a contour to the model."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)

    #
    # Add Button
    #
    self.removeButton = qt.QPushButton("Remove")
    self.removeButton.toolTip = "Remove a contour from the model."
    self.removeButton.enabled = False
    parametersFormLayout.addRow(self.removeButton)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.removeButton.connect('clicked(bool)', self.onRemoveButton)
    self.ImagetoDetectorTransform.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.DetectortoRASTransform.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.tumourContour.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.ImagetoDetectorTransform.currentNode() and self.DetectortoRASTransform.currentNode() and self.tumourContour.currentNode()
    self.removeButton.enabled = self.applyButton.enabled
  def onApplyButton(self):
    logic = SimulateAndReconstructLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    imageThreshold = self.imageThresholdSliderWidget.value
    logic.run(self.ImagetoDetectorTransform.currentNode(), self.DetectortoRASTransform.currentNode(),self.tumourContour.currentNode(), self.ModelFiducials, self.outputModel, self.outputDisplayModel)

  def onRemoveButton(self):
    logic = SimulateAndReconstructLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    imageThreshold = self.imageThresholdSliderWidget.value
    logic.runWithRemove(self.ImagetoDetectorTransform.currentNode(), self.DetectortoRASTransform.currentNode(),self.tumourContour.currentNode(),self.ModelFiducials, self.outputModel, self.outputDisplayModel)

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

  # Generates tumour contours that have been elliptically distorted
  # Parameters:
  #    ctr: the center point of the sphere
  #    radius: the radius of the sphere
  #    numPoints: the number of points to be generated
  #    maxOff: the maximum offset
  # Returns:
  #    dataPoints: the simulated tumour contours
  def TumourContourSimulator(self, ctr, radius, numPoints, maxOff):

    dataPoints = []
    numPointsGenerated = 0
    
    #generate coordinates of points
    xOffset = numpy.random.uniform((1 - maxOff), (1 + maxOff))
    yOffset = numpy.random.uniform((1 - maxOff), (1 + maxOff))
    for n in range(0,numPoints):
        phi = numpy.random.randint(-180,180) #angle displaced around the z axis
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

  # Projects the tumour contour points onto the detector plane, transforms points to (u,v,w) coordinate frame
  # Parameters:
  #    tumour: the simulated tumour contour
  # Returns:
  #    the tumour contour in (u,v,w) coordinates
  def GenerateImage(self,tumour):
    source = numpy.array([[0],[750],[0]])

    detectorP1 = numpy.array([[1],[-750],[0]])
    normal = numpy.array([[0],[1],[0]])

    source = numpy.array([source.item(0), source.item(1), source.item(2)])
    detectorP1 = numpy.array([detectorP1.item(0), detectorP1.item(1), detectorP1.item(2)])
    normal = numpy.array([normal.item(0), normal.item(1), normal.item(2)])

    numPoints = len(tumour)
    projectedPoints = []

    for i in range (0, numPoints):
        point = self.LineAndPlane(source, tumour[i], detectorP1, normal)
        projectedPoints.append(point)

    projectedPoints = self.xyzToUVW(projectedPoints)
    return projectedPoints

  # Creates a reconstruction of a tumour given a set of images and angles
  # Parameters:
  #    numImages: the number of images
  #    angles: the angle that the c-arm is rotated in that image, given in (z,x) pairs
  #    images: the contour images
  # Returns:
  #    the surface area of the reconstruction
  #    the volume of the reconstruction
  def ReconstructTumour(self,numImages, angles, images):
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
          imagePoints = self.uvwToXYZ(angles[i],images[i]) # project points into (x,y,z) coordinates
          for j in range(0,numPoints):
              imagePointSourceLine = source - imagePoints[j]
              imagePointSourceDistance = numpy.linalg.norm(imagePointSourceLine)
              pointSourceLine = imagePointSourceLine/imagePointSourceDistance
              projectedPointSourceDistance = (750*1500)/imagePointSourceDistance
              imageProjectedPointDisance = imagePointSourceDistance - projectedPointSourceDistance
              projectedPoint = imagePoints[j] + imageProjectedPointDisance*pointSourceLine
              reconstructionPoints.append(projectedPoint)
      surfaceArea,Volume = self.createSurface(numImages*numPoints,reconstructionPoints)

      return (surfaceArea,Volume)

  # Creates a surface model on a set of points
  # Parameters:
  #    numPoints: the number of points
  #    dataPoints: the points that will be used to create the surface model
  # Returns:
  #    the surface area and volume of the model
  def createSurface(self,numPoints, datapoints, outputModel, outputDisplayModel):
      points = vtk.vtkPoints()
      cellArray = vtk.vtkCellArray()

      points.SetNumberOfPoints(numPoints)

      for i in range(numPoints):
          #points.SetPoint(i, datapoints[i])
          position = [0, 0, 0]
          datapoints.GetNthFiducialPosition(i, position)
          points.SetPoint(i, position)

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

      #self.tumorModel = slicer.vtkMRMLModelNode()
      outputModel.SetName("tumour")
      outputModel.SetPolyDataConnection(normals.GetOutputPort())
      #modelDisplayNode = slicer.vtkMRMLModelDisplayNode()
      outputDisplayModel.SetColor(0, 1, 0)
      slicer.mrmlScene.AddNode(outputDisplayModel)
      outputModel.SetAndObserveDisplayNodeID(outputDisplayModel.GetID())
      outputModel.Modified()
      slicer.mrmlScene.AddNode(outputModel)
      properties = vtk.vtkMassProperties()
      properties.SetInputData(outputModel.GetPolyData())
      volume = properties.GetVolume()
      surfaceArea = properties.GetSurfaceArea()

      return (surfaceArea,volume)

  # transforms points from xyz coordinates to (u,v,0) coordinates
  # Parameters:
  #    xyzPoints: the points to be transformed
  # Returns:
  #    the xyzPoints given in (u,v,0) coordinates
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

  # transforms points from (u,v,0) coordinates to xyz space
  #  Parameters:
  #    angles: the angle that the image was rotated about the xyz origin
  #    uvwPoints: the points to be transformed
  # Returns:
  #    the (u,v,0) points given in xyz coordinates
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

  # tumour simulator, generates contours and projects them into (u,v,0) space
  # Parameters:
  #    N: the number of images to be taken
  #    Emax: the maximum distortion
  # Returns:
  #    a list of images of the simulated tumour
  def Simulator(self, center, radius, Emax):
      tumor = self.TumourContourSimulator(center,radius,25,Emax)
      image = self.GenerateImage(tumor)
      return image

  def SimulatorWithNImages(self,NumImages):
      for i in range(0,NumImages):
          image = self.Simulator([0, 0, 0], 25, 0.15)
          self.ImageFiducials = slicer.vtkMRMLMarkupsFiducialNode()
          self.ImageFiducials.SetName('Image'+str(i))
          slicer.mrmlScene.AddNode(self.ImageFiducials)
          for k in range(0, len(image)):
              x = image[k][0]
              y = image[k][1]
              z = image[k][2]
              self.ImageFiducials.AddFiducial(x,y,z)
      self.SimulateTransformsForNImages()

  def SimulateTransformsForNImages(self):
    angles = [(0, -180), (0, -135), (0, -90), (0, -45), (0, 0), (0, 45), (0, 90), (0, 135),
              (45, -180), (45, -135), (45, -90), (45, -45), (45, 0), (45, 45), (45, 90), (45, 135),
              (-45, -180), (-45, -135), (-45, -90), (-45, -45), (-45, 0), (-45, 45), (-45, 90), (-45, 135)]

    numAngles = len(angles)
    for i in range(0, numAngles):
            self.ImageToDetector = slicer.vtkMRMLLinearTransformNode()
            self.ImageToDetector.SetName('ImageToDetector'+str(i))
            slicer.mrmlScene.AddNode(self.ImageToDetector)

            self.ImageToDetectorTransform = vtk.vtkTransform()
            self.ImageToDetectorTransform.RotateZ(angles[i][0])
            self.ImageToDetectorTransform.RotateX(angles[i][1])

            self.ImageToDetector.SetAndObserveTransformToParent(self.ImageToDetectorTransform)

            self.DetectorToRAS = slicer.vtkMRMLLinearTransformNode()
            self.DetectorToRAS.SetName('DetectorToRAS' + str(i))
            slicer.mrmlScene.AddNode(self.DetectorToRAS)

            self.DetectorToRASTransform = vtk.vtkTransform()
            self.DetectorToRASTransform.Scale(2,2,2)

            self.DetectorToRAS.SetAndObserveTransformToParent(self.DetectorToRASTransform)



  def run(self, ImagetoDetector, DetectortoRAS, tumourContour, modelFiducials, outputModel, outputDisplayModel):

    logging.info('Processing started')

    tumourContour.SetAndObserveTransformNodeID(ImagetoDetector.GetID())
    ImagetoDetector.SetAndObserveTransformNodeID(DetectortoRAS.GetID())
    #tumourContour.ApplyTransformMatrix(ImagetoDetector.GetMatrixTransformFromParent())
    #tumourContour.ApplyTransformMatrix(DetectortoRAS.GetMatrixTransformFromParent())

    numContourPoints = tumourContour.GetNumberOfFiducials()
    pos = [0,0,0]

    for i in range (0, numContourPoints):
        tumourContour.GetNthFiducialPosition(i,pos)
        modelFiducials.AddFiducial(pos[0],pos[1],pos[2])

    #tumourContour.ApplyTransformMatrix(ImagetoDetector.GetMatrixTransformToParent())
    #tumourContour.ApplyTransformMatrix(DetectortoRAS.GetMatrixTransformToParent())
    print modelFiducials.GetNumberOfFiducials()

    self.createSurface(modelFiducials.GetNumberOfFiducials(),modelFiducials, outputModel, outputDisplayModel)
    for j in range (0, modelFiducials.GetNumberOfFiducials()):
        modelFiducials.SetNthFiducialVisibility(j,False)

    logging.info('Processing completed')

    return True

  def runWithRemove(self, ImagetoDetector, DetectortoRAS, tumourContour, modelFiducials, outputModel, outputDisplayModel):

    logging.info('Processing started')

    numContourPoints = tumourContour.GetNumberOfFiducials()

    #tumourContour.ApplyTransformMatrix(ImagetoDetector.GetMatrixTransformFromParent())
    #tumourContour.ApplyTransformMatrix(DetectortoRAS.GetMatrixTransformFromParent())

    toBeRemoved = []
    for i in range(0, numContourPoints):
        pos = [0, 0, 0]
        tumourContour.GetNthFiducialPosition(i, pos)
        toBeRemoved.append(pos)

    #tumourContour.ApplyTransformMatrix(ImagetoDetector.GetMatrixTransformToParent())
    #tumourContour.ApplyTransformMatrix(DetectortoRAS.GetMatrixTransformToParent())

    self.tempFiducials = slicer.vtkMRMLMarkupsFiducialNode()
    for j in range(0,modelFiducials.GetNumberOfFiducials()):
        pos = [0, 0, 0]
        modelFiducials.GetNthFiducialPosition(j,pos)
        if pos not in toBeRemoved:
            self.tempFiducials.AddFiducial(pos[0],pos[1],pos[2])

    modelFiducials.Copy(self.tempFiducials)

    print modelFiducials.GetNumberOfFiducials()

    self.createSurface(modelFiducials.GetNumberOfFiducials(), modelFiducials, outputModel, outputDisplayModel)

    for j in range(0, modelFiducials.GetNumberOfFiducials()):
        modelFiducials.SetNthFiducialVisibility(j, False)

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

  def TestSimulatorWithOneImage(self):
      logic = SimulateAndReconstructLogic()
      logic.SimulatorWithOneImage()

  def TestSimulatorWithTwoImages(self):
      logic = SimulateAndReconstructLogic()
      logic.SimulatorWithTwoImages()



