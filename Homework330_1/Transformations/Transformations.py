import os
import unittest
import vtk, qt, ctk, slicer
import numpy
import math
from slicer.ScriptedLoadableModule import *
import logging

#
# Transformations
#

class Transformations(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Transformations" # TODO make this more human readable by adding spaces
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
# TransformationsWidget
#

class TransformationsWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

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
    #parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

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
    #parametersFormLayout.addRow("Input Volume: ", self.inputSelector)

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
    #parametersFormLayout.addRow("Output Volume: ", self.outputSelector)

    #
    # threshold value
    #
    self.imageThresholdSliderWidget = ctk.ctkSliderWidget()
    self.imageThresholdSliderWidget.singleStep = 0.1
    self.imageThresholdSliderWidget.minimum = -100
    self.imageThresholdSliderWidget.maximum = 100
    self.imageThresholdSliderWidget.value = 0.5
    self.imageThresholdSliderWidget.setToolTip("Set threshold value for computing the output image. Voxels that have intensities lower than this value will set to zero.")
    #parametersFormLayout.addRow("Image threshold", self.imageThresholdSliderWidget)

    #
    # check box to trigger taking screen shots for later use in tutorials
    #
    self.enableScreenshotsFlagCheckBox = qt.QCheckBox()
    self.enableScreenshotsFlagCheckBox.checked = 0
    self.enableScreenshotsFlagCheckBox.setToolTip("If checked, take screen shots for tutorials. Use Save Data to write them to disk.")
    #parametersFormLayout.addRow("Enable Screenshots", self.enableScreenshotsFlagCheckBox)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    #parametersFormLayout.addRow(self.applyButton)

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
    logic = TransformationsLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    imageThreshold = self.imageThresholdSliderWidget.value
    logic.run(self.inputSelector.currentNode(), self.outputSelector.currentNode(), imageThreshold, enableScreenshotsFlag)

#
# TransformationsLogic
#

class TransformationsLogic(ScriptedLoadableModuleLogic):
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

  # Computes an orthonormal coordinate system centered at the center of a triangle
  #  using the Gram-Schmidt process
  # Parameters:
  #    A, B, C: 3 points in a triangle
  # Returns:
  #    e1,e2,e3: the orthonormal base vectors
  #    ctr = the center of the triangle given by A, B and C
  def OrthoNormalCoordSystem(self, A, B, C):
      
    # find the center of gravity
    ctrX = ((A[0]+B[0]+C[0])/3)
    ctrY = ((A[1]+B[1]+C[1])/3)
    ctrZ = ((A[2]+B[2]+C[2])/3)
    ctr = numpy.array([ctrX,ctrY,ctrZ])
    
    #find 3 vectors from the center of gravity to each of the points
    x = A - ctr
    y = B - ctr
    z = C - ctr

    # compute the basis vectors using Gram-Schmidt process
    length = numpy.linalg.norm(x)
    if length == 0:
        e1 = x
    else:
        e1 = x / length

    e2 = (y - ((numpy.dot(e1,y)*e1)))
    lengthE2 = numpy.linalg.norm(e2)
    if lengthE2 == 0: # vector space is spanned by less than 3 vectors
        e2 = (z - (numpy.dot(z,e1)*(e1)))
        lengthE2 = numpy.linalg.norm(e2)
        if lengthE2 == 0:
            print ('Error: Vector space is spanned by only 1 vector')
            return
    else:
        e2 = e2 / lengthE2

    e3 = (z - (numpy.dot(e1,z)*e1) - (numpy.dot(e2,z)*e2))
    lengthE3 = numpy.linalg.norm(e3)
    if lengthE3 == 0: #vector space is spanned by only 2 vectors
        e3 = numpy.cross(e1,e2)
        lengthE3 = numpy.linalg.norm(e3)
        if lengthE3 == 0:
            print ('Error: e1, e2 parallel')
            return
        e3 = e3 / lengthE3
        print ('  Vector space provided by points is spanned by only 2 vectors')
        print ('  E3 is a unit vector perpendicular to e1 and e2')
    else:
        e3 = e3 / lengthE3

    return (e1, e2, e3, ctr)
    
  def run(self, inputVolume, outputVolume, imageThreshold, enableScreenshots=0):
    """
    Run the actual algorithm
    """

    return True


class TransformationsTest(ScriptedLoadableModuleTest):
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
    print('Find Orthonormal Coordinate system:')
    self.testOrthoNormalCoordSystem()

  # Tests the logic function OrthoNormalCoordSystem
  def testOrthoNormalCoordSystem(self):
    A = numpy.array([0,5,17])
    B = numpy.array([4,0,27])
    C = numpy.array([1,9,11])
    
    #The points of the triangle
    print ('  Point A: ' + str(A))
    print ('  Point B: ' + str(B))
    print ('  Point C: ' + str(C))

    #The known center point
    ctr = [10,10,10]

    logic = TransformationsLogic()
    e1,e2,e3,ctr = logic.OrthoNormalCoordSystem(A, B, C)
    
    #dot products and lengths printed to show that it is an orthonormal coordinate system
    print ('\n  e1: ' + str([round(e1[0],1),round(e1[1],1),round(e1[2],1)]))
    print ('\tLength e1: ' + str(round(numpy.linalg.norm(e1),1)))
    print ('  e2: ' + str([round(e2[0],1),round(e2[1],1),round(e2[2],1)]))
    print ('\tLength e2: ' + str(round(numpy.linalg.norm(e2),1)))
    print ('  e3: ' + str([round(e3[0],1),round(e3[1],1),round(e3[2],1)]))
    print ('\tLength e3: ' + str(round(numpy.linalg.norm(e3),1)))
    print ('\n  Center point: ' + str(ctr))
    print ('\n  Dot product e1 & e2: ' + str(round(numpy.dot(e1,e2),1)))
    print ('  Dot product e1 & e3: ' + str(round(numpy.dot(e1,e3),1)))
    print ('  Dot product e2 & e3: ' + str(round(numpy.dot(e2,e3),1)))








