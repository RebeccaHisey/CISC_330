import os
import unittest
import vtk, qt, ctk, slicer
import numpy
import math
from slicer.ScriptedLoadableModule import *
import logging

#
# SimAndRecon
#

class SimAndRecon(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "SimAndRecon" # TODO make this more human readable by adding spaces
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
# SimAndReconWidget
#

class SimAndReconWidget(ScriptedLoadableModuleWidget):
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
    logic = SimAndReconLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    imageThreshold = self.imageThresholdSliderWidget.value
    logic.run(self.inputSelector.currentNode(), self.outputSelector.currentNode(), imageThreshold, enableScreenshotsFlag)

#
# SimAndReconLogic
#

class SimAndReconLogic(ScriptedLoadableModuleLogic):
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
      
  # Computes the error between the initial values and the computed values
  # Parameters:
  #    initial: the list of initial values
  #    computed: the list of computed values
  # Returns:
  #    error: the standard deviation from the initial values
  def findError(self,initial, computed):

    #calculate the difference between the initial values and the computed values
    #square these values and sum them
    sumSquare = 0
    for i in range(0,len(initial)):
        sumSquare = sumSquare + math.pow((computed[0] - initial[0]),2)

    #find the std deviation by taking the square root and rounds to 3 decimal places
    error = math.sqrt(sumSquare / len(initial))
    error = round(error,3)
    
    return error


  # Generates points on the upper half of a sphere, then randomly offsets them
  # Parameters:
  #    ctr: the center point of the sphere
  #    radius: the radius of the sphere
  #    numPoints: the number of points to be generated
  #    maxOff: the maximum offset
  # Returns:
  #    dataPoints: the simulated datapoints on the sphere
  def sphereSim(self, ctr, radius, numPoints, maxOff):
    
    dataPoints = []

    #generate coordinates of points
    p = numpy.linspace(0.5,1,numPoints).tolist()
    for n in range(0,numPoints):
        xOffset = numpy.random.uniform((-maxOff), maxOff)
        yOffset = numpy.random.uniform((-maxOff), maxOff)
        zOffset = numpy.random.uniform((-maxOff), maxOff)
        phi = numpy.random.randint(-180,180) #angle displaced around the z axis
        psi = math.acos(2*p[n] - 1) #angle rotated down from the z axis
        x = radius*math.cos(phi)*math.sin(psi) + ctr[0] + xOffset
        y = radius*math.sin(phi)*math.sin(psi) + ctr[1] + yOffset
        z = radius*math.cos(psi) + ctr[2] + zOffset
        
        dataPoints.append(numpy.array([x,y,z]))
    
    return dataPoints
  
  # Simulates and reconstructs a sphere
  # Parameters:
  #    ctr: the center point of the sphere
  #    radius: the radius of the sphere
  #    numPoints: the number of points to be generated on the sphere
  #    maxOff: the maximum offset
  # Returns:
  #    cctr: the reconstructed center point
  #    cradius: the radius of the reconstruction
  #    error: the standard deviation from the original data
  def sphereRecon(self, ctr, radius, numPoints, maxOff):
    logic = SimAndReconLogic()
    
    #create a list of data points on the sphere
    data = logic.sphereSim(ctr, radius, numPoints, maxOff)
  
    # see github wiki for explanation
    M = numpy.matrix([data[0][0], data[0][1], data[0][2], 1])
    b = numpy.matrix([(-1)*(numpy.dot(data[0],data[0]))])
    for n in range(1,numPoints):
        M = numpy.append(M, [[data[n][0], data[n][1], data[n][2], 1]], axis = 0)
        b = numpy.append(b, [[(-1)*(numpy.dot(data[n],data[n]))]], axis = 0)

    ctrAndSigma = numpy.matrix.getI(M)*b
    
    cctr = numpy.array([ctrAndSigma.item(0), ctrAndSigma.item(1), ctrAndSigma.item(2)])
    sigma = ctrAndSigma.item(3)
        
    cradius = round(math.sqrt(numpy.dot(cctr,cctr) - sigma),1)
    
    cctr = [round(cctr[0],1), round(cctr[1],1), round(cctr[2],1)]
    
    initial = [ctr[0], ctr[1], ctr[2], radius]
    computed = [cctr[0], cctr[1], cctr[2], cradius]

    error = logic.findError (initial, computed)
    
    return (cctr, cradius, error)
        
  def run(self, inputVolume, outputVolume, imageThreshold, enableScreenshots=0):
    """
    Run the actual algorithm
    """
    return True


class SimAndReconTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)
    self.sphereFiducials = slicer.vtkMRMLMarkupsFiducialNode()
    self.sphereFiducials.SetName('Sphere')
    slicer.mrmlScene.AddNode(self.sphereFiducials)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    print('Test Sphere simulation and reconstruction: ')
    self.testSphereSimAndRecon()

  # Tests the logic function sphereRecon
  def testSphereSimAndRecon(self):
    ctr = numpy.array([0,0,0])
    print ('  Original center: ' + str(ctr))
    print ('  Original radius: ' + str(100) + '\n')
    logic = SimAndReconLogic()
    # repeats test for all max offsets from 1-10
    for x in range(0,11):
        data = logic.sphereRecon(ctr, 100, 25, x)
        print ('  Max Offset = ' + str(x))
        print ('\tCenter: ' + str(data[0]))
        print ('\tRadius: ' + str(data[1]))
        print ('\tError: ' + str(data[2]))
    



                     
                     
                     
                     
                     
                     
    