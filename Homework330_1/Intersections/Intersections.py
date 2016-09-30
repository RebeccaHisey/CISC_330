import os
import unittest
import vtk, qt, ctk, slicer
import numpy
import math
from slicer.ScriptedLoadableModule import *
import logging

#
# Intersections
#

class Intersections(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Intersections" # TODO make this more human readable by adding spaces
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
# IntersectionsWidget
#

class IntersectionsWidget(ScriptedLoadableModuleWidget):
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
    logic = IntersectionsLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    imageThreshold = self.imageThresholdSliderWidget.value
    logic.run(self.inputSelector.currentNode(), self.outputSelector.currentNode(), imageThreshold, enableScreenshotsFlag)

#
# IntersectionsLogic
#

class IntersectionsLogic(ScriptedLoadableModuleLogic):
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

  def run(self, l0, l1, p0, n):
    """
    Run the actual algorithm
    """
    return True
  
  def findError(self,distances):
    avg = 0
    length = len(distances)
    for d in distances:
        avg = avg + d
    avg = avg / length
    
    sumSquareD = 0
    for d in distances:
        sumSquareD = sumSquareD + math.pow((d - avg),2)

    stdDev = math.sqrt(sumSquareD / length)
    return stdDev
        

  # finds the intersection between a line (specified by 2 points) and
  #   a plane (specified by a point and a normal vector)
  def LineAndPlane(self, l0, l1, p0, n):
    
    l = l1 - l0
    l = l / numpy.linalg.norm(l) #normalizes the direction vector of the line
    
    #checks to see if the line and the plane are parallel
    denom = numpy.dot(l, n)
    if denom == 0:
        print('No intersection! Line Parallel to plane')
        return

    numer = numpy.dot((p0 - l0), n)
    d = numer / denom

    ci = d*l + l0 #Computed intesection

    return ci

  def TwoLines(self, l0_1, l1_1, l0_2, l1_2):

    l_1 = l1_1 - l0_1
    l_1 = l_1 / numpy.linalg.norm(l_1)
    
    l_2 = l1_2 - l0_2
    l_2 = l_2 / numpy.linalg.norm(l_2)

    l_3 = numpy.cross(l_1, l_2)
    l_3 = l_3 / numpy.linalg.norm(l_3)

    vMatrix = numpy.matrix([[(-1)*l_1[0], l_2[0], l_3[0]],
                            [(-1)*l_1[1], l_2[1], l_3[1]],
                            [(-1)*l_1[2], l_2[2], l_3[2]]])

    pMatrix = numpy.matrix([[l0_1[0] - l0_2[0]],
                            [l0_1[1] - l0_2[1]],
                            [l0_1[2] - l0_2[2]]])

    t = (numpy.matrix.getI(vMatrix))*pMatrix

    L1 = l0_1 + t[0]*l_1
    L2 = l0_2 + t[1]*l_2

    M = (L1 + L2)/2
    
    avgDistance = numpy.linalg.norm(M - L1) / 2

    return (M,avgDistance)

  def NLines(self, numLines, listOfLines):
    if numLines <= 1:
        print ('There must be more than 1 line')
        return
    logic = IntersectionsLogic()
    
    iAndDistance = []

    for i in range(0,2*numLines,2):
        l0_1 = listOfLines[i]
        l1_1 = listOfLines[i+1]
        for j in range((i+2),2*numLines,2):
            l0_2 = listOfLines[j]
            l1_2 = listOfLines[j+1]
            iAndDistance = iAndDistance + [logic.TwoLines(l0_1, l1_1, l0_2, l1_2)]
            

    numIntersections = len(iAndDistance)
    
    xCoord = 0
    yCoord = 0
    zCoord = 0
    allDistances = []

    for n in range(0, numIntersections):
        xCoord += iAndDistance[n][0].item(0)
        yCoord += iAndDistance[n][0].item(1)
        zCoord += iAndDistance[n][0].item(2)
        allDistances.append(iAndDistance[n][1])
    

    xCoord = xCoord / numIntersections
    yCoord = yCoord / numIntersections
    zCoord = zCoord / numIntersections

    symInt = [xCoord, yCoord, zCoord]
    error = logic.findError(allDistances)

    return (symInt, error)

  def LineAndEllipsoid(self, l0, l1, a, b, c):
    l = l1 - l0
    l = l / numpy.linalg.norm(l)
    Px = l0[0]
    Py = l0[1]
    Pz = l0[2]

    #values to plug into quadratic formula to solve for t
    u = (math.pow(b*c*l[0],2)) + (math.pow(a*c*l[1],2)) + (math.pow(a*b*l[2], 2))
    v = (2*math.pow(b*c,2)*Px*l[0]) + (2*math.pow(a*c,2)*Py*l[1]) + (2*math.pow(a*b,2)*Pz*l[2])
    w = (math.pow(b*c*Px,2)) + (math.pow(a*c*Py,2)) + (math.pow(a*b*Pz,2)) - (math.pow(a*b*c,2))

    try:
        t1 = (-v + math.sqrt(math.pow(v,2) - 4*u*w)) / (2*u)
        t2 = (-v - math.sqrt(math.pow(v,2) - 4*u*w)) / (2*u)
    except ValueError:
        print ('No intersection!')
        return False

    Lx1 = Px + (t1)*l[0]
    Ly1 = Py + (t1)*l[1]
    Lz1 = Pz + (t1)*l[2]

    i1 = [round(Lx1), round(Ly1), round(Lz1)]

    Lx2 = Px + (t2)*l[0]
    Ly2 = Py + (t2)*l[1]
    Lz2 = Pz + (t2)*l[2]

    i2 = [round(Lx2), round(Ly2), round(Lz2)]

    return (i1, i2)


class IntersectionsTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)
  
    self.lineFiducials = slicer.vtkMRMLMarkupsFiducialNode()
    self.lineFiducials.SetName('Line')
    slicer.mrmlScene.AddNode(self.lineFiducials)
    #self.lineFiducials.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onFiducialModified)
    self.lineFiducials.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onLineModified)
  
    self.planeFiducials = slicer.vtkMRMLMarkupsFiducialNode()
    self.planeFiducials.SetName('Plane')
    slicer.mrmlScene.AddNode(self.planeFiducials)
    self.planeFiducials.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onFiducialModified)
  
    self.intersection = slicer.vtkMRMLMarkupsFiducialNode()
    self.intersection.SetName('Int')
    slicer.mrmlScene.AddNode(self.intersection)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.testLineAndPlaneIntersection()
    self.setUp()
    self.testTwoLines()
    self.setUp()
    self.testNLines()
    self.setUp()
    self.testLineAndEllipsoid()

  # Tests the logic function LineAndPlane
  def testLineAndPlaneIntersection(self):
    i = numpy.array([2,3,0]) #Known intersection
    l = numpy.array([0,0,1])
    l0 = i - 10*l
    l1 = i - 20*l
    p0 = numpy.array([1,-20, 0])
    n = numpy.array([0,0,1])

    logic = IntersectionsLogic()

    self.lineFiducials.AddFiducial(l0[0], l0[1], l0[2])
    self.lineFiducials.AddFiducial(l1[0], l1[1], l1[2])
    self.planeFiducials.AddFiducial(p0[0], p0[1], p0[2])
    self.planeFiducials.AddFiducial(25,0,0)
    self.planeFiducials.AddFiducial(0,25,0)

    ci = logic.LineAndPlane(l0, l1, p0, n) #Computed intersection
    
    self.intersection.AddFiducial(ci[0],ci[1],ci[2])

    print('Original: ' + str(i))
    print('Computed: ' + str(ci))

    # compares the known intersection to the computed value to see if they are the same
    for x in range(0,3):
        if i[x] != ci[x]:
            print('Test Failed')
            return

    print('Test Passed!')

  # recomputes the intersection point when the line points or plane points are
  #   modified in slicer
  def onFiducialModified(self, caller, eventid):
    
    # initialize all points
    l0 = numpy.array([0,0,0], dtype = numpy.float32)
    l1 = numpy.array([0,0,0], dtype = numpy.float32)
    p0 = numpy.array([0,0,0], dtype = numpy.float32)
    p1 = numpy.array([0,0,0], dtype = numpy.float32)
    p2 = numpy.array([0,0,0], dtype = numpy.float32)

    # find coordinates of new line points
    self.lineFiducials.GetNthFiducialPosition(0,l0)
    self.lineFiducials.GetNthFiducialPosition(1,l1)

    # find the coordinates of the plane points
    self.planeFiducials.GetNthFiducialPosition(0,p0)
    self.planeFiducials.GetNthFiducialPosition(1,p1)
    self.planeFiducials.GetNthFiducialPosition(2,p2)

    # recompute the normal of the plane
    n = numpy.cross((p1-p0), (p2-p1))
    n = n / numpy.linalg.norm(n)

    # calculate the intersection point
    logic = IntersectionsLogic()
    i = logic.LineAndPlane(l0,l1,p0,n)

    self.intersection.SetNthFiducialPosition(0, i[0], i[1], i[2])

  def testTwoLines(self):
    i = numpy.array([1,1,0])
    
    l0_1 = numpy.array([10,0,0])
    dirVec1 = l0_1 - i
    dirVec1 = dirVec1 / numpy.linalg.norm(dirVec1)
    l1_1 = 10*dirVec1 + l0_1

    l0_2 = numpy.array([3,4,0])
    dirVec2 = l0_2 - i
    dirVec2 = dirVec2/ numpy.linalg.norm(dirVec2)
    l1_2 = 10*dirVec2 + l0_2

    self.lineFiducials.AddFiducial(l0_1[0], l0_1[1], l0_1[2])
    self.lineFiducials.AddFiducial(l1_1[0], l1_1[1], l1_1[2])
    self.lineFiducials.AddFiducial(l0_2[0], l0_2[1], l0_2[2])
    self.lineFiducials.AddFiducial(l1_2[0], l1_2[1], l1_2[2])

    logic = IntersectionsLogic()

    IntAndError = logic.TwoLines(l0_1, l1_1, l0_2, l1_2)
    
    ciMatrixForm = IntAndError[0]
    ci = numpy.array([ciMatrixForm.item(0), ciMatrixForm.item(1), ciMatrixForm.item(2)])
    
    self.intersection.AddFiducial(ci[0], ci[1], ci[2])
    
    error = IntAndError[1]
    
    print('Original: ' + str(i))
    print('Computed: ' + str(ci))
    print('Error: ' + str(error))

  def onLineModified(self, caller, eventid):
    l0_1 = numpy.array([0,0,0], dtype = numpy.float32)
    l1_1 = numpy.array([0,0,0], dtype = numpy.float32)
    l0_2 = numpy.array([0,0,0], dtype = numpy.float32)
    l1_2 = numpy.array([0,0,0], dtype = numpy.float32)

    self.lineFiducials.GetNthFiducialPosition(0,l0_1)
    self.lineFiducials.GetNthFiducialPosition(1,l1_1)
    self.lineFiducials.GetNthFiducialPosition(2,l0_2)
    self.lineFiducials.GetNthFiducialPosition(3,l1_2)

    logic = IntersectionsLogic()
    i = logic.TwoLines(l0_1, l1_1, l0_2, l1_2)
    ci = numpy.array([i[0].item(0), i[0].item(1), i[0].item(2)])
    
    error = i[1]
    
    self.intersection.SetNthFiducialPosition(0, ci[0], ci[1], ci[2])

  def testNLines(self):
    knownI = [1,1,0]
    l1 = numpy.array([2,5,0])
    l2 = numpy.array([-1,5,0])
    l3 = numpy.array([-2,-30,0])

    l0_1 = knownI - 10*l1
    l1_1 = knownI - 20*l1
    l0_2 = knownI - 10*l2
    l1_2 = knownI - 20*l2
    l0_3 = knownI - l3
    l1_3 = knownI - 2*l3
    
    self.lineFiducials.AddFiducial(l0_1[0], l0_1[1], l0_1[2])
    self.lineFiducials.AddFiducial(l1_1[0], l1_1[1], l1_1[2])
    self.lineFiducials.AddFiducial(l0_2[0], l0_2[1], l0_2[2])
    self.lineFiducials.AddFiducial(l1_2[0], l1_2[1], l1_2[2])
    self.lineFiducials.AddFiducial(l0_3[0], l0_3[1], l0_3[2])
    self.lineFiducials.AddFiducial(l1_3[0], l1_3[1], l1_3[2])

    logic = IntersectionsLogic()

    iAndError = logic.NLines(3, [l0_1, l1_1, l0_2, l1_2, l0_3, l1_3])

    ci = [iAndError[0][0], iAndError[0][1], iAndError[0][2]]
    error = iAndError[1]

    self.intersection.AddFiducial(ci[0], ci[1], ci[2])

    print('Known Intersection: ' + str(knownI))
    print('Computed: ' + str(ci))
    print('Error: ' + str(error))

  def testLineAndEllipsoid(self):
    a = 1
    b = 2
    c = 3
    knownI1 = numpy.array([1,0,0])
    knownI2 = numpy.array([0,2,0])

    l = knownI2 - knownI1
    l = l / numpy.linalg.norm(l)

    l0 = knownI1 - 5*l
    l1 = knownI1 - 10*l
    
    logic = IntersectionsLogic()

    ci = logic.LineAndEllipsoid(l0, l1, a, b, c)

    if ci == False:
        print('Test Failed')
        return
            
    print ('Known Intersections:')
    print ('\t' + str(knownI1))
    print ('\t' + str(knownI2))
    print ('Computed Intersections:')

    for x in range(0, len(ci)):
        print ('\t' + str(ci[x]))



















