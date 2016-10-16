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
  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    self.logic = IntersectionsLogic()

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
  
  # Computes the standard deviation from a list of values
  # Parameters:
  #    values: the list of values used to calculate the standard deviation
  # Returns:
  #    stdDev: the standard deviation
  def findError(self,values):
    
    #Sums all values in values
    avg = 0
    for v in values:
        avg = avg + v
    
    #calculates the averages
    length = len(values)
    avg = avg / length
    
    #calculate the difference between the values in values and the average
    #square these values and sum them
    sumSquareV = 0
    for v in values:
        sumSquareV = sumSquareV + math.pow((v - avg),2)

    #find the std deviation by taking the square root and rounds to 3 decimal places
    stdDev = math.sqrt(sumSquareV / length)
    stdDev = round(stdDev,3)
    
    return stdDev
        

  # finds the intersection between a line (specified by 2 points) and
  # a plane (specified by a point and a normal vector)
  # Parameters:
  #    l0: an initial point on a line
  #    l1: a second point on the line
  #    p0: a point on the plane
  #    n: the normal vector of the plane
  # Returns:
  #    ci: the computed intersection point between the line and the plane
  def LineAndPlane(self, l0, l1, p0, n):
    
    #compute the direction vector of the line and normalizes it
    l = l1 - l0
    length = numpy.linalg.norm(l)
    if length == 0:
        print ('The line supplied has length 0')
        return
    l = l / length
    
    #checks to see if the line and the plane are parallel
    denom = numpy.dot(l, n)
    if denom == 0:
        return False
    numer = numpy.dot((p0 - l0), n)
    d = numer / denom

    ci = d*l + l0 #Computed intesection

    return ci

  # Computes the intersection point between 2 lines
  # Parameters:
  #    l0_1: the initial point on line 1
  #    l1_1: a second point on line 1
  #    l0_2: the initial point on line 2
  #    l1_2: a second point on line 2
  # Returns:
  #    M: the closest intersection point fo the 2 lines
  #    avgDistance: the average distance from of each line to the point
  def TwoLines(self, l0_1, l1_1, l0_2, l1_2):

    # Compute the direction vectors of each line and normalize them
    l_1 = l1_1 - l0_1
    length1 = numpy.linalg.norm(l_1)
    if length1 == 0:
        print ('The 1st line supplied has length 0')
        return
    l_1 = l_1 / length1
    l_2 = l1_2 - l0_2
    length2 = numpy.linalg.norm(l_2)
    if length2 == 0:
        print ('The second line supplied has length 0')
        return
    l_2 = l_2 / length2

    # Find a 3rd line that is perpendicular to the first two lines
    l_3 = numpy.cross(l_1, l_2)
    length3 = numpy.linalg.norm(l_3)
    if length3 == 0:
        return False
    l_3 = l_3 / length3

    #solve the system of linear equations to find the shortest vector between the two lines
    vMatrix = numpy.matrix([[(-1)*l_1[0], l_2[0], l_3[0]],
                            [(-1)*l_1[1], l_2[1], l_3[1]],
                            [(-1)*l_1[2], l_2[2], l_3[2]]])

    pMatrix = numpy.matrix([[l0_1[0] - l0_2[0]],
                            [l0_1[1] - l0_2[1]],
                            [l0_1[2] - l0_2[2]]])

    t = (numpy.matrix.getI(vMatrix))*pMatrix

    # find the closest point to the intersection on either line
    L1 = l0_1 + t[0]*l_1
    L2 = l0_2 + t[1]*l_2

    # find the average of those two points
    M = (L1 + L2)/2
    
    avgDistance = numpy.linalg.norm(M - L1) #distance from M to L1 and M to L2 are equal, can use either
    avgDistance = round(avgDistance,3)

    return (M,avgDistance)

  # Computes the symbolic intersection between a number of line
  # Parameters:
  #    numLines: the number of lines
  #    listOfLines: a list of lines represented by points (2 points per line)
  # Returns:
  #    symInt: the symbolic intersection of the lines
  #    error: the error metric (standard deviation)
  def NLines(self, numLines, listOfLines):
      
    if numLines <= 1:
        print ('There must be more than 1 line')
        return
    logic = IntersectionsLogic()
    
    # do a pairwise comparison of all lines, to find their intersections
    iAndDistance = []
    for i in range(0,2*numLines,2):
        l0_1 = listOfLines[i]
        l1_1 = listOfLines[i+1]
        for j in range((i+2),2*numLines,2):
            l0_2 = listOfLines[j]
            l1_2 = listOfLines[j+1]
            iAndDistance = iAndDistance + [logic.TwoLines(l0_1, l1_1, l0_2, l1_2)]
            
    # find the average of all intersection points, this is the sybolic intersection
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

    symInt = [round(xCoord,1), round(yCoord,1), round(zCoord,1)]
    error = logic.findError(allDistances)

    return (symInt, error)

  # Computes the intersection points of a line and an ellipsoid
  # Parameters:
  #    l0: the initial point on the line
  #    l1: a second point on the line
  #    a,b,c: axes of the ellipsoid
  # Returns:
  #    i1, i2: the intersection points
  def LineAndEllipsoid(self, l0, l1, a, b, c):
      
    # compute the direction vector of the line and normalize it
    l = l1 - l0
    length = numpy.linalg.norm(l)
    if length == 0:
        print ('The 1st line supplied has length 0')
        return
    l = l / length

    #find x,y,z coordinates of a point on the line
    Px = l0[0]
    Py = l0[1]
    Pz = l0[2]

    #values to plug into quadratic formula to solve for t, see github wiki for equations
    u = (math.pow(b*c*l[0],2)) + (math.pow(a*c*l[1],2)) + (math.pow(a*b*l[2], 2))
    v = (2*math.pow(b*c,2)*Px*l[0]) + (2*math.pow(a*c,2)*Py*l[1]) + (2*math.pow(a*b,2)*Pz*l[2])
    w = (math.pow(b*c*Px,2)) + (math.pow(a*c*Py,2)) + (math.pow(a*b*Pz,2)) - (math.pow(a*b*c,2))

    # quadratic formula
    try:
        t1 = (-v + math.sqrt(math.pow(v,2) - 4*u*w)) / (2*u)
        t2 = (-v - math.sqrt(math.pow(v,2) - 4*u*w)) / (2*u)
    except ValueError: #quadratic formula gave complex roots
        return False

    # coordinates of the 1st intersection point
    Lx1 = Px + (t1)*l[0]
    Ly1 = Py + (t1)*l[1]
    Lz1 = Pz + (t1)*l[2]

    i1 = [round(Lx1), round(Ly1), round(Lz1)]

    # coordinates of the second intersection point
    Lx2 = Px + (t2)*l[0]
    Ly2 = Py + (t2)*l[1]
    Lz2 = Pz + (t2)*l[2]

    i2 = [round(Lx2), round(Ly2), round(Lz2)]
    
    for x in range(0,3):
        if i1[x] != i2[x]:
            return (i1, i2)
    
    # intersection points are identical
    return i1


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
    #self.lineFiducials.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onLineModified)
  
    self.planeFiducials = slicer.vtkMRMLMarkupsFiducialNode()
    self.planeFiducials.SetName('Plane')
    slicer.mrmlScene.AddNode(self.planeFiducials)
    #self.planeFiducials.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onFiducialModified)
  
    self.intersection = slicer.vtkMRMLMarkupsFiducialNode()
    self.intersection.SetName('Int')
    slicer.mrmlScene.AddNode(self.intersection)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    print('Test intersection of Line and Plane')
    self.test1LineAndPlaneIntersection()
    self.test2LineAndPlaneIntersection()
    print('\nTest intersection of two lines:')
    self.test1TwoLines()
    self.test2TwoLines()
    print('\nTest intersection of n lines:')
    self.test1NLines()
    self.test2NLines()
    print ('\nTest intersection of Line and Ellipsoid')
    self.test1LineAndEllipsoid()
    self.test2LineAndEllipsoid()
    self.test3LineAndEllipsoid()

  # Tests the logic function LineAndPlane with a known intersection point
  def test1LineAndPlaneIntersection(self):
    i = numpy.array([2,3,0]) #Known intersection
    l = numpy.array([0,0,1])
    l0 = i - 10*l
    l1 = i - 20*l
    p0 = numpy.array([1,-20, 0])
    n = numpy.array([0,0,1])

    logic = IntersectionsLogic()

    #self.lineFiducials.AddFiducial(l0[0], l0[1], l0[2])
    #self.lineFiducials.AddFiducial(l1[0], l1[1], l1[2])
    #self.planeFiducials.AddFiducial(p0[0], p0[1], p0[2])
    #self.planeFiducials.AddFiducial(25,0,0)
    #self.planeFiducials.AddFiducial(0,25,0)

    ci = logic.LineAndPlane(l0, l1, p0, n) #Computed intersection
    
    #self.intersection.AddFiducial(ci[0],ci[1],ci[2])
   
    print('  Test 1:')
    print('\tOriginal: ' + str(i))
    print('\tComputed: ' + str(ci))

    # compares the known intersection to the computed value to see if they are the same
    for x in range(0,3):
        if i[x] != ci[x]:
            print('Test 1 Failed')
            return

    print('\tTest 1 Passed!')

  # Tests the logic function LineAndPlane with a line parallel to the plane
  def test2LineAndPlaneIntersection(self):
    print('  Test 2:')
    print('\tLine parallel to the plane')
    
    l0 = numpy.array([2,0,0])
    l1 = numpy.array([3,0,0])
    p0 = numpy.array([1,0,1])
    n = numpy.array([0,1,0])
    
    logic = IntersectionsLogic()
            
    ci = logic.LineAndPlane(l0, l1, p0, n)
    
    if ci != False:
        print ('\tTest 2 Failed')
        return
    print('\tResult: No intersection! Line Parallel to plane')
    print('\tTest 2 Passed!')
    
  # recomputes the intersection point when the line points or plane points are
  #   modified in slicer
  #def onFiducialModified(self, caller, eventid):
    
    # initialize all points
    #l0 = numpy.array([0,0,0], dtype = numpy.float32)
    #l1 = numpy.array([0,0,0], dtype = numpy.float32)
    #p0 = numpy.array([0,0,0], dtype = numpy.float32)
    #p1 = numpy.array([0,0,0], dtype = numpy.float32)
    #p2 = numpy.array([0,0,0], dtype = numpy.float32)

    # find coordinates of new line points
    #self.lineFiducials.GetNthFiducialPosition(0,l0)
    #self.lineFiducials.GetNthFiducialPosition(1,l1)

    # find the coordinates of the plane points
    #self.planeFiducials.GetNthFiducialPosition(0,p0)
    #self.planeFiducials.GetNthFiducialPosition(1,p1)
    #self.planeFiducials.GetNthFiducialPosition(2,p2)

    # recompute the normal of the plane
    #n = numpy.cross((p1-p0), (p2-p1))
    #length = numpy.linalg.norm(n)
    #if length == 0:
    #   print('length of normal is zero')
    #   return
    #n = n / length

    # calculate the intersection point
    #logic = IntersectionsLogic()
    #i = logic.LineAndPlane(l0,l1,p0,n)

    #self.intersection.SetNthFiducialPosition(0, i[0], i[1], i[2])

  # Tests the logic function TwoLines with two lines with a known intersection point
  def test1TwoLines(self):
      
    #known intersection point
    i = numpy.array([1,1,0])
    
    # compute points for two lines that intersect at the known intersection point
    l0_1 = numpy.array([10,0,0])
    dirVec1 = l0_1 - i
    dirVec1 = dirVec1 / numpy.linalg.norm(dirVec1)
    l1_1 = 10*dirVec1 + l0_1

    l0_2 = numpy.array([3,4,0])
    dirVec2 = l0_2 - i
    dirVec2 = dirVec2/ numpy.linalg.norm(dirVec2)
    l1_2 = 10*dirVec2 + l0_2

    #self.lineFiducials.AddFiducial(l0_1[0], l0_1[1], l0_1[2])
    #self.lineFiducials.AddFiducial(l1_1[0], l1_1[1], l1_1[2])
    #self.lineFiducials.AddFiducial(l0_2[0], l0_2[1], l0_2[2])
    #self.lineFiducials.AddFiducial(l1_2[0], l1_2[1], l1_2[2])

    logic = IntersectionsLogic()

    IntAndError = logic.TwoLines(l0_1, l1_1, l0_2, l1_2)
    
    ciMatrixForm = IntAndError[0]
    ci = numpy.array([ciMatrixForm.item(0), ciMatrixForm.item(1), ciMatrixForm.item(2)])
    
    #self.intersection.AddFiducial(ci[0], ci[1], ci[2])
    
    error = IntAndError[1]
    
    print('  Test 1:')
    print('\tOriginal: ' + str(i))
    print('\tComputed: ' + str(ci))
    for x in range(0,3):
        if i[x] != round(ci[x]):
            print('Test 1 Failed')
            return
    print('\tError: ' + str(error))
    print('\tTest 1 Passed')
  
  # Tests the logic function TwoLines with two lines that are parallel
  def test2TwoLines(self):
    
    # compute points for two lines that intersect at the known intersection point
    dirVec1 = numpy.array([1,0,0])
    l0_1 = numpy.array([2,0,0])
    l1_1 = numpy.array([3,0,0])
                          
    dirVec2 = numpy.array([1,0,0])
    l0_2 = numpy.array([1,1,0])
    l1_2 = numpy.array([2,1,0])
                                          
    #self.lineFiducials.AddFiducial(l0_1[0], l0_1[1], l0_1[2])
    #self.lineFiducials.AddFiducial(l1_1[0], l1_1[1], l1_1[2])
    #self.lineFiducials.AddFiducial(l0_2[0], l0_2[1], l0_2[2])
    #self.lineFiducials.AddFiducial(l1_2[0], l1_2[1], l1_2[2])
                                          
    logic = IntersectionsLogic()
                                              
    IntAndError = logic.TwoLines(l0_1, l1_1, l0_2, l1_2)
                                                              
    print('  Test 2:')
    print('\tTwo parallel lines')
    if IntAndError != False:
        print('\tTest 2 Failed')
        return
    print('\tResult: Lines are parallel no intersection point')
    print('\tTest 2 Passed')


  #def onLineModified(self, caller, eventid):
    #l0_1 = numpy.array([0,0,0], dtype = numpy.float32)
    #l1_1 = numpy.array([0,0,0], dtype = numpy.float32)
    #l0_2 = numpy.array([0,0,0], dtype = numpy.float32)
    #l1_2 = numpy.array([0,0,0], dtype = numpy.float32)

    #self.lineFiducials.GetNthFiducialPosition(0,l0_1)
    #self.lineFiducials.GetNthFiducialPosition(1,l1_1)
    #self.lineFiducials.GetNthFiducialPosition(2,l0_2)
    #self.lineFiducials.GetNthFiducialPosition(3,l1_2)

    #logic = IntersectionsLogic()
    #i = logic.TwoLines(l0_1, l1_1, l0_2, l1_2)
    #ci = numpy.array([i[0].item(0), i[0].item(1), i[0].item(2)])
    
    #error = i[1]
    
    #self.intersection.SetNthFiducialPosition(0, ci[0], ci[1], ci[2])

  # Tests logic function NLines with 3 lines with a known intersection point
  def test1NLines(self):
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
    
    #self.lineFiducials.AddFiducial(l0_1[0], l0_1[1], l0_1[2])
    #self.lineFiducials.AddFiducial(l1_1[0], l1_1[1], l1_1[2])
    #self.lineFiducials.AddFiducial(l0_2[0], l0_2[1], l0_2[2])
    #self.lineFiducials.AddFiducial(l1_2[0], l1_2[1], l1_2[2])
    #self.lineFiducials.AddFiducial(l0_3[0], l0_3[1], l0_3[2])
    #self.lineFiducials.AddFiducial(l1_3[0], l1_3[1], l1_3[2])

    logic = IntersectionsLogic()

    iAndError = logic.NLines(3, [l0_1, l1_1, l0_2, l1_2, l0_3, l1_3])

    ci = [iAndError[0][0], iAndError[0][1], iAndError[0][2]]
    error = iAndError[1]

    #self.intersection.AddFiducial(ci[0], ci[1], ci[2])

    print('  Test 1:')
    print('\tKnown Intersection: ' + str(knownI))
    for x in range(0,3):
        if knownI[x] != round(ci[x]):
            print('Test 1 Failed')
            return
    print('\tComputed: ' + str(ci))
    print('\tError: ' + str(error))
    print('\tTest 1 Passed!')
  
  # Tests logic function NLines with 3 lines that form a triangle with the center of gravity at [1,1,0]
  def test2NLines(self):
    symI = [1,1,0]
    
    #points form a triangle with the center of gravity at symI
    l0_1 = numpy.array([0,0,0])
    l1_1 = numpy.array([2,0,0])
    l0_2 = l1_1
    l1_2 = numpy.array([1,3,0])
    l0_3 = l1_2
    l1_3 = l0_1
                                          
    #self.lineFiducials.AddFiducial(l0_1[0], l0_1[1], l0_1[2])
    #self.lineFiducials.AddFiducial(l1_1[0], l1_1[1], l1_1[2])
    #self.lineFiducials.AddFiducial(l0_2[0], l0_2[1], l0_2[2])
    #self.lineFiducials.AddFiducial(l1_2[0], l1_2[1], l1_2[2])
    #self.lineFiducials.AddFiducial(l0_3[0], l0_3[1], l0_3[2])
    #self.lineFiducials.AddFiducial(l1_3[0], l1_3[1], l1_3[2])
                                                                  
    logic = IntersectionsLogic()
                                                                      
    iAndError = logic.NLines(3, [l0_1, l1_1, l0_2, l1_2, l0_3, l1_3])
                                                                          
    ci = [iAndError[0][0], iAndError[0][1], iAndError[0][2]]
    error = iAndError[1]
                                                                                  
    #self.intersection.AddFiducial(ci[0], ci[1], ci[2])
    
    print('  Test 2:')
    print('\tKnown Intersection: ' + str(symI))
    for x in range(0,3):
        if symI[x] != round(ci[x]):
            print('Test 1 Failed')
            return
    print('\tComputed: ' + str(ci))
    print('\tError: ' + str(error))
    print('\tTest 2 Passed!')


  # Tests logic function LineAndEllipsoid with a line that has 2 intersection points
  def test1LineAndEllipsoid(self):
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

    # check that there are 2 intersection points
    if ci == False or len(ci)!= 2:
        print('Test 1 Failed')
        return
    
    print ('  Test 1:')
    print ('\tKnown Intersections:')
    print ('\t\t' + str(knownI1))
    print ('\t\t' + str(knownI2))
    
    #check that the computed intersections match the known intersections
    for i in range(0,3):
        if knownI1[i] != round(ci[0][i]):
            print('\tTest 1 Failed')
            return
        if knownI2[i] != round(ci[1][i]):
            print('\tTest 1 Failed')
            return
    print ('\tComputed Intersections:')
    for x in range(0, 2):
        print ('\t\t' + str(ci[x]))
    print ('\tTest 1 Passed!')

  # Tests logic function LineAndEllipsoid with a line that has 1 intersection point
  def test2LineAndEllipsoid(self):
    a = 1
    b = 2
    c = 3
    knownI = numpy.array([0,2,0])
    l0 = numpy.array([2,2,0])
    l1 = numpy.array([3,2,0])
    
    logic = IntersectionsLogic()
    
    ci = logic.LineAndEllipsoid(l0, l1, a, b, c)
    
    # check that there is only 1 intersection point
    if ci == False or len(ci) != 3:
        print('\tTest 2 Failed')
        return

    print ('  Test 2:')
    print ('\tKnown Intersections:')
    print ('\t\t' + str(knownI))
    #check that the computed intersection match the known intersection
    for x in range(0,3):
        if knownI[x] != round(ci[x]):
            print ('\tTest 2 Failed')
    print ('\tComputed Intersections:')
    print ('\t\t' + str(ci))
    print ('\tTest 2 Passed!')

  # Tests logic function LineAndEllipsoid with a line that has no intersection points
  def test3LineAndEllipsoid(self):
    a = 1
    b = 2
    c = 3
    l0 = numpy.array([1,3,0])
    l1 = numpy.array([2,3,0])
    
    logic = IntersectionsLogic()
    
    ci = logic.LineAndEllipsoid(l0, l1, a, b, c)
    
    print ('  Test 3')
    print ('\tThere should be no intersection point')
    
    #check that there is no intersection point
    if ci != False:
        print ('Test 3 Failed')
        return
    print ('\tResult: No intersection!')
    print ('\tTest 3 Passed!')



















