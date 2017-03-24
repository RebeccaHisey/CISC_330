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

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "SimulateAndReconstruct" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["Rebecca Hisey (Queen's University)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
    """
    self.parent.acknowledgementText = """
    """

#
# SimulateAndReconstructWidget
#

class SimulateAndReconstructWidget(ScriptedLoadableModuleWidget):

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

    #self.outputModel = slicer.vtkMRMLModelNode()
    #self.outputModel.SetName('tumour')
    #slicer.mrmlScene.AddNode(self.outputModel)

    #self.outputDisplayModel = slicer.vtkMRMLModelDisplayNode()
    #self.outputDisplayModel.SetName('tumourDisplay')
    #slicer.mrmlScene.AddNode(self.outputDisplayModel)

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
    # Add Button
    #
    self.addButton = qt.QPushButton("Add")
    self.addButton.toolTip = "Add a contour to the model."
    self.addButton.enabled = False
    parametersFormLayout.addRow(self.addButton)

    #
    # Add Button
    #
    self.removeButton = qt.QPushButton("Remove")
    self.removeButton.toolTip = "Remove a contour from the model."
    self.removeButton.enabled = False
    parametersFormLayout.addRow(self.removeButton)

    # connections
    self.addButton.connect('clicked(bool)', self.onAddButton)
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
    self.addButton.enabled = self.ImagetoDetectorTransform.currentNode() and self.DetectortoRASTransform.currentNode() and self.tumourContour.currentNode()
    self.removeButton.enabled = self.addButton.enabled
  
  def onAddButton(self):
    self.logic.AddContourToModel(self.ImagetoDetectorTransform.currentNode(), self.DetectortoRASTransform.currentNode(),self.tumourContour.currentNode(), self.ModelFiducials)

  def onRemoveButton(self):
    self.logic.RemoveContourFromModel(self.ImagetoDetectorTransform.currentNode(), self.DetectortoRASTransform.currentNode(),self.tumourContour.currentNode(),self.ModelFiducials)

#
# SimulateAndReconstructLogic
#

class SimulateAndReconstructLogic(ScriptedLoadableModuleLogic):

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
          numPoints = len(images)
          imagePoints = self.uvwToXYZ(angles[i],images[i]) # project points into (x,y,z) coordinates
          for j in range(0,numPoints):

              imagePointSourceLine = source - imagePoints[j]
              imagePointSourceDistance = numpy.linalg.norm(imagePointSourceLine)
              pointSourceLine = imagePointSourceLine/imagePointSourceDistance
              projectedPointSourceDistance = (750*1500)/imagePointSourceDistance
              imageProjectedPointDistance = imagePointSourceDistance - projectedPointSourceDistance
              projectedPoint = imagePoints[j] + imageProjectedPointDistance*pointSourceLine
              reconstructionPoints.append(projectedPoint)

      surfaceArea,Volume = self.createSurface(len(reconstructionPoints),reconstructionPoints, "tumour", "tumourDisplay")

      return (surfaceArea,Volume)

  # Creates a surface model on a set of points
  # Parameters:
  #    numPoints: the number of points
  #    dataPoints: the points that will be used to create the surface model
  # Returns:
  #    the surface area and volume of the model
  def createSurface(self,numPoints, datapoints, outputModelName, outputDisplayModelName):

      outputModel = slicer.mrmlScene.GetFirstNodeByName(outputModelName)
      outputDisplayModel = slicer.mrmlScene.GetFirstNodeByName(outputDisplayModelName)

      if outputModel == None:
          outputModel = slicer.vtkMRMLModelNode()
          outputModel.SetName(outputModelName)
          slicer.mrmlScene.AddNode(outputModel)

      if outputDisplayModel == None:
          outputDisplayModel = slicer.vtkMRMLModelDisplayNode()
          outputDisplayModel.SetName(outputDisplayModelName)
          slicer.mrmlScene.AddNode(self.outputDisplayModel)

      points = vtk.vtkPoints()
      cellArray = vtk.vtkCellArray()

      points.SetNumberOfPoints(numPoints)

      for i in range(numPoints):
          if str(type(datapoints)) == "<type 'list'>":
              points.SetPoint(i, datapoints[i])
          else:
              position = [0, 0, 0, 0]
              datapoints.GetNthFiducialWorldCoordinates(i, position)
              points.SetPoint(i, [position[0],position[1],position[2]])

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

      #outputModel.SetName("tumour")
      outputModel.SetPolyDataConnection(normals.GetOutputPort())
      outputDisplayModel.SetColor(0, 1, 0)
      #slicer.mrmlScene.AddNode(outputDisplayModel)
      outputModel.SetAndObserveDisplayNodeID(outputDisplayModel.GetID())
      outputModel.Modified()
      #slicer.mrmlScene.AddNode(outputModel)
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
      angles = [(0, -180), (0, -135), (0, -90), (0, -45), (0, 0), (0, 45), (0, 90), (0, 135),
                (45, -180), (45, -135), (45, -90), (45, -45), (45, 0), (45, 45), (45, 90), (45, 135),
                (-45, -180), (-45, -135), (-45, -90), (-45, -45), (-45, 0), (-45, 45), (-45, 90), (-45, 135)]

      self.SimulateTransformsForNImages(angles)

  def SimulateTransformsForNImages(self,angles):

    numAngles = len(angles)
    for i in range(0, numAngles):
            self.ImageToDetector = slicer.vtkMRMLLinearTransformNode()
            self.ImageToDetector.SetName('ImageToDetector'+str(i))
            slicer.mrmlScene.AddNode(self.ImageToDetector)

            self.ImageToDetectorTransform = vtk.vtkTransform()
            self.ImageToDetectorTransform.RotateY(angles[i][0])
            self.ImageToDetectorTransform.RotateX(angles[i][1])

            self.ImageToDetector.SetAndObserveTransformToParent(self.ImageToDetectorTransform)

            self.DetectorToRAS = slicer.vtkMRMLLinearTransformNode()
            self.DetectorToRAS.SetName('DetectorToRAS' + str(i))
            slicer.mrmlScene.AddNode(self.DetectorToRAS)

            self.DetectorToRASTransform = vtk.vtkTransform()
            self.DetectorToRASTransform.Scale(2,2,2)

            self.DetectorToRAS.SetAndObserveTransformToParent(self.DetectorToRASTransform)

  def CreateSlab(self, ImagetoDetector, DetectortoRAS, imageNo):

      tumourContour = slicer.mrmlScene.GetFirstNodeByName("contour")

      slabFiducials = slicer.mrmlScene.GetFirstNodeByName('Slab'+str(imageNo))

      if slabFiducials == None:
          slabFiducials = slicer.vtkMRMLMarkupsFiducialNode()
          slabFiducials.SetName('Slab'+str(imageNo))
          slicer.mrmlScene.AddNode(slabFiducials)
      else:
          slabFiducials.RemoveAllMarkups()

      #slabRadius = self.FindLargestRadius(tumourContour)
      slabRadius = 25
      shrinkFactor = math.pow((750 - 2*slabRadius)/ 1500.0,2)
      enlargeFactor = math.pow((750 + 2*slabRadius)/1500.0,2)
      slabSourceEnd = slicer.mrmlScene.GetFirstNodeByName('slabSourceEnd'+str(imageNo))

      if slabSourceEnd == None:
          slabSourceEnd = slicer.vtkMRMLLinearTransformNode()
          slabSourceEnd.SetName('SlabSourceEnd'+str(imageNo))
          slicer.mrmlScene.AddNode(slabSourceEnd)
      slabSourceEnd.SetAndObserveTransformToParent(None)
      slabSourceEndTransform = vtk.vtkTransform()
      slabSourceEndTransform.Translate(0,0,2*slabRadius)
      slabSourceEndTransform.Scale(shrinkFactor,shrinkFactor,shrinkFactor)
      slabSourceEnd.SetAndObserveTransformToParent(slabSourceEndTransform)

      tumourContour.SetAndObserveTransformNodeID(slabSourceEnd.GetID())
      numContourPoints = tumourContour.GetNumberOfFiducials()
      pos = [0, 0, 0, 0]

      for i in range(0, numContourPoints):
          tumourContour.GetNthFiducialWorldCoordinates(i, pos)
          slabFiducials.AddFiducial(pos[0], pos[1], pos[2])

      tumourContour.SetAndObserveTransformNodeID(None)

      slabDetectorEnd = slicer.mrmlScene.GetFirstNodeByName('slabDetectorEnd' + str(imageNo))

      if slabDetectorEnd == None:
          slabDetectorEnd = slicer.vtkMRMLLinearTransformNode()
          slabDetectorEnd.SetName('SlabDetectorEnd'+str(imageNo))
          slicer.mrmlScene.AddNode(slabDetectorEnd)
      slabDetectorEnd.SetAndObserveTransformToParent(None)
      slabDetectorEndTransform = vtk.vtkTransform()
      slabDetectorEndTransform.Translate(0,0,-2*slabRadius)
      slabDetectorEndTransform.Scale(enlargeFactor, enlargeFactor, enlargeFactor)
      slabDetectorEnd.SetAndObserveTransformToParent(slabDetectorEndTransform)

      tumourContour.SetAndObserveTransformNodeID(slabDetectorEnd.GetID())

      for i in range(0, numContourPoints):
          tumourContour.GetNthFiducialWorldCoordinates(i, pos)
          slabFiducials.AddFiducial(pos[0], pos[1], pos[2])

      tumourContour.SetAndObserveTransformNodeID(None)
      slabFiducials.SetAndObserveTransformNodeID(ImagetoDetector.GetID())
      ImagetoDetector.SetAndObserveTransformNodeID(DetectortoRAS.GetID())

      self.outputModel = slicer.mrmlScene.GetFirstNodeByName('tumour'+str(imageNo))
      self.outputDisplayModel = slicer.mrmlScene.GetFirstNodeByName('tumourDisplay'+str(imageNo))
      if self.outputModel == None:
          self.outputModel = slicer.vtkMRMLModelNode()
          self.outputModel.SetName('tumour'+str(imageNo))
          slicer.mrmlScene.AddNode(self.outputModel)

          self.outputDisplayModel = slicer.vtkMRMLModelDisplayNode()
          self.outputDisplayModel.SetName('tumourDisplay'+str(imageNo))
          slicer.mrmlScene.AddNode(self.outputDisplayModel)

      self.createSurface(slabFiducials.GetNumberOfFiducials(), slabFiducials, self.outputModel.GetName(), self.outputDisplayModel.GetName())

      for j in range(0,slabFiducials.GetNumberOfFiducials()):
          slabFiducials.SetNthFiducialVisibility(j,False)


  def IntersectionPolyDataFilter(self,model1,model2):
      intersectionPolyDataFilter = vtk.vtkIntersectionPolyDataFilter()

      intersectionPolyDataFilter.SetInputConnection(0, model1.GetPolyDataConnection())
      intersectionPolyDataFilter.SetInputConnection(1, model2.GetPolyDataConnection())
      intersectionPolyDataFilter.Update()

      delaunay = vtk.vtkDelaunay3D()
      delaunay.SetInputConnection(intersectionPolyDataFilter.GetOutputPort())

      surfaceFilter = vtk.vtkDataSetSurfaceFilter()
      surfaceFilter.SetInputConnection(delaunay.GetOutputPort())

      normals = vtk.vtkPolyDataNormals()
      normals.SetInputConnection(surfaceFilter.GetOutputPort())
      normals.SetFeatureAngle(100.0)

      outputModel = slicer.vtkMRMLModelNode()
      outputModel.SetName("Intersection")
      outputModel.SetPolyDataConnection(normals.GetOutputPort())
      outputDisplayModel = slicer.vtkMRMLModelDisplayNode()
      outputDisplayModel.SetColor(1, 0, 0)
      slicer.mrmlScene.AddNode(outputDisplayModel)
      outputModel.SetAndObserveDisplayNodeID(outputDisplayModel.GetID())
      outputModel.Modified()
      slicer.mrmlScene.AddNode(outputModel)

  def BooleanOperation(self,model1,model2):

      booleanOperation =vtk.vtkBooleanOperationPolyDataFilter()

      booleanOperation.SetOperationToIntersection()
      booleanOperation.SetInputData(0, model1.GetPolyData())
      booleanOperation.SetInputData(1, model2.GetPolyData())

      booleanOperationMapper = vtk.vtkPolyDataMapper()
      booleanOperationMapper.SetInputConnection(booleanOperation.GetOutputPort())
      booleanOperationMapper.ScalarVisibilityOff()

      delaunay = vtk.vtkDelaunay3D()
      delaunay.SetInputConnection(booleanOperation.GetOutputPort())

      surfaceFilter = vtk.vtkDataSetSurfaceFilter()
      surfaceFilter.SetInputConnection(delaunay.GetOutputPort())

      normals = vtk.vtkPolyDataNormals()
      normals.SetInputConnection(surfaceFilter.GetOutputPort())
      normals.SetFeatureAngle(100.0)

      outputModel = slicer.mrmlScene.GetFirstNodeByName("Boolean")
      outputDisplayModel = slicer.mrmlScene.GetFirstNodeByName("BooleanDisplay")
      if outputModel == None:
          outputModel = slicer.vtkMRMLModelNode()
          outputModel.SetName("Boolean")
      if outputDisplayModel == None:
          outputDisplayModel = slicer.vtkMRMLModelDisplayNode()
          outputDisplayModel.SetName("BooleanDisplay")
          outputDisplayModel.SetColor(0, 0, 1)
          slicer.mrmlScene.AddNode(outputDisplayModel)
          outputModel.SetAndObserveDisplayNodeID(outputDisplayModel.GetID())
          slicer.mrmlScene.AddNode(outputModel)
      outputModel.SetPolyDataConnection(normals.GetOutputPort())
      #outputDisplayModel = slicer.vtkMRMLModelDisplayNode()
      #outputDisplayModel.SetColor(0, 0, 1)
      #slicer.mrmlScene.AddNode(outputDisplayModel)

      outputModel.Modified()


  def ReconstructTumourFromSlabs(self, angles, contour):
      self.SimulateTransformsForNImages(angles)

      ImagetoDetector = slicer.mrmlScene.GetFirstNodeByName("ImageToDetector" + str(0))
      DetectortoRAS = slicer.mrmlScene.GetFirstNodeByName("DetectorToRAS" + str(0))
      self.CreateSlab(ImagetoDetector, DetectortoRAS,0)

      if len(angles) == 1:
          slab = slicer.mrmlScene.GetFirstNodeByName("tumour0")
          slab.GetModelDisplayNode().SetColor(0,0,1)
      else:
          ImagetoDetector = slicer.mrmlScene.GetFirstNodeByName("ImageToDetector" + str(1))
          DetectortoRAS = slicer.mrmlScene.GetFirstNodeByName("DetectorToRAS" + str(1))
          self.CreateSlab(ImagetoDetector, DetectortoRAS,1)

          firstSlab = slicer.mrmlScene.GetFirstNodeByName("tumour0")
          secondSlab = slicer.mrmlScene.GetFirstNodeByName("tumour1")

          firstSlab.GetModelDisplayNode().SetOpacity(0)
          secondSlab.GetModelDisplayNode().SetOpacity(0)

          self.BooleanOperation(firstSlab,secondSlab)

      for i in range(2,len(angles)):
          ImagetoDetector = slicer.mrmlScene.GetFirstNodeByName("ImageToDetector" + str(i))
          DetectortoRAS = slicer.mrmlScene.GetFirstNodeByName("DetectorToRAS" + str(i))

          self.CreateSlab(ImagetoDetector,DetectortoRAS, i)
          CurrentSlab = slicer.mrmlScene.GetFirstNodeByName("tumour"+str(i))
          CurrentSlab.GetModelDisplayNode().SetOpacity(0)

          IntersectionModel = slicer.mrmlScene.GetFirstNodeByName("Boolean")

          self.BooleanOperation(CurrentSlab,IntersectionModel)
      intersectionModel = slicer.mrmlScene.GetFirstNodeByName("Boolean")
      properties = vtk.vtkMassProperties()
      properties.SetInputData(intersectionModel.GetPolyData())
      volume = properties.GetVolume()
      surfaceArea = properties.GetSurfaceArea()
      return(surfaceArea,volume)


  def AddContourToModel(self, ImagetoDetector, DetectortoRAS, tumourContour, modelFiducials):

    logging.info('Processing started')

    tumourContour.SetAndObserveTransformNodeID(ImagetoDetector.GetID())
    ImagetoDetector.SetAndObserveTransformNodeID(DetectortoRAS.GetID())
    
    numContourPoints = tumourContour.GetNumberOfFiducials()
    pos = [0,0,0,0]

    for i in range (0, numContourPoints):
        tumourContour.GetNthFiducialWorldCoordinates(i,pos)
        modelFiducials.AddFiducial(pos[0],pos[1],pos[2])

    self.createSurface(modelFiducials.GetNumberOfFiducials(),modelFiducials, "tumour", "tumourDisplay")
    for j in range (0, modelFiducials.GetNumberOfFiducials()):
        modelFiducials.SetNthFiducialVisibility(j,False)

    tumourContour.SetAndObserveTransformNodeID(None)
    ImagetoDetector.SetAndObserveTransformNodeID(None)

    logging.info('Processing completed')

    return True

  def RemoveContourFromModel(self, ImagetoDetector, DetectortoRAS, tumourContour, modelFiducials):

    logging.info('Processing started')

    numContourPoints = tumourContour.GetNumberOfFiducials()

    tumourContour.SetAndObserveTransformNodeID(ImagetoDetector.GetID())
    ImagetoDetector.SetAndObserveTransformNodeID(DetectortoRAS.GetID())

    toBeRemoved = []
    pos = [0, 0, 0, 0]
    for i in range(0, numContourPoints):
        tumourContour.GetNthFiducialWorldCoordinates(i, pos)
        position = [pos[0],pos[1],pos[2]]
        toBeRemoved.append(position)

    tumourContour.SetAndObserveTransformNodeID(None)
    ImagetoDetector.SetAndObserveTransformNodeID(None)

    self.tempFiducials = slicer.vtkMRMLMarkupsFiducialNode()
    for j in range(0,modelFiducials.GetNumberOfFiducials()):
        pos = [0, 0, 0]
        modelFiducials.GetNthFiducialPosition(j,pos)
        if pos not in toBeRemoved:
            self.tempFiducials.AddFiducial(pos[0],pos[1],pos[2])

    modelFiducials.Copy(self.tempFiducials)

    self.createSurface(modelFiducials.GetNumberOfFiducials(), modelFiducials, "tumour", "tumourDisplay")

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
    #self.TestSimulatorWithTwoImages()

  def TestSimulatorWithOneImage(self):
      logic = SimulateAndReconstructLogic()
      logic.SimulatorWithNImages(1)

  def TestSimulatorWithTwoImages(self):
      logic = SimulateAndReconstructLogic()
      logic.SimulatorWithNImages(2)



