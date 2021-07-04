

import argparse
import SimpleITK as sitk
from vtk.util.vtkImageImportFromArray import *
import vtk
import numpy as np


def vtkShow(numpyImage, spacing=(1.0, 1.0, 1.0)):
    # 键盘控制交互式操作
    class KeyPressInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

        def __init__(self, parent=None, *args, **kwargs):
            super(KeyPressInteractorStyle).__init__(*args, **kwargs)
            self.parent = vtk.vtkRenderWindowInteractor()
            if parent is not None:
                self.parent = parent

            self.AddObserver("KeyPressEvent", self.keyPress)

        def keyPress(self, obj, event):
            key = self.parent.GetKeySym()
            if key == 'Up':
                # gradtfun.AddPoint(-100, 1.0)
                # gradtfun.AddPoint(10, 1.0)
                # gradtfun.AddPoint(20, 1.0)
                #
                # volumeProperty.SetGradientOpacity(gradtfun)
                renWin.Render()
            if key == 'Down':
                # print('Down')
                # tfun.RemoveAllPoints()
                # tfun.AddPoint(1129, 0)
                renWin.Render()

    # def StartInteraction():
    #     renWin.SetDesiredUpdateRate(10)
    #
    # def EndInteraction():
    #     renWin.SetDesiredUpdateRate(0.001)
    #
    # def ClipVolumeRender(obj):
    #     obj.GetPlanes(planes)
    #     volumeMapper.SetClippingPlanes(planes)
    spacing = tuple(reversed(spacing))
    numpyImage = numpyImage.astype(np.float32) - np.min(numpyImage)

    minValue, maxValue = np.min(numpyImage), np.max(numpyImage)
    if maxValue - minValue < 100:
        numpyImage = 1000 * numpyImage
        minValue, maxValue = np.min(numpyImage), np.max(numpyImage)

    print('shape of data', numpyImage.shape)
    print('minValue, maxValue', minValue, maxValue)

    img_arr = vtkImageImportFromArray()
    img_arr.SetArray(numpyImage)
    img_arr.SetDataSpacing(spacing)
    img_arr.SetDataOrigin((0, 0, 0))
    img_arr.Update()

    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    iren = vtk.vtkRenderWindowInteractor()
    renWin.AddRenderer(ren)
    iren.SetRenderWindow(renWin)  # 把上面那个窗口加入交互操作
    iren.SetInteractorStyle(KeyPressInteractorStyle(parent=iren))  # 在交互操作里面添加这个自定义的操作例如up,down

    # diffusion = vtk.vtkImageAnisotropicDiffusion3D()
    # diffusion.SetInputData(img_arr.GetOutput())
    # diffusion.SetNumberOfIterations(10)
    # diffusion.SetDiffusionThreshold(5)
    # diffusion.Update()

    # shifter = vtk.vtkImageShiftScale()  # 对偏移和比例参数来对图像数据进行操作 数据转换，之后直接调用shifter
    # shifter.SetShift(0.0)
    # shifter.SetScale(1.0)
    # shifter.SetOutputScalarTypeToUnsignedShort()
    # shifter.SetInputData(img_arr.GetOutput())
    # shifter.ReleaseDataFlagOff()
    # shifter.Update()

    inputData = img_arr

    tfun = vtk.vtkPiecewiseFunction()  # 不透明度传输函数---放在tfun
    tfun.AddPoint(minValue, 0.0)
    tfun.AddPoint(maxValue, 1.0)

    gradtfun = vtk.vtkPiecewiseFunction()  # 梯度不透明度函数---放在gradtfun
    gradtfun.AddPoint(0, 0)
    gradtfun.AddPoint(0.7, 0.1)
    gradtfun.AddPoint(1.0, 1.0)

    ctfun = vtk.vtkColorTransferFunction()  # 颜色传输函数---放在ctfun
    ctfun.AddRGBPoint(minValue, 0.5, 0.0, 0.0)
    ctfun.AddRGBPoint(maxValue, 0.9, 0.2, 0.3)

    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()  # 映射器volumnMapper使用vtk的管线投影算法
    volumeMapper.SetInputData(inputData.GetOutput())  # 向映射器中输入数据：shifter(预处理之后的数据)
    volumeProperty = vtk.vtkVolumeProperty()  # 创建vtk属性存放器,向属性存放器中存放颜色和透明度
    volumeProperty.SetColor(ctfun)
    volumeProperty.SetScalarOpacity(tfun)
    volumeProperty.SetGradientOpacity(gradtfun)
    volumeProperty.SetInterpolationTypeToLinear()  # ???
    volumeProperty.ShadeOn()

    newvol = vtk.vtkVolume()  # 演员
    newvol.SetMapper(volumeMapper)
    newvol.SetProperty(volumeProperty)

    outline = vtk.vtkOutlineFilter()
    outline.SetInputConnection(inputData.GetOutputPort())
    outlineMapper = vtk.vtkPolyDataMapper()
    outlineMapper.SetInputConnection(outline.GetOutputPort())
    outlineActor = vtk.vtkActor()
    outlineActor.SetMapper(outlineMapper)
    ren.AddActor(outlineActor)

    ren.AddVolume(newvol)
    ren.SetBackground(0.8, 0.8, 0.8)
    renWin.SetSize(600, 600)

    planes = vtk.vtkPlanes()

    # boxWidget = vtk.vtkBoxWidget()
    # boxWidget.SetInteractor(iren)
    # boxWidget.SetPlaceFactor(1.0)
    # boxWidget.PlaceWidget(0, 0, 0, 0, 0, 0)
    # boxWidget.InsideOutOn()
    # boxWidget.AddObserver("StartInteractionEvent", StartInteraction)
    # boxWidget.AddObserver("InteractionEvent", ClipVolumeRender)
    # boxWidget.AddObserver("EndInteractionEvent", EndInteraction)
    #
    # outlineProperty = boxWidget.GetOutlineProperty()
    # outlineProperty.SetRepresentationToWireframe()
    # outlineProperty.SetAmbient(1.0)
    # outlineProperty.SetAmbientColor(1, 1, 1)
    # outlineProperty.SetLineWidth(9)
    #
    # selectedOutlineProperty = boxWidget.GetSelectedOutlineProperty()
    # selectedOutlineProperty.SetRepresentationToWireframe()
    # selectedOutlineProperty.SetAmbient(1.0)
    # selectedOutlineProperty.SetAmbientColor(1, 0, 0)
    # selectedOutlineProperty.SetLineWidth(3)

    # cameraR = vtk.vtkCamera()
    # cameraR.SetPosition(200, 200, 200)
    # cameraR.SetFocalPoint(0, 0, 0)
    # ren.SetActiveCamera(cameraR)
    ren.ResetCamera()
    iren.Initialize()
    renWin.Render()

    sliderRep_min = vtk.vtkSliderRepresentation2D()
    sliderRep_min.SetMinimumValue(minValue)
    sliderRep_min.SetMaximumValue(maxValue)
    sliderRep_min.SetValue(minValue + 1)
    sliderRep_min.SetTitleText("minValue")
    sliderRep_min.GetPoint1Coordinate().SetCoordinateSystemToNormalizedViewport()
    sliderRep_min.GetPoint1Coordinate().SetValue(0.15, 0.1)
    sliderRep_min.GetPoint2Coordinate().SetCoordinateSystemToNormalizedViewport()
    sliderRep_min.GetPoint2Coordinate().SetValue(0.45, 0.1)
    sliderRep_min.SetSliderLength(0.05)
    sliderRep_min.SetSliderWidth(0.05)
    sliderRep_min.SetEndCapLength(0.025)

    sliderWidget_min = vtk.vtkSliderWidget()
    sliderWidget_min.SetInteractor(iren)
    sliderWidget_min.SetRepresentation(sliderRep_min)
    sliderWidget_min.SetAnimationModeToAnimate()
    sliderWidget_min.EnabledOn()

    sliderRep_max = vtk.vtkSliderRepresentation2D()
    sliderRep_max.SetMinimumValue(minValue)
    sliderRep_max.SetMaximumValue(maxValue)
    sliderRep_max.SetValue(maxValue - 1)
    sliderRep_max.SetTitleText("maxValue")
    sliderRep_max.GetPoint1Coordinate().SetCoordinateSystemToNormalizedViewport()
    sliderRep_max.GetPoint1Coordinate().SetValue(0.65, 0.1)
    sliderRep_max.GetPoint2Coordinate().SetCoordinateSystemToNormalizedViewport()
    sliderRep_max.GetPoint2Coordinate().SetValue(0.95, 0.1)
    sliderRep_max.SetSliderLength(0.05)
    sliderRep_max.SetSliderWidth(0.05)
    sliderRep_max.SetEndCapLength(0.025)

    sliderWidget_max = vtk.vtkSliderWidget()
    sliderWidget_max.SetInteractor(iren)
    sliderWidget_max.SetRepresentation(sliderRep_max)
    sliderWidget_max.SetAnimationModeToAnimate()
    sliderWidget_max.EnabledOn()

    def update_minmax(obj, ev):
        minValue = sliderWidget_min.GetRepresentation().GetValue()
        maxValue = sliderWidget_max.GetRepresentation().GetValue()
        tfun.RemoveAllPoints()
        tfun.AddPoint(minValue, 0.0)
        tfun.AddPoint(maxValue, 1.0)
        volumeProperty.SetScalarOpacity(tfun)
        renWin.Render()

        # print(ren.GetActiveCamera().GetPosition())

    sliderWidget_min.AddObserver(vtk.vtkCommand.InteractionEvent, update_minmax)
    sliderWidget_max.AddObserver(vtk.vtkCommand.InteractionEvent, update_minmax)

    iren.Initialize()
    renWin.Render()

    iren.Start()


if __name__ == "__main__":
    # print("-----")
    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    args = parser.parse_args()

    # print(args)
    vtkShow(sitk.GetArrayFromImage(sitk.ReadImage(args.file)))