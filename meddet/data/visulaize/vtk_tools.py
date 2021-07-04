from vtk.util.vtkImageImportFromArray import *
import vtk
import numpy as np


def getRenderOfSrcImageWithClip(render, renWinInteractor, numpyImage_src, spacing,
                                minValue=0, maxValue=10, pos=(0, 0, 1.0, 1.0)):
    numpyImage_src = numpyImage_src.astype(np.float32) - np.min(numpyImage_src)
    numpyImage_src = maxValue * numpyImage_src / np.max(numpyImage_src)
    print('minValue, maxValue', minValue, maxValue)
    count = 1
    # render = vtk.vtkRenderer()
    # render.SetBackground(0.8, 0.8, 0.8)
    # render.SetActiveCamera(camera)
    # render.SetViewport(*pos)

    img_arr = vtkImageImportFromArray()
    img_arr.SetArray(numpyImage_src)
    img_arr.SetDataSpacing(spacing)
    img_arr.SetDataOrigin((0, 0, 0))
    img_arr.Update()

    tcfun = vtk.vtkPiecewiseFunction()  # 不透明度传输函数---放在tfun
    tcfun.AddPoint(minValue, 0.0)
    # tcfun.AddPoint(minValue + 1, 0.3)
    tcfun.AddPoint(maxValue, 0.6)

    gradtfun = vtk.vtkPiecewiseFunction()  # 梯度不透明度函数---放在gradtfun
    gradtfun.AddPoint(0.0, 0.3)
    gradtfun.AddPoint(0.2, 0.4)
    gradtfun.AddPoint(0.6, 0.6)
    gradtfun.AddPoint(1.0, 1.0)

    ctfun = vtk.vtkColorTransferFunction()  # 颜色传输函数---放在ctfun
    ctfun.AddRGBPoint(minValue, 0.5, 0.0, 0.0)
    ctfun.AddRGBPoint(maxValue, 0.9, 0.2, 0.3)

    outline = vtk.vtkOutlineFilter()
    outline.SetInputConnection(img_arr.GetOutputPort())
    outlineMapper = vtk.vtkPolyDataMapper()
    outlineMapper.SetInputConnection(outline.GetOutputPort())
    outlineActor = vtk.vtkActor()
    outlineActor.SetMapper(outlineMapper)

    dims = img_arr.GetOutput().GetDimensions()
    print(dims)

    extractVOI = vtk.vtkExtractVOI()
    extractVOI.SetInputConnection(img_arr.GetOutputPort())
    extractVOI.SetVOI(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
    extractVOI.Update()

    print(extractVOI.GetOutput().GetDimensions())

    volumeMapper_src = vtk.vtkGPUVolumeRayCastMapper()  # 映射器volumnMapper使用vtk的管线投影算法
    # 向映射器中输入数据：shifter(预处理之后的数据)
    volumeMapper_src.SetInputData(extractVOI.GetOutput())

    volumeProperty = vtk.vtkVolumeProperty()  # 创建vtk属性存放器,向属性存放器中存放颜色和透明度
    volumeProperty.SetColor(ctfun)
    volumeProperty.SetScalarOpacity(tcfun)
    volumeProperty.SetGradientOpacity(gradtfun)
    volumeProperty.SetInterpolationTypeToLinear()  # ???
    volumeProperty.ShadeOn()

    render_volume = vtk.vtkVolume()  # 演员
    render_volume.SetMapper(volumeMapper_src)
    render_volume.SetProperty(volumeProperty)

    render.AddActor(outlineActor)
    render.AddVolume(render_volume)
    render.ResetCamera()

    sliderRep_min = vtk.vtkSliderRepresentation2D()
    sliderRep_min.SetMinimumValue(0)
    sliderRep_min.SetMaximumValue(10)
    sliderRep_min.SetValue(1)
    sliderRep_min.SetTitleText("minValue")
    sliderRep_min.SetSliderLength(0.025)
    sliderRep_min.SetSliderWidth(0.05)
    sliderRep_min.SetEndCapLength(0.005)
    sliderRep_min.SetEndCapWidth(0.025)
    sliderRep_min.SetTubeWidth(0.0125)
    sliderRep_min.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep_min.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep_min.GetPoint1Coordinate().SetValue(1 - 0.05 * count, 0.05)
    sliderRep_min.GetPoint2Coordinate().SetValue(1 - 0.05 * count, 0.45)

    sliderWidget_min = vtk.vtkSliderWidget()
    sliderWidget_min.SetInteractor(renWinInteractor)
    sliderWidget_min.SetRepresentation(sliderRep_min)
    sliderWidget_min.SetCurrentRenderer(render)
    sliderWidget_min.SetAnimationModeToAnimate()

    sliderRep_max = vtk.vtkSliderRepresentation2D()
    sliderRep_max.SetMinimumValue(0)
    sliderRep_max.SetMaximumValue(10)
    sliderRep_max.SetValue(9)
    sliderRep_max.SetTitleText("maxValue")
    sliderRep_max.SetSliderLength(0.025)
    sliderRep_max.SetSliderWidth(0.05)
    sliderRep_max.SetEndCapLength(0.005)
    sliderRep_max.SetEndCapWidth(0.025)
    sliderRep_max.SetTubeWidth(0.0125)
    sliderRep_max.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep_max.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep_max.GetPoint1Coordinate().SetValue(1 - 0.05 * count, 0.55)
    sliderRep_max.GetPoint2Coordinate().SetValue(1 - 0.05 * count, 0.95)

    sliderWidget_max = vtk.vtkSliderWidget()
    sliderWidget_max.SetInteractor(renWinInteractor)
    sliderWidget_max.SetRepresentation(sliderRep_max)
    sliderWidget_max.SetCurrentRenderer(render)
    sliderWidget_max.SetAnimationModeToAnimate()

    def update_minmax(obj, ev):
        # print(obj)
        minValue = sliderWidget_min.GetRepresentation().GetValue()
        maxValue = sliderWidget_max.GetRepresentation().GetValue()
        # # reset value
        if minValue >= maxValue:
            if obj == sliderWidget_max:
                sliderWidget_max.GetRepresentation().SetValue(max(maxValue, minValue + 0.01))
            elif obj == sliderWidget_min:
                sliderWidget_min.GetRepresentation().SetValue(min(maxValue - 0.01, minValue))
        minValue = sliderWidget_min.GetRepresentation().GetValue()
        maxValue = sliderWidget_max.GetRepresentation().GetValue()

        tcfun.RemoveAllPoints()
        tcfun.AddPoint(minValue, 0.0)
        tcfun.AddPoint(maxValue, 1.0)
        volumeProperty.SetScalarOpacity(tcfun)
        print('update_minmax')

    sliderWidget_min.AddObserver(
        vtk.vtkCommand.InteractionEvent, update_minmax)
    sliderWidget_max.AddObserver(
        vtk.vtkCommand.InteractionEvent, update_minmax)
    sliderWidget_min.EnabledOn()
    sliderWidget_max.EnabledOn()

    ##########################################################

    def getCropSlider(dim_index, dim_size):
        sliderRep_min = vtk.vtkSliderRepresentation2D()
        sliderRep_min.SetMinimumValue(0)
        sliderRep_min.SetMaximumValue(dim_size - 1)
        sliderRep_min.SetValue(0)
        sliderRep_min.SetSliderLength(0.025)  # 滑块 长度
        sliderRep_min.SetSliderWidth(0.025)  # 滑块 宽度
        sliderRep_min.SetEndCapLength(0.005)
        sliderRep_min.SetEndCapWidth(0.025)
        sliderRep_min.SetTubeWidth(0.0125)
        sliderRep_min.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        sliderRep_min.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        sliderRep_min.GetPoint1Coordinate().SetValue(0.05 * dim_index, 0.05)
        sliderRep_min.GetPoint2Coordinate().SetValue(0.05 * dim_index, 0.45)

        sliderWidget_min = vtk.vtkSliderWidget()
        sliderWidget_min.SetInteractor(renWinInteractor)
        sliderWidget_min.SetRepresentation(sliderRep_min)
        sliderWidget_min.SetCurrentRenderer(render)
        sliderWidget_min.SetAnimationModeToAnimate()

        sliderRep_max = vtk.vtkSliderRepresentation2D()
        sliderRep_max.SetMinimumValue(0)
        sliderRep_max.SetMaximumValue(dim_size - 1)
        sliderRep_max.SetValue(dim_size - 1)
        sliderRep_max.SetSliderLength(0.025)
        sliderRep_max.SetSliderWidth(0.025)
        sliderRep_max.SetEndCapLength(0.005)
        sliderRep_max.SetEndCapWidth(0.025)
        sliderRep_max.SetTubeWidth(0.0125)
        sliderRep_max.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        sliderRep_max.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        sliderRep_max.GetPoint1Coordinate().SetValue(0.05 * dim_index, 0.55)
        sliderRep_max.GetPoint2Coordinate().SetValue(0.05 * dim_index, 0.95)

        sliderWidget_max = vtk.vtkSliderWidget()
        sliderWidget_max.SetInteractor(renWinInteractor)
        sliderWidget_max.SetRepresentation(sliderRep_max)
        sliderWidget_max.SetCurrentRenderer(render)
        sliderWidget_max.SetAnimationModeToAnimate()

        return sliderWidget_min, sliderWidget_max

    def update_crop(obj, ev):
        # print(obj)
        dim1_minValue = dim1_sliderWidget_min.GetRepresentation().GetValue()
        dim1_maxValue = dim1_sliderWidget_max.GetRepresentation().GetValue()
        dim2_minValue = dim2_sliderWidget_min.GetRepresentation().GetValue()
        dim2_maxValue = dim2_sliderWidget_max.GetRepresentation().GetValue()
        dim3_minValue = dim3_sliderWidget_min.GetRepresentation().GetValue()
        dim3_maxValue = dim3_sliderWidget_max.GetRepresentation().GetValue()
        # # reset value
        if dim1_minValue >= dim1_maxValue:
            if obj == dim1_sliderWidget_max:
                dim1_sliderWidget_max.GetRepresentation().SetValue(
                    max(dim1_maxValue, dim1_minValue + 0.01))
            elif obj == dim1_sliderWidget_min:
                dim1_sliderWidget_min.GetRepresentation().SetValue(
                    min(dim1_maxValue - 0.01, dim1_minValue))
        if dim2_minValue >= dim2_maxValue:
            if obj == dim2_sliderWidget_max:
                dim2_sliderWidget_max.GetRepresentation().SetValue(
                    max(dim2_maxValue, dim2_minValue + 0.01))
            elif obj == dim2_sliderWidget_min:
                dim2_sliderWidget_min.GetRepresentation().SetValue(
                    min(dim2_maxValue - 0.01, dim2_minValue))
        if dim3_minValue >= dim3_maxValue:
            if obj == dim3_sliderWidget_max:
                dim3_sliderWidget_max.GetRepresentation().SetValue(
                    max(dim3_maxValue, dim3_minValue + 0.01))
            elif obj == dim3_sliderWidget_min:
                dim3_sliderWidget_min.GetRepresentation().SetValue(
                    min(dim3_maxValue - 0.01, dim3_minValue))

        dim1_minValue = dim1_sliderWidget_min.GetRepresentation().GetValue()
        dim1_maxValue = dim1_sliderWidget_max.GetRepresentation().GetValue()
        dim2_minValue = dim2_sliderWidget_min.GetRepresentation().GetValue()
        dim2_maxValue = dim2_sliderWidget_max.GetRepresentation().GetValue()
        dim3_minValue = dim3_sliderWidget_min.GetRepresentation().GetValue()
        dim3_maxValue = dim3_sliderWidget_max.GetRepresentation().GetValue()

        print(dim1_minValue, dim1_maxValue)
        print(dims)
        extractVOI.SetVOI(int(dim1_minValue), int(dim1_maxValue),
                          int(dim2_minValue), int(dim2_maxValue),
                          int(dim3_minValue), int(dim3_maxValue))
        extractVOI.Update()
        print(extractVOI.GetOutput().GetDimensions())
        print('update_crop')

    dim1_sliderWidget_min, dim1_sliderWidget_max = getCropSlider(
        1, dim_size=dims[0])
    dim2_sliderWidget_min, dim2_sliderWidget_max = getCropSlider(
        2, dim_size=dims[1])
    dim3_sliderWidget_min, dim3_sliderWidget_max = getCropSlider(
        3, dim_size=dims[2])

    dim1_sliderWidget_min.AddObserver(
        vtk.vtkCommand.InteractionEvent, update_crop)
    dim1_sliderWidget_max.AddObserver(
        vtk.vtkCommand.InteractionEvent, update_crop)
    dim2_sliderWidget_min.AddObserver(
        vtk.vtkCommand.InteractionEvent, update_crop)
    dim2_sliderWidget_max.AddObserver(
        vtk.vtkCommand.InteractionEvent, update_crop)
    dim3_sliderWidget_min.AddObserver(
        vtk.vtkCommand.InteractionEvent, update_crop)
    dim3_sliderWidget_max.AddObserver(
        vtk.vtkCommand.InteractionEvent, update_crop)

    dim1_sliderWidget_min.EnabledOn()
    dim1_sliderWidget_max.EnabledOn()
    dim2_sliderWidget_min.EnabledOn()
    dim2_sliderWidget_max.EnabledOn()
    dim3_sliderWidget_min.EnabledOn()
    dim3_sliderWidget_max.EnabledOn()

    return render, sliderWidget_min, sliderWidget_max


def getRenderofSeg(render,
                   renWinInteractor,
                   renWin,
                   numpyImage_segs,
                   spacing,
                   minValue=0, maxValue=10, pos=(0, 0, 1.0, 1.0)):

    volumeProperty_segs = []
    for i, numpyImage_seg in enumerate(numpyImage_segs):
        print("add seg")
        numpyImage_seg = numpyImage_seg.astype(
            np.float32) - np.min(numpyImage_seg)
        numpyImage_seg = maxValue * numpyImage_seg / np.max(numpyImage_seg)
        numpyImage_seg = (numpyImage_seg > 4) * 10.0

        img_arr_seg = vtkImageImportFromArray()
        img_arr_seg.SetArray(numpyImage_seg)
        img_arr_seg.SetDataSpacing(spacing)
        img_arr_seg.SetDataOrigin((0, 0, 0))
        img_arr_seg.Update()

        tcfun_seg = vtk.vtkPiecewiseFunction()  # 不透明度传输函数---放在tfun
        tcfun_seg.AddPoint(minValue+1, 0.3)
        tcfun_seg.AddPoint(maxValue, 0.8)

        gradtfun_seg = vtk.vtkPiecewiseFunction()  # 梯度不透明度函数---放在gradtfun
        gradtfun_seg.AddPoint(minValue, 0.0)
        gradtfun_seg.AddPoint(1.0, 0.9)
        gradtfun_seg.AddPoint(maxValue, 1.0)

        ctfun_seg = vtk.vtkColorTransferFunction()  # 颜色传输函数---放在ctfun
        ctfun_seg.AddRGBPoint(minValue, 0.9 * i, 0.9, 0.0)
        ctfun_seg.AddRGBPoint(maxValue, 0.9 * i, 0.9, 0.3)

        outline = vtk.vtkOutlineFilter()
        outline.SetInputConnection(img_arr_seg.GetOutputPort())
        outlineMapper = vtk.vtkPolyDataMapper()
        outlineMapper.SetInputConnection(outline.GetOutputPort())
        outlineActor = vtk.vtkActor()
        outlineActor.SetMapper(outlineMapper)

        volumeMapper_seg = vtk.vtkGPUVolumeRayCastMapper()  # 映射器volumnMapper使用vtk的管线投影算法
        # 向映射器中输入数据：shifter(预处理之后的数据)
        volumeMapper_seg.SetInputData(img_arr_seg.GetOutput())

        volumeProperty_seg = vtk.vtkVolumeProperty()  # 创建vtk属性存放器,向属性存放器中存放颜色和透明度
        volumeProperty_seg.SetColor(ctfun_seg)
        volumeProperty_seg.SetScalarOpacity(tcfun_seg)
        volumeProperty_seg.SetGradientOpacity(gradtfun_seg)
        volumeProperty_seg.SetInterpolationTypeToLinear()  # ???
        volumeProperty_seg.ShadeOn()
        volumeProperty_segs.append(volumeProperty_seg)

        render_volume_seg = vtk.vtkVolume()  # 演员
        render_volume_seg.SetMapper(volumeMapper_seg)
        render_volume_seg.SetProperty(volumeProperty_seg)

        render.AddActor(outlineActor)
        render.AddVolume(render_volume_seg)

    render.ResetCamera()

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
            if key.upper() == 'X':
                opacity = tcfun_seg.GetValue(0)
                if opacity:
                    print('Hide Label')
                    tcfun_seg.RemoveAllPoints()
                    tcfun_seg.AddPoint(minValue, 0.0)
                    tcfun_seg.AddPoint(maxValue, 0.0)
                    for volumeProperty_seg in volumeProperty_segs:
                        volumeProperty_seg.SetScalarOpacity(tcfun_seg)
                    renWin.Render()

                else:
                    print('Show Label')
                    tcfun_seg.RemoveAllPoints()
                    tcfun_seg.AddPoint(minValue+1, 0.3)
                    tcfun_seg.AddPoint(maxValue, 0.8)
                    for volumeProperty_seg in volumeProperty_segs:
                        volumeProperty_seg.SetScalarOpacity(tcfun_seg)
                    renWin.Render()

            if key == 'Down':
                # print('Down')
                # tfun.RemoveAllPoints()
                # tfun.AddPoint(1129, 0)
                renWin.Render()

    renWinInteractor.SetInteractorStyle(KeyPressInteractorStyle(
        parent=renWinInteractor))  # 在交互操作里面添加这个自定义的操作例如up,down
    return render
