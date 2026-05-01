from rknn.api import RKNN

rknn = RKNN(verbose=False)
rknn.config(
    target_platform='rk3588',
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]]
)
rknn.load_onnx(model='yolo11n.onnx')
rknn.build(do_quantization=False)
rknn.export_rknn('yolo11n.rknn')
rknn.release()
print('转换完成')
