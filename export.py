# coding=utf-8
from my_utils.load_checkpoint import load_checkpoint
import torch
import onnx
import onnx_graphsurgeon as gs
from io import BytesIO
from my_utils.events import LOGGER
from models.end2end import End2End
weights = 'weights/yolov5s.pt'
topk_all = 100
iou_thres = 0.5
conf_thres = 0.45
dynamic_batch = True
end2end = True
model = load_checkpoint(weights, 'cpu')
batch_size = 1
height = 640
width = 640
img = torch.randn(batch_size, 3, height, width)
model.eval()
if dynamic_batch:
    batch_size = 'batch'
    dynamic_axes = {
        'images': {
            0: 'batch',
        }, }
    if end2end:
        output_axes = {
            'num_dets': {0: 'batch'},
            'det_boxes': {0: 'batch'},
            'det_scores': {0: 'batch'},
            'det_classes': {0: 'batch'},
        }
    else:
        output_axes = {
            'outputs': {0: 'batch'},
        }
    dynamic_axes.update(output_axes)
if end2end:
    model = End2End(model, max_obj=topk_all, iou_thres=iou_thres, score_thres=conf_thres,
                    device='cpu', ort=False, trt_version=8, with_preprocess=False)
try:
    LOGGER.info('\nStarting to export ONNX...')
    export_file = weights.replace('.pt', '.onnx')  # filename
    with BytesIO() as f:
        torch.onnx.export(model, img, f, verbose=False, opset_version=13,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=['images'],
                          output_names=['num_dets', 'det_boxes', 'det_scores', 'det_classes']
                          if end2end else ['outputs'], dynamic_axes=dynamic_axes)
        f.seek(0)
        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        graph = gs.import_onnx(onnx_model)
        graph.cleanup().toposort()  #从图形中删除未使用的节点和张量，并对图形进行拓扑排序
         # Shape Estimation
        estimated_graph = None
        try:
            estimated_graph = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
        except:
            estimated_graph = gs.export_onnx(graph)
        onnx.save(estimated_graph, export_file)
except Exception as e:
    LOGGER.info(f'ONNX export failure: {e}')




