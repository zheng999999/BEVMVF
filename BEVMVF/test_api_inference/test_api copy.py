
from mmdet.models.builder import DETECTORS
from mmseg.apis import init_segmentor,inference_segmentor,show_result_pyplot


config = "test_api_inference/seg/deeplabv3_r18-d8_512x1024_80k_cityscapes.py"
check_point = "test_api_inference/seg/deeplabv3_r18-d8_512x1024_80k_cityscapes_20201225_021506-23dffbe2.pth"
filename = "test_api_inference/000031.png"

model = init_segmentor(config,check_point)

result = inference_segmentor(model,filename)
print(result)
show_result_pyplot(model,filename,result)
print("end")
