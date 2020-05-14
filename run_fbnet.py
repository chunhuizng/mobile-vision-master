from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
from ptflops import get_model_complexity_info
import torch
from thop import profile


model_name = None # set this by yourself. What's more, dmasking is FBNetv2's another name. fbnet is FBNetv1.
if model_name == "dmasking_f4_1.08_1.08_300":
  model = fbnet("dmasking_f4_1.08_1.08_300",scale_factor=1.06, pretrained=False) # 300.03M
elif model_name == "dmasking_f4_1.2_1.2_400":
  model = fbnet("dmasking_f4_1.2_1.2_400",scale_factor=1.2, pretrained=False)  # 405.23M
elif model_name == "dmasking_f4_1.28_1.28_500":
  model = fbnet("dmasking_f4_1.28_1.28_500",scale_factor=1.35, pretrained=False)  # 500.15M
elif model_name == "fbnet_a_1.07_1.07_300":
  model = fbnet("fbnet_a_1.07_1.07_300",scale_factor=1.125, pretrained=False)  # 330.15M
elif model_name == "fbnet_a_1.2_1.17_400":
  model = fbnet("fbnet_a_1.2_1.17_400",scale_factor=1.2, pretrained=False)  # 410.15M
elif model_name == "fbnet_a_1.25_1.26_500":
  model = fbnet("fbnet_a_1.25_1.26_500", scale_factor=1.325, pretrained=False)  # 508.15M
else:
  raise NotImplementedError
model.eval()
with torch.no_grad():
  net = model
  macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))
  input = torch.randn(1, 3, 224, 224)
  macs, params = profile(model, inputs=(input,))
  print('macs:', macs/1e6)
