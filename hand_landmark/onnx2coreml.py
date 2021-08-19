import coremltools  ## 4.1
from coremltools.proto.NeuralNetwork_pb2 import NeuralNetworkPreprocessing
import tflite2onnx

tflite_path = "./hand_landmark.tflite"
onnx_path = "./hand_landmark.onnx"
tflite2onnx.convert(tflite_path, onnx_path)


mlmodel = coremltools.converters.onnx.convert(
    model = onnx_path,
)

spec = mlmodel.get_spec()
spec_layers = getattr(spec, spec.WhichOneof("Type")).layers

unary_list = []
for index, layer in enumerate(spec_layers):
    if spec_layers[index].unary.type in [x for x in range(1,10)]:
        unary_list.append(index)

def unaryToClip(layers, index):
    params = layers[index].clip
    params.minVal = 0 
    params.maxVal = 6
for i in unary_list:
    unaryToClip(spec_layers, i)

p = NeuralNetworkPreprocessing()
p.featureName =  spec.description.input[0].name
p.scaler.channelScale = 1/ 255.
spec.neuralNetwork.preprocessing.append(p)



c,w,h = spec.description.input[0].type.multiArrayType.shape
spec.description.input[0].type.imageType.width = w
spec.description.input[0].type.imageType.height = h
spec.description.input[0].type.imageType.colorSpace = coremltools.proto.FeatureTypes_pb2.ImageFeatureType.RGB
coremltools.utils.save_spec(spec, "./hand_landmark.mlmodel")

new_mlmodel = coremltools.models.MLModel(spec)
