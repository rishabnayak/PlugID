(* ::Package:: *)

connList = {"MIDImale", "MIDIfemale", "RJ45male", "RJ45female", "TOSLINKmale", "TOSLINKfemale",
    "compositevideomalecable", "compositevideoport", "componentvideocable", "componentvideoport",
    "VGAcable", "VGAport", "dvicable", "dviport", "minidisplayportcable", "minidisplayportport",
    "HDMIcable", "HDMIport", "DisplayPortcable", "DisplayPortport", "usbamale", "usbafemale", "usbcmale",
    "usbcfemale", "microusbfemale", "microusbmalecable", "firewire800cable", "firewire800port",
    "firewire400port", "firewire400cable", "coaxcable", "coaxport"};

trainingrules = Import["Test.mx"];

validationrules = Import["Validation.mx"];

trainingset = RandomSample[trainingrules, All];
validationset = RandomSample[validationrules, All];

ademNet = Take[NetModel["Ademxapp Model A Trained on ImageNet Competition Data"], {1, -3}];

myademnet = NetChain[Association["augLayer" -> ImageAugmentationLayer[{320, 320},
      "ReflectionProbabilities" -> {0.5, 0.5}], "pretrainednet" -> ademNet, "linear" -> LinearLayer[],
    "softmax" -> SoftmaxLayer[]],"Input" ->
 NetEncoder[{"Image", {320, 320},
   "MeanImage" -> {0.485, 0.456, 0.406},
   "VarianceImage" -> {0.0524, 0.0502, 0.0506}}], "Output" -> NetDecoder[{"Class", connList}]];

trainedademNet = NetTrain[myademnet, trainingset, All, BatchSize -> 32,
LearningRateMultipliers -> {{"pretrainednet", "6a"} -> 1,
  {"pretrainednet", "7a"} -> 1, {"pretrainednet", "bn7"} -> 1, "linear" -> 1,
  _ -> 0}, ValidationSet -> validationset, TargetDevice -> "GPU", TrainingProgressCheckpointing -> {"File","checkpointademnet.wlnet"}];

Export["trainedademNetv2.wlnet",trainedademNet["TrainedNet"]];

Export["trainedademNetv2.mx",trainedademNet];
