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

resNet = Take[NetModel["ResNet-152 Trained on ImageNet Competition Data"], {1,-4}];

myresnet = NetChain[Association["augLayer" -> ImageAugmentationLayer[{224, 224},
"ReflectionProbabilities" -> {0.5, 0.5}], "pretrainednet" -> resNet, "linear" -> LinearLayer[],
    "softmax" -> SoftmaxLayer[]],"Input" -> NetEncoder[{"Image", {224, 224}}], "Output" -> NetDecoder[{"Class", connList}]];

trainedresNet = NetTrain[myresnet, trainingset, All, BatchSize -> 32,
      LearningRateMultipliers -> {"linear" -> 1, _ -> 0}, ValidationSet -> validationset,
      TargetDevice -> "GPU", TrainingProgressCheckpointing -> {"File","checkpointresnet.wlnet"}];

Export["trainedresNet.wlnet",trainedresNet["TrainedNet"]];

Export["trainedresNet.mx",trainedresNet];
