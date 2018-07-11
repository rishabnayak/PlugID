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

vgg19Net = Take[NetModel["VGG-19 Trained on ImageNet Competition Data"], {1, -9}];

myvgg19net = NetChain[Association["augLayer" -> ImageAugmentationLayer[{224, 224},
      "ReflectionProbabilities" -> {0.5, 0.5}], "pretrainednet" -> vgg19Net, "linear" -> LinearLayer[],
    "softmax" -> SoftmaxLayer[]], "Input" ->
 NetEncoder[{"Image", {224, 224},
   "MeanImage" -> {0.4850196078431373, 0.457956862745098,
     0.4076039215686274}}],"Output" -> NetDecoder[{"Class", connList}]];

trainedvgg19Net = NetTrain[myvgg19net, trainingset, All, BatchSize -> 32,
      LearningRateMultipliers -> {{"pretrainednet", "conv5_4"} -> 1,{"pretrainednet", "conv5_3"} -> 1,"linear" -> 1, _ -> 0}, ValidationSet -> validationset,
      TargetDevice -> "GPU", TrainingProgressCheckpointing -> {"File","checkpointvgg19net.wlnet"}];

Export["trainedvgg19Netv2.wlnet",trainedvgg19Net["TrainedNet"]];

Export["trainedvgg19Netv2.mx",trainedvgg19Net];
