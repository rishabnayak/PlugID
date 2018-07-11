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

inceptionNet = Take[NetModel["Inception V3 Trained on ImageNet Competition Data"], {1, -4}];

myinceptionnet = NetChain[Association["augLayer" -> ImageAugmentationLayer[{299, 299},
      "ReflectionProbabilities" -> {0.5, 0.5}], "pretrainednet" -> inceptionNet,
    "linear" -> LinearLayer[], "softmax" -> SoftmaxLayer[]],"Input" ->
 NetEncoder[{"Image", {299, 299}, "MeanImage" -> {0.5, 0.5, 0.5}}], "Output" -> NetDecoder[{"Class", connList}]];

trainedinceptionNet = NetTrain[myinceptionnet, trainingset, All, BatchSize -> 32,
      LearningRateMultipliers -> {"linear" -> 1, _ -> 0}, ValidationSet -> validationset,
      TargetDevice -> "GPU", TrainingProgressCheckpointing -> {"File","checkpointinceptionnet.wlnet"}];

Export["trainedinceptionNet.wlnet",trainedinceptionNet["TrainedNet"]];

Export["trainedinceptionNet.mx",trainedinceptionNet];
