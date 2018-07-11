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

imageidentifyNet = Take[NetModel["Wolfram ImageIdentify Net for WL 11.1"], {1, -3}];

myimageidentifynet = NetChain[Association["auglayer" -> ImageAugmentationLayer[{224, 224},
"ReflectionProbabilities" -> {0.5, 0.5}], "pretrainednet" -> imageidentifyNet, "linear" -> LinearLayer[],
    "softmax" -> SoftmaxLayer[]], "Input" -> NetEncoder[{"Image", {224, 224}, "MeanImage" -> {0.48, 0.46, 0.4}}], "Output" -> NetDecoder[{"Class", connList}]];

trainedimageidentifyNet = NetTrain[myimageidentifynet, trainingset, All,
        BatchSize -> 32, LearningRateMultipliers -> {{"pretrainednet", "5b"} -> 1,
          "linear" -> 1, _ -> 0}, ValidationSet -> validationset, TargetDevice -> "GPU", TrainingProgressCheckpointing -> {"File","checkpointimageidentifynet.wlnet"}];

Export["trainedimageidentifyNetv2.wlnet",trainedimageidentifyNet["TrainedNet"]];

Export["trainedimageidentifyNetv2.mx",trainedimageidentifyNet];
