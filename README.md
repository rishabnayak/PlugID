# PlugID

This project was created as part of WSS2018, as an exploration into the use of neural networks to classify images. It went on to analyze the implications of using different network frameworks on accuracy and compute time, and understand the tradeoffs.

The increasing use of Neural Networks in recent times have necessitated an analysis of their relative performance in terms of accuracy and evaluation time, and further, the compromises one makes while trying to use a network that is fast to evaluate.

Goal
====

The goal of this project was to analyze the performance of different Neural Networks to solve the same problem: Identify plugs and connectors from its image. This would allow us to understand the implications of using different network frameworks on accuracy and compute time, and understand the tradeoffs.

Networks - v1
=============

These Neural Networks were trained using Transfer Learning, using the first few layers of an original, trained network, leaving them unchanged and frozen, adding an ImageAugmentation Layer at the beginning of the Network to produce reflected versions of the input images along both the x and y axes, with a probability of 0.5 for both. The networks were cut off before the final Linear Layer, and a new Linear Layer was added to reflect the different output dimensions, which was followed up with a Softmax Layer to create a final one-hot vector for the output. The Output Decoder was setup to recognize classes, the classes being the names of the recognizable entities. The networks used to extracted the pretrained networks were:

 - Ademxapp Model A
 - Inception V3
 - ResNet-152
 - VGG-19
 - ImageIdentify

Networks - v2
=============

These Neural Networks were yet again trained using Transfer Learning, similar to v1 Networks, adding an ImageAugmentation Layer at the beginning, and a Linear Layer followed by a Softmax Layer at the end of the network. What was different was that that the entire network wasn't frozen, the last few layers of each of these networks were retrained, which boosted up the accuracy significantly, as will be shown in the data.

Analysis
========


----------


Best Performer: Accuracy
-------
The Ademxapp v2 Net trained on a dataset containing a total of 24000 images of 32 Port and connector categories was 91% accurate at classifying input images. This network is a high performance model which aimed to find a proper depth for ResNets, improving on the feature search method, trying to classify images accurately without grid-searching the whole space. As a result, the original network had only 17 residual units, and outperformed various deeper architectures. 

Modifying the Network, I added an ImageAugmentation Layer to augment the dataset, producing reflected versions of the input images along both the x and y axes, with a probability of 0.5 for both. The residual units 6a and 7a were allowed to retrain, along with the Batch Normalization Layer bn7, and the penultimate Linear Layer, which was followed up with a Softmax Layer. The final NetChain was - 

![Ademxapp v2 NetChain][1]

The error rate fell quite fast during the first few training rounds, and then plateaued. The validation error fluctuated in tandem with the training error, a good sign, showing that the network was stable.

![Ademxapp v2 Error Rate Evolution][2]

Best Performer: Evaluation Time
-------
The ImageIdentify v2 network, again was trained on the same dataset used to train the Ademxapp v2 network. This network is based off of Wolfram's Image Identify Network, and uses transfer learning to speed up the training process.  This network was the fastest in terms of runtime, but suffered in terms of accuracy. The network was right only 82% of the time, but evaluated in merely 0.062s. It also had a much smaller memory footprint, weighing in at 42MB. 

This network was modified similarly to the Ademxapp v2 network to augment images, and the NetGraph Layer 5b was allowed to retrain. The network architecture was - 

![ImageIdentify v2 NetChain][3]

This network converged satisfactorily too, however, the gap between the validation and training error evolutions were more noticeable, suggesting that the network wasn't as great at recognizing patterns as the Ademxapp v2 network.

![ImageIdentify v2 Error Rate Evolution][4]

Results
=======
Below, I present the results of the speed and accuracy comparison of multiple networks evaluated on the same dataset containing images of plugs and connectors.

![Speed and Accuracy][5]

Conclusions
=======

On analyzing the data, it was found that the Ademxapp v2 network performed the best, with an accuracy of 91%, but also was one of the slowest networks, having an evaluation time of 1.10s. The fastest network was the ImageIdentify v2 network, with an evaluation time of 0.062s, however, it compromised on accuracy, dropping to 82%. 

From the data collected, it appears that there exists a direct relationship between accuracy and speed for Neural Networks. As the depth of the network increases, the accuracy increases, corresponding to an increase in evaluation time. 

Future Work
=======

 - Use a more diverse dataset, recognize more categories of images.
 - Implement a faster version of the network/networks as an API for use as an app for mobile phones, part of a larger application that provides users with contextual awareness about their environment.

Author Contact Information
=======

<a href="mailto:rishab@bu.edu">rishab@bu.edu</a>

Github Repo - <a href="https://github.com/rishab2113/PlugID">Click Here</a>

  [1]: http://community.wolfram.com//c/portal/getImageAttachment?filename=Untitled.png&userId=1352120
  [2]: http://community.wolfram.com//c/portal/getImageAttachment?filename=ScreenShot2018-07-11at3.21.56PM.png&userId=1352120
  [3]: http://community.wolfram.com//c/portal/getImageAttachment?filename=Untitled1.png&userId=1352120
  [4]: http://community.wolfram.com//c/portal/getImageAttachment?filename=ScreenShot2018-07-11at3.29.48PM.png&userId=1352120
  [5]: http://community.wolfram.com//c/portal/getImageAttachment?filename=6529Untitled.png&userId=1352120
