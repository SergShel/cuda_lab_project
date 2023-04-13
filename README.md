### cuda_lab_project
Authors: Siarhei Sheludzko, Patrick Schwemmer, Sebastián Gómez Ruiz

# Video Semantic Segmentation with Recurrent U-Net
The idea for this work is taken from the paper 
[Recurrent U-Net for Resource-Constrained Segmentation.](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Recurrent_U-Net_for_Resource-Constrained_Segmentation_ICCV_2019_paper.pdf)<br>
Training and evaluation is done with  [Cityscapes dataset.](https://www.cityscapes-dataset.com/)


## Introduction

Semantic segmentation and video semantic segmentation are dense-prediction vision tasks,
which goal is to predict a semantic class for every single pixel of an image, or respectively for every pixel for every frame that forms a video sequence. 
Its motivation comes from the wide range of applications they can be used at, such as object detection and tracking, 
which plays an important role for e.g. autonomous driving systems. We focus on video semantic segmentation, 
which brings additional challenges in comparison to its non-temporal variant. One of them being temporal consistency,
meaning that coherence of the segmentation results of adjacent frames should be present while taking into account the state of an object, 
such as its motion and appearance. Other difficulties arise directly from objects' movement, such as occlusion or motion blur. 

We use an [UNet based architecture neural network](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) to tackle this task, which we adapt with convolutional 
temporal modules in order to take advantage of the temporal nature of the data and overcome the previously mentioned challenges.
Multiple variations of the models are implemented, as well as trained and evaluated on the [Cityscapes dataset](https://www.cityscapes-dataset.com/), 
a benchmark large-dataset that provides semantic and pixel-wide annotations for urban street scenery. 
The different model configurations and results are presented and analyzed in detail in further sections, as well as the process itself. 
In which we contrast a baseline result obtained by using the model to process the data in a frame by frame manner, 
with the results obtained by using the model to process the data with the temporal modules in a sequenced (multiple-frames) manner. 
Analysis is given through the whole method itself and results, for the model design and its evaluation, 
we also discuss about limitations during the development of the project itself, difficulties and challenges of the process and task as a whole, 
as well as possible aspects of improvement.
