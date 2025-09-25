# ECSPLAIN: Explainability-Constrained Classifier for Pairing the Detection and the Localization of Moving Areas From SAR Interferograms

This repository is dedicated to the ECSPLAIN framework which consists in constraining the Class Activation Map (CAM) of a classifier network during training to perform a segmentation task.

One can find the paper relative to this in <a href="https://ieeexplore.ieee.org/abstract/document/11108247?casa_token=qtbrLcl9AuwAAAAA:QKmShJuyNKGCt9O0QkP2h8DOPgEGJpsw-VMZ3h6viX7yCbOJR3ASTzAnjryYWe0Bo-AOcjzGug"> ECSPLAIN: Explainability-Constrained Classifier for Pairing the Detection and the Localization of Moving Areas From SAR Interferograms </a>

The repository contains two folders :
<ul>
    <li> ecsplain:</b> used to run the proposed ECSPLAIN framework
    <li> <b>baselineNetworks:</b> used to run the comparative methods in the paper
</ul>

## Running the ECSPLAIN framework

To ensure that the code runs correctly, pleae follow the next steps:

<ol>
<li> After cloning the repository, first move to the ecsplain repository: </li>

```shell
cd ECSPLAIN/ecsplain
```

<li> Then ensure to complete the files "main.py" (l.57-67) and "inference.py" (l.50-61) with the paths corresponding to the dataset and to the result folders. </li>

<li> Train the network. One may modify the default configuration by specifying different arguments: 
    <ul>
        <li> --cam_type: the CAM to be used (either grad_cam, layer_cam, score_cam, norm_cam) </li>
        <li> --model_type: type of classifier (either resnet18 or resnet50) </li>
        <li> --wSegm: the weight for the segmentation loss </li>
        <li> --wClass: the weight for classification loss </li> 
        <li> --nb_epochs: number of epochs </li>
        <li> --lr: learning rate </li>
        <li> --batch_size </li>
        <li> --model: path to the model to be loaded from a checkpoint </li>
        <li> --segLoss: segmentation loss to use (MSE or BCE) </li>
        <li> --seed </li>
    </ul>
 </li>

```shell
python3 main.py [--options]
```

<li> Test the network. One may modify the default configuration by specifying different arguments:
    <ul>
        <li> --cam_type: CAM to be used </li>
        <li> --model: path to the model to be loaded </li>
        <li> --save_ims: whether to same images after computing the metrics </li>
        <li> --thresh: specific threshold to be used to compute the metrics </li>
        <li> --threshNb: Number of threshold to be tested to find the best metrics</li>
        <li> --cuda: whether to use or not a GPU </li>
        <li> --seed </li>
    </ul>
</li>

```shell
python3 inference.py --model ./Path/To/The/Model/To/Be/Loaded.pth [--options]
```
</ol>

## Running baseline models

To ensure that the code runs correctly, pleae follow the next steps:

<ol>
<li> After cloning the repository, first move to the baselineNetworks repository: </li>

```shell
cd ECSPLAIN/baselineNetworks
```

<li> Then ensure to complete the files "main.py" (l.47-52) and "inference.py" (l.32-40) with the paths corresponding to the dataset and to the result folders. </li>

<li> Train the network. One may modify the default configuration by specifying different arguments: 
    <ul>
        <li> --model_type: type of the architecture (either UNet, ResUNet, SepUNet, FCN, DeepLabV3) </li>
        <li> --nb_epochs: number of epochs </li>
        <li> --lr: learning rate </li>
        <li> --batch_size </li>
        <li> --model: path to the model to be loaded from a checkpoint </li>
        <li> --seed </li>
    </ul>
 </li>

```shell
python3 main.py [--options]
```

<li> Test the network. Contrary to the ecsplain folder, the current implementation requires that the path to the model is given in line 37. In addition, it will compute the metrics for 100 thresholds (modifiable at line 64). The code will be updated to match the ecsplain folder as soon as possible.</li>

```shell
python3 inference.py --model ./Path/To/The/Model/To/Be/Loaded.pth [--options]
```
</ol>

## Citation

If this work was useful for you, please ensure citing our works :

<i>ECSPLAIN: Explainability Constrained-claSsifier for Pairing the detection and the Localization of moving Areas from SAR INterferograms, Bralet, A., Atto, A. M., Chanussot, J., & Trouvé, E., in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2025.3595267.</i>

## Any troubles ?

If you have any troubles with the article or the code, do not hesitate to contact us !
