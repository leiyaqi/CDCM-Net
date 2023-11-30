# CDCM-Net
This repo is the official implementation of "CDDM-Net" as well as the follow-ups. we  propose a Confidence-based Dynamic Cross-modal Memory Network (CDCM-Net) for aesthetic quality evaluation of images with missing modalities. 
 It currently includes code and models for the following tasks: 
 # Updates
* 29/11/2023  
     We release  a portion the code of CDCM-Net
* .......  
      We will release the complete code after the review
# Model structure   

![asd](https://github.com/leiyaqi/CDCM-Net/assets/34058709/42159044-cb52-42c1-82f1-e5b51d86c8f4)  
     
Figure 1: Visual comparison of conventional multi-modal IAA architectures and our
proposed CDCM-Net. (a)Traditional deep multi-modal architecture require images and
comments as input. (b)The proposed CDCM-Net use memory blocks to remember the
inter-relationship of two modalities. The textual (i.e., target) modality is recalled from
memory by querying the visual (i.e., source) modality. Then, both the visual and the recalled
textual modalities are fused via a confidence-based dynamic fusion block to conduct
a trustworthy prediction.
# Requirements
* torch
* tqdm
* numpy
* PIL
* scikit-learn
* torchvision
* matplotlib
* clip
* pandas
# Usage
## clone this repository
  git clone [https://github.com/bo-zhang-cs/CACNet-Pytorch.git](https://github.com/leiyaqi/CDCM-Net.git)  
Download pretrained models:  
  https://pan.baidu.com/s/1pw0jd0w0HKXCVDLY_KrmbA?pwd=prx4
 
## Testing 
Annotate the training code then:  
python  main.py


## Training
python main.py

# License
The codes are released under the GNU 3.0.
