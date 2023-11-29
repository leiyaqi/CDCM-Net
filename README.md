# CDCM-Net
This repo is the official implementation of "CDDM-Net" as well as the follow-ups. we  propose a Confidence-based Dynamic Cross-modal Memory Network (CDCM-Net) for aesthetic quality evaluation of images with missing modalities. 
 It currently includes code and models for the following tasks: 
 # Updates
* 29/11/2023  
     We release  a portion the code of CDCM-Net
* .......  
      We will release the complete code after the review
# Introduction 
Image aesthetic assessment (IAA) aims to design algorithms that can make
human-like aesthetic decisions. Due to its high subjectivity and complexity,
visual information alone is limited to fully predict the image’s aesthetic
quality. More and more researchers try to use complementary information
from user comments. But the user comments are not always available for
various technical and practical reasons. To this end, it is necessary to find a
way to reconstruct the missing textual information for aesthetic prediction
with visual information only. This paper solves the problem by proposing
a Confidence-based Dynamic Cross-modal Memory Network (CDCM-Net).
Specifically, the proposed CDCM-Net consists of two key components: Visual
and Textual Memory (VTM) network and Confidence-based Dynamical
Multi-modal Fusion module (CDMF). VTM is based on the key-value memory
network. It consists of a visual key memory and a textual value memory.
The visual key memory learns the visual information. In contrast, the textual
value memory learns to remember the textual feature and align them with the
corresponding visual features. During the inference, textual information can
be reconstructed using visual features only. Furthermore, we also introduce a
(CDMF) module to perform trustworthy fusion. CDMF evaluates modalitylevel
informativeness and then dynamically integrates reliable information.
Extensive experiments are performed to demonstrate the superiority of the
proposed method.
                  

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
  Usage
Testing
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
The code and DPC2022 dataset are released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for NonCommercial use only. Any commercial use should get formal permission first.
