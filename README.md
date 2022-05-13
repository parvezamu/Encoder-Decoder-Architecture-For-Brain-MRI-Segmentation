This is our solution used for Multimedia Tools and Applications paper(https://link.springer.com/article/10.1007/s11042-021-10915-y). 

Introduction:

Inspired by the success of the residual and dense connections, dilation, and the ASPP techniques, we have proposed a variant form of 3D U-Net with the combination of the residual connections, dilation, and dense ASPP. We have offered an RD2A (Residual-Dilated Dense ASPP) 3D U-Net model. The key contributions of this study are given below:

1.	A variant form of 3D U-Net. We used a combined approach of residual connections and densely connected ASPP
2.	To avoid possible loss of information during training in the proposed model, we choose appropriate rates of dilation layer to gain the proper size of the receptive field on BRATS datasets. Additionally, we used dense connections among the multiple sizes of the receptive field in ASPP on the feature maps of a residual-dilated 3D U-Net model for exploiting the full contextual information of the 3D brain MRI datasets.
3.	We have worked on BRATS 2018 and BRATS 2019 datasets, where the proposed model achieved state-of-the-art performances compared to other recent methods in terms of both parameters and accuracy.
4.	We have worked on iSeg-2019 datasets and achieved the best scores on the testing dataset against the best method of the iSeg-2019 validation dataset.


Server:

V100 GPU of 16 GB

ubuntu16.04 + virtualenv + python==3.5 + tensorflow-gpu==1.8.0 + keras==2.2.4

Packages: Here are some packages you may need to install.


For n4itk bias correction (https://ieeexplore.ieee.org/document/5445030) preprocessing, you may need to install ants.
a.) Just follow the installation guide on their homepage here (http://neuro.debian.net/install_pkg.html?p=ants) 
b.) Add ants to your environment variable PATH, for instance like $ export PATH=${PATH}:/usr/lib/ants/

sudo apt-get install libhdf5-serial-dev

pip install numpy, nibable, SimpleITK, tqdm, xlrd, pandas, progressbar, matplotlib, nilearn, sklearn, tables

For Instance Normalization, you may need to download and install keras-contrib
a.) git clone https://www.github.com/farizrahman4u/keras-contrib.git 
b.) pip install

You can also install instance normalization

pip install git+https://www.github.com/keras-team/keras-contrib.git

Please follow the "How to run it" section from here (https://github.com/woodywff/brats_2019).

Acknowledgement:

Again, this work refers to Isensee et.al's paper (https://www.nature.com/articles/s41592-020-01008-z), ellisdg's repository (https://github.com/ellisdg/3DUnetCNN/tree/master/legacy) and brats_2019's repository (https://github.com/woodywff/brats_2019). We deeply appreciate their contributions to the community. Many thanks to the host of the BraTS (https://www.med.upenn.edu/sbia/brats2018/data.html) and iSeg-2019 (https://ieeexplore.ieee.org/document/9339962) datasets.
