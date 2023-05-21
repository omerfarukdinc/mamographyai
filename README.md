# mamographyai

First, the image numbers belonging to the three different classes in the dataset, namely BI-RADS0, BI-RADS1-2, and BI-RADS4-5, have been separated. For the training phase, approximately balanced numbers of different breast compositions (A, B, C, D) within each class were adjusted, and a total of 1770 patients were used. Before the algorithm training, the images go through a pre-processing stage. Due to the different breast sizes of each patient, the images are cropped up to the nipple and aligned to face the same direction. The color tone behind the breast image is examined, and the background is set to black. Then, the single-channel image is converted to three channels (RGB) using the Contrast Limited Adaptive Histogram Equalization (CLAHE) algorithm [2].
As seen in the algorithm below, the mediolateral oblique (MLO) and craniocaudal (CC) images of a single patient's right and left breasts are inputted into the algorithm simultaneously. After the convolutional layers, the feature maps of the images are flattened into a one-dimensional vector and processed in the decision layer. The output provides a probability distribution for 12 classes (BI-RADS0-A, BI-RADS0-B, BIRADS0-C, BI-RADS0-D, etc.). As a result, the patient's BI-RADS class and breast composition are provided. Quadrant information is obtained using the Grad-CAM algorithm [3]. Grad-CAM provides information on where it predominantly focuses for the predicted class. This allows for determining which quadrant of the patient's right or left breast was examined for classification.

![image](https://github.com/omerfarukdinc/mamographyai/assets/96438908/413c7432-f4be-416f-ad07-24baddcd1ccb)

Results and Analysis:
After the training phase, the weights obtained were used to test the system with 1200 patients. In the testing system, the BI-RADS classes were distributed equally. No data augmentation was performed in the dataset. When examining patient images, it was observed that there were some noisy images. Therefore, during training, we introduced noise to the images with a probability of 0.2 to train the algorithm. The learning of the algorithm was completed with a learning rate of 10-5 (gradient learning rate) and 122 epochs. To evaluate the results, the patient's BI-RADS classes, breast composition, and quadrant information were used to calculate recall and precision values, which are shown in Table-1. Due to the unequal distribution of the numbers of breast compositions in the data classes, a low F1-score was observed in breast composition [5].

![mamo_omer](https://github.com/omerfarukdinc/mamographyai/assets/96438908/cc411bcf-4ee5-40c4-9e0e-540c200578aa)
