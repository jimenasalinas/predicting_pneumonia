# Predicting Pneumonia

We develop a computer vision deep learning model to predict pneumonia from chest X-ray images in Pytorch, with the image dataset obtained from Kermany et al. (2018) in Kaggle. We first start with an overview of the literature on deep learning models used for medical imaging and discuss their advantages and limitations, taking a particular look at a successful model used to predict different pulmonary diseases as motivation for our project. We then describe our technical approach to data preparation and construction of our model. Finally, we discuss our model results through accuracy and F1 metrics. We find that our baseline two-layered convolutional neural network and the alternate three-layer model both have a testing accuracy and F1 score around 80%.

## Introduction

Deep learning techniques have gained significant attention in the medical field for their potential to improve disease identification and diagnosis. In recent years, several studies have explored the application of deep learning methods in specific medical domains, such as lung cancer detection, dermatology, and most recently, Covid-19 diagnosis.

Kingsley Kuan et al. (2017)  focused on accurately identifying lung cancer using deep learning models applied to computed tomography (CT) scans. By participating in the Kaggle Data Science Bowl 2017 challenge, the authors demonstrated the effectiveness of convolutional neural networks (CNNs) in detecting lung cancer. The authors built a multi-stage framework that detected nodules in 3D lung scans, determined if each nodule was malignant, and finally, assigned a cancer probability based on those results. They analyzed the classifier performance using sensitivity (the true positive rate) and specificity (the true negative rate) instead of accuracy. By changing the strides values of the convolutional filter they obtained an improvement of sensitivity and F1 score.

In a study by Ophir Gozesi et al. (2020), the authors employed convolutional neural networks to analyze CT images of COVID-19 patients. By leveraging large datasets of CT scans, the researchers developed an automated system for detecting COVID-19 infection and monitoring disease progression. The deep learning models exhibited promising results in accurately identifying COVID-19 cases and distinguishing them from other respiratory diseases. This research holds significant implications for enhancing diagnostic capabilities and enabling timely interventions, especially in regions with limited resources or overwhelmed healthcare systems.

In another study, Hyeon Ki Jeong et al. (2022) conducted a systematic review that provides a comprehensive overview of deep learning approaches in dermatology, covering applications such as skin lesion classification, melanoma detection, and disease diagnosis. The authors discussed various deep learning techniques employed in dermatology, including CNNs, recurrent neural networks (RNNs), and generative adversarial networks (GANs). The review emphasized the outcomes achieved by these approaches while highlighting limitations related to dataset availability and model interpretability.

These studies collectively demonstrate the potential of deep learning and machine learning techniques in medical applications. Kingsley Kuan et al. (2017) showcased the efficacy of deep learning in lung cancer detection, while Gozesi et al. (2020) explored the use of machine learning for COVID-19 diagnosis based on symptoms. Jeong et al. (2022) provided insights into the current state of deep learning approaches in dermatology. These studies emphasize the importance of appropriate model design, training strategies, and dataset availability for accurate and reliable results.

However, limitations were identified across the studies. Kingsley Kuan et al. (2017) and Jeong et al. (2022) noted challenges related to the generalizability of the models to real-world clinical settings and the availability of diverse datasets. A particular point raised by the latter authors is that several ML algorithms tend to underperform on images from patients with skin of color because the datasets used to train these models have been collected heavily from fair skinned patients (Jeong et. al, 2022). There is a need for studies to deploy models for validation in real-world settings in which the models will be used. Addressing these limitations will be crucial for the wider adoption and clinical implementation of deep learning and machine learning techniques in healthcare. 

When it comes to our specific application of computer vision for the prediction of pneumonia, the model CheXNet, built by Rajpurkar et al. (2017), stands out. Rajpurkar et al. (2017) developed a 121 layered convolutional neural network trained on the largest available chest X-ray dataset publicly available exhibiting 14 different diseases. Their approach consisted of having radiologists label images according to their expert knowledge, and comparing the model’s predictions to that of the radiologists. The researchers found CheXNet had a better performance than the radiologists when measured with the F1 metric, which takes into account the balance between precision and recall important for reducing the false positive and false negative rates in medical diagnoses.

Rajpurkar et al. (2017) found that compared to other diseases, pneumonia was one of the hardest to diagnose. With their state-of-the-art model, they were able to accurately predict pneumonia about 77% of the time. For diseases like Cardiomegaly and Hernia, CheXNet’s accuracy rose to about 92%. Due to our limitations in computational power and time constraints, our model for this classification task is heavily simplified from ChestXNet, with only two convolutional and one linear layers. Regardless, our model was successfully able to predict pneumonia from our chest X-ray dataset successfully at a higher rate, showing the potential simple deep learning models have for the future of medical imaging and health policy, while taking into account the limitations that these models can have when human lives are at risk.


## Technical Approach

### Data Set

To carry out our binary classification project, we utilized the Kermany et al. labeled dataset titled "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification" as our primary source of data. This dataset, obtained from Kaggle, but originally sourced from Mendeley Data, provided us with a comprehensive collection of labeled chest X-ray images in pediatric patients. As illustrated in figure 1 below, each image in the dataset has been annotated to indicate whether pneumonia is present or not. Since our aim with this analysis was to develop a deep-learning model capable of accurately predicting whether a given chest X-ray indicates pneumonia, this data set allowed us to leverage pneumonia labels to train and evaluate different deep-learning models. Outside of this project, this dataset has played a vital role in advancing computer vision research within the medical field.

Figure 1

<img width="584" alt="Screen Shot 2023-09-19 at 8 05 49 PM" src="https://github.com/jimenasalinas/predicting_pneumonia/assets/111835409/3e80e7d9-580b-4d43-af1d-75c7f560cb22">

Pneumonia labeled image vs. normal image from the testing data subset



The Kermany et al. dataset includes a total of 5,856 chest X-ray images of different sizes, which have been pre-partitioned into three different subsets: training, validation, and testing. The training dataset includes 5,216 images, the validation dataset includes 16 images, and the testing dataset includes 624 images. Across these three subsets, we observed a diverse range of unique height x width combinations. Specifically, the training dataset had 4,366 unique height x width combinations, the validation dataset had 16 unique height x width combinations, and the testing dataset had 598 unique height x width combinations. 

Given the variability in image sizes within our dataset, we implemented a standardization process to enhance the efficiency of our data processing steps and to facilitate the normalization of inputs for our model. Specifically, we resized all the images to a uniform size of 150 x 150 pixels. This standardization step allowed us to streamline the data-handling process as well as to ensure consistency in the input dimensions across all images. Additionally, standardizing the size of our images helped us speed up and improve the efficiency of our model training and evaluation procedures. 

In addition to resizing the images, we applied a range of image transformations to enhance the performance of our model. This process, known as data augmentation, helps introduce additional variations in the data, effectively expanding the size and diversity of the training dataset. In the following section, we provide a detailed description of the specific transformations we applied to our dataset to augment the data and improve the overall performance of our deep learning model.

### Data Augmentation

After conducting an initial analysis of our dataset, we observed a significant class imbalance, with a larger number of images labeled as Pneumonia X-rays compared to normal X-rays. This imbalance is illustrated in Figure 2 below, which shows that in the training set there are 1,341 normal X-rays and 3,875 X-rays with pneumonia. Recognizing the potential impact of this data imbalance on our results and with the goal of enhancing the predictive quality of our model, we implemented a series of data augmentation techniques to mitigate the risk of overfitting. Taking inspiration from previous works on this dataset available on Kaggle, we adopted various transformations to augment the images. The specific augmentations we applied include rotating the image by 30 degrees, zooming into the image by 20%, horizontally flipping the image, enhancing the image's sharpness, and adjusting the color depth of the image. By employing these augmentation steps, we wanted to introduce additional diversity and variability into the dataset, allowing our model to learn from a wider range of variations and achieve better generalization performance.

<p align="center">
<img width="390" alt="Screen Shot 2023-09-19 at 8 06 26 PM" src="https://github.com/jimenasalinas/predicting_pneumonia/assets/111835409/d1c2db6b-03f4-43f4-bc5c-097491b365d1">
</p>

Figure 3 below provides an example of an image before and after the data augmentation process. Both images are displayed using a false color mapping called a "heat map." Heat maps use a color gradient to represent different intensity values in the tensor image. This color mapping can help to make it easier to interpret and visualize the image, as different intensity levels are notably different.

<p align="center">
<img width="657" alt="Screen Shot 2023-09-19 at 8 06 53 PM" src="https://github.com/jimenasalinas/predicting_pneumonia/assets/111835409/1668d338-eca1-4d2b-ab51-64c6d2cb0803">
</p>

### Model Architecture

We experimented with alternative models by adjusting the hyperparameters in each model. For ease of presentation, the following section describes the baseline model as well as the best of the alternative ones.

#### Baseline Model

Our baseline model was inspired by the LeNet-5 model developed by LeCun, Bengio, and Hinton, but we adapted the design to improve our model's results in the context of pneumonia prediction. We follow a similar model architecture, using the same number of convolutional layers, batch normalization, ReLU activation functions, and max pooling operations. Our model incorporates these elements with the purpose of extracting important features from input X-ray images. Similarly to LeNet-5, we also include fully connected layers for further processing and classification. 

The architecture of our baseline model consists of two convolutional layers, each followed by batch normalization, ReLU activation function, and a max pooling operation. The first layer takes a grayscale 150 x 150 pixels image and applies 6 convolutional filters, while the second layer takes the output of the first layer as input and applies 16 convolutional filters. Both convolutional layers are followed by a max pooling operation to reduce the dimensions of the outputs. The output from the convolutional layers is then flattened using the Flatten() operation.

After the convolutional layers, our model includes three fully connected layers. The linear layers consist of two hidden layers with ReLU activation functions and a final output layer with two binary labels (pneumonia vs. no pneumonia). The final layer performs another linear transformation to produce the output predictions.

Finally, we do a forward pass through the convolutional layers, followed by flattening the output and passing it through the fully connected layers. This final linear layer output represents the predicted class probabilities for the input image.

In the training process, we train our model using 20 epochs, and we fine-tune our hyperparameters using our validation data set. We use a CrossEntropy Loss function to compute the loss from our model, and we then use Stochastic Gradient Descent (SGD) to optimize our model, using a learning rate of 0.001 and momentum of 0.9. The SGD updates the model parameters based on the gradients of the loss function to minimize the loss during training. After applying a loss function and optimizing our model, we estimate the associated accuracy and F1 scores for performance evaluation. 

#### Alternative model
The architecture of our second model consists of three convolutional layers, each followed by batch normalization, ReLU activation function, and max pooling operations. As before, the first layer takes a grayscale 150 x 150 pixels image and applies 10 convolutional filters, while the second layer takes the output of the first layer as input and applies 16 convolutional filters. The third layer takes the output from the previous layer and applies 20 convolutional filters. After each layer, the max pooling operation helps to reduce the dimensions of the outputs, which are flattened after the convolutional layers and passed into two linear functions to get a fully connected layer with 64 neurons. Finally, we have the output layer which has 2 output neurons, since our images are represented as pneumonia or normal.

To make results comparable, we also train the model using 20 epochs. A major difference between both models is that we used an Adam optimizer for the alternate model to incorporate a regularization term that helps in reducing the overfitting of our baseline model. After these calculations, we see that our model is less complex as the number of parameters estimated is 304,742 instead of 1,190,714 from the baseline.

### Model Results
#### Accuracy metrics

To assess the prediction capacity of our model we compare the model's loss and classification power. The first one is defined as the accuracy of the model in terms of the predicted probabilities, while the second refers to how good the model classifies the actual labels (*e.g.*, normal lungs or lungs with pneumonia).

The loss function that we will use is the cross-entropy, which is defined as:

$$Loss = - \frac{1}{N} \sum_{i = 1}^{N} y_i \cdot log(\hat y_i)$$

where $y_i$ is the label for the $i$-th observation, and $\hat y_i$ is the prediction. Intuitively the loss penalizes incorrect predictions more severely, with the logarithmic term amplifying the error when the predicted probability deviates from the true label. The overall loss is computed by averaging this binary cross-entropy loss over the entire training set.

On the other hand, the accuracy measures we'll use are:

1. Accuracy: provides an estimate of how well the model predicts the correct class labels

2. F1 score: combines the precision and recall metrics into a single value, providing a balanced assessment of the model's performance. This metric is particularly important for our case given that we have an imbalanced dataset that contains more pneumonia images vs. normal images, and that pneumonia is a life-threatening disease that requires accurate diagnosis for prompt medical intervention. A high F1 score indicates that the model achieves a good balance between minimizing false negatives (avoid missing pneumonia cases) and false positives (maintaining high precision to reduce misdiagnoses). By optimizing the F1 score, we aim to achieve a model that accurately detects pneumonia while minimizing errors that could have significant clinical implications.

Both accuracy measures were calculated for the train, validation and test datasets to evaluate the model's performance and potential overfitting.

### Results

We begin by plotting each one of the performance metrics, accuracy and F1 score, for both the baseline and the alternative model. The number of epochs are shown in x-axis, while the average value for each metric is represented in the y-axis.

<p align="center">
<img width="669" alt="Screen Shot 2023-09-19 at 8 07 55 PM" src="https://github.com/jimenasalinas/predicting_pneumonia/assets/111835409/3777ff72-cc19-403b-b432-10b27bee430c">
</p>

<p align="center">
<img width="672" alt="Screen Shot 2023-09-19 at 8 07 59 PM" src="https://github.com/jimenasalinas/predicting_pneumonia/assets/111835409/35cfbda3-38ac-4e7a-a3f9-2f7bb5591398">
</p>

As we can see, it appears that, in terms of accuracy and F1 score, both models are pretty similar in their performance. In both cases we observe that the values for the train dataset are high, indicating that the model tends to overfit. Furthermore, after 2 epochs both models converge to approximately 95% for both metrics, indicating that there is not a lot of variation regardless of the epoch we analyze.

The training data set is unbalanced towards pneumonia, so it is learning to identify these images really well, but not so accurately when predicting them in unseen data. This can be explained by the test accuracy and F1 scores, which in both cases is around 82%, marginally better for the alternate model. Given these results, and the model’s architecture, we would select the second one as the best model because the adam optimizer we used includes a regularization term that helps with the overfitting problem. We would also prefer this model because the number of parameters are drastically reduced, while the marginal gain related to the testing metrics is small.

The main difference from both models comes from the validation dataset, where the baseline is more unstable than the second. This could indicate that, even though the metrics are quite similar for unseen data, the hyperparameter tuning helped in reducing the overfitting of the baseline model. As a future step, we could resplit our train, validation and test datasets into more even proportions, since our validation set consisted of only 16 images which caused a high imbalance reflected unreliability of our model. Re-splitting would not only lead to a larger validation set, but it could also help in increasing the number of images without pneumonia. Having a larger validation set, as well as a more balanced training set, could potentially increase our accuray and F1 scores, producing a more reliable model.


### Conclusion

In conclusion, we aimed to investigate the application of deep learning techniques in pneumonia prediction based on chest X-ray images. The literature review highlighted the significance of deep learning in various medical domains, including lung cancer detection, COVID-19 diagnosis, and dermatology. The studies by Van der Heijden et al. (2020) and Ophir Gozes et al. (2020) demonstrated the potential of deep learning in accurately identifying lung cancer and detecting COVID-19 infection using CT scans. However, we acknowledged limitations such as generalizability and dataset availability.

The paper presented two models for pneumonia prediction. The baseline model was inspired by LeNet-5, with adjustments to improve performance. The alternative model had a simpler architecture, reducing the number of parameters. The models were trained and evaluated using accuracy and F1 scores. The results showed that both models achieved similar performance, with high accuracy and F1 scores on the training dataset, indicating potential overfitting. However, the test accuracy and F1 scores were around 82%, slightly better for the alternative model. This suggests that, even though our model was able to predict pneumonia from the chest X-ray dataset, there are still limitations in predicting pneumonia accurately on unseen data.


