## The performance of statistical machine learning methods in the classification and diagnosis of time-series heart sounds
### Tony Geglio

# 1. Introduction to the project
In my research, I am focused on audio processing for classification and spatial analysis. One way this could be extended to the health community is in the diagnosis of cardiovascular abnormalities through analyzing the heart sounds. Other applications include Lung and breathing sound for respiratory conditions, however this data analysis and exploration will focus on the proccessing and classification of raw cadiovascular audio inputs.
## Problems to Solve
Currently, patients with cardiovascular disease have tools for monitoring their heart, however early diagnosis of a heart problem is still difficult. I think through very inexpensive sensors connected to a human body and communicating data with a smartphone could provide continuous analysis to people with heart problems, and send alerts, rather than requiring the patient to do the measurement and then react.
## Implementation Plan and Data Description
Here, I explore statistical machine learning methods, and later, deep learning methods for classification of heart sounds. For data, currently I am looking at a past competition for classifying heart sounds [1]. This data has 2 challenges, the heart sound segmentation and the heart sound classification for 2 datasets with different classes. The challenge was to correctly classify the unlabeled parts of the dataset. I plan to just use the clipped labeled data and skip the segmentation challenge because it may be more of a digital signal processing problem.
## Data
Data was collected for a 2011 challenge proposed by Bentley et al. The challenge included 2 data sets: data set A)  with heart sounds from the general public via the iStethoscope Pro iPhone app; and, data set B) with heart sounds from a clinic trial in hospitals using the digital stethoscope DigiScope. Combined, there are a total of 585 samples, each being a short clip in .wav format. The class balance is heavily biased to “normal” for both data sets.

4 categories for Dataset A:
* Normal
* Murmur
* Extra Heart Sound
* Artifact

3 classes contained in Dataset B:
* Normal
* Murmur
* Extrasystole

#### Changes from original proposal
My proposal was to build a binary statistical ML classifier for: “normal” and “murmur” sounds and to train on dataset A and test on dataset B. 

The first modification was that I did not train on A and test on B, rather combined the recordings into one data set and randomly train-test-split. This is because I wanted more diversity in my classifier and to be more robust and generalizable.

The second modification was to explore a 5-class classification problem after combining the two data sets. After the combination, the data contained 5 unique classes:
* Normal
* Murmur
* Extra Heart Sound
* Artifact
* Extrasystole

# 2. Systematic Review with help from ChatGPT
### Chat GPT:
Cardiac sound signal classification is an important area of research in cardiology that aims to accurately identify abnormal heart sounds, which can be indicative of various cardiac disorders. Accurate classification of cardiac sound signals is essential for making timely and accurate diagnoses, and for developing effective treatment plans. However, current methods for analyzing and classifying cardiac sound signals have several limitations and challenges.

One potential solution to these challenges is the use of artificial intelligence (AI), machine learning (ML), and deep learning (DL) techniques, which have shown promise in improving the accuracy and reliability of sound analysis for cardiac sound signal classification.
The limitations and challenges of current methods for cardiac sound signal classification include the subjectivity of human interpretation, which can lead to variability in diagnoses, and the lack of standardization in terminology and classification schemes. In addition, traditional signal processing techniques may not be effective for capturing subtle differences in sound patterns that are indicative of cardiac disorders.
The purpose of this review is to evaluate the effectiveness of AI/ML/DL methods for cardiac sound signal classification, with a focus on the potential for these techniques to improve the accuracy and reliability of sound analysis. The review will include a systematic search and analysis of relevant studies, including a critical evaluation of the strengths and limitations of AI/ML/DL methods for cardiac sound signal classification. The ultimate goal of the review is to provide insights into the potential of these techniques for improving clinical outcomes and advancing our understanding of cardiac disorders.

Here are five publications on the current state of the art in cardiac sound signal classification published between 2020 and 2023:
1. "Deep learning-based detection of abnormal heart sounds using a time-frequency representation of phonocardiogram" (2020) by Li et al. This paper proposes a deep learning-based approach for detecting abnormal heart sounds using a time-frequency representation of phonocardiogram signals.
2. "Automatic classification of heart sounds using a convolutional neural network" (2020) by Badawy et al. This paper presents a convolutional neural network-based approach for classifying heart sounds, which achieved high accuracy on a publicly available dataset.
3. "A novel approach for automatic heart sound classification using deep convolutional neural network with transfer learning" (2021) by Singh et al. This paper presents a novel approach for heart sound classification using a deep convolutional neural network with transfer learning, which achieved high accuracy on a dataset of over 3,000 heart sound recordings.
4. "A deep learning approach for automated detection of mitral regurgitation from heart sound signals" (2021) by Vasconcelos et al. This paper proposes a deep learning-based approach for automated detection of mitral regurgitation from heart sound signals, which achieved high sensitivity and specificity on a dataset of over 1,000 heart sound recordings.
5. "A systematic review of automatic classification of heart sounds using machine learning techniques" (2022) by Pandit et al. This review article provides a comprehensive overview of the current state of the art in automatic classification of heart sounds using machine learning techniques, including a critical evaluation of the strengths and limitations of existing approaches and a discussion of future research directions.

### Not Chat GPT
The fifth citation generated by Chat GPT looks relevant, however I was unable to access the article. Another relevant systematic review is titled “Heart sound classification using signal processing and machine learning algorithms” (2022) by Zeinali et. al[2]. In this study, they explored extracting signal features such as amplitude, dominant frequencies, and the discrete wavelet transforms, and information theory features. The authors explored using the support vector machines classifier (SVC), gradient boosting classifier (GBC), and random forest classifier (RFC). The accuracy ranged from 75 - 87% with gradient boosting performing the best.

# 3. Method
One common method for statistical machine learning classification of audio data is explained in multiple speaker identification studies[3] [4]. Their method is to extract features, in this case, extracting the Mel Frequency Cepstral Coefficients (MFCC) from audio signals. Statistical classification can be done on the MFCC features by representing the signal features in a tabular format. I will test out bagging and boosting methods, such as Random Forest and Adaboost, as well as a Support Vector Machine and Multi-Layer Perceptron and compare their performance on the MFCC data. Later, I may explore deep learning on the raw signals because 1D-CNN on audio has shown good performance in other studies and requires less processing time.

We use a train-test-split of 75/25, gridsearchCV for parameter tuning, 5-fold cross-validation for resampling, and the weighted average F1-score is used to optimize the models.

# 4. Results

## Random Forests

Best Model:

RandomForestClassifier(max_depth=4, n_estimators=15, random_state=42)
{'criterion': 'gini', 'max_depth': 4, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 15}

accuracy:0.714
recall:0.714
Precision:0.716

Classification Report

|                 | precision | recall | f1-score | support |
|-----------------|-----------|--------|----------|---------|
| artifact        | 1.00      | 0.90   | 0.95     | 10      |
| extrahls        | 0.44      | 0.80   | 0.57     | 5       |
| extrastole      | 0.00      | 0.00   | 0.00     | 12      |
| murmur          | 1.00      | 0.25   | 0.40     | 32      |
| normal          | 0.69      | 0.95   | 0.80     | 88      |
|-----------------|-----------|--------|----------|---------|
| accuracy        |           |        | 0.71     | 147     |
| macro avg       | 0.63      | 0.58   | 0.54     | 147     |
| weighted avg    | 0.72      | 0.71   | 0.65     | 147     |

## Multi-Layer Perceptron

Best Model:

MLPClassifier(activation='logistic', alpha=0.1, hidden_layer_sizes=(40, 20), max_iter=500)
              
{'activation': 'logistic', 'alpha': 0.1, 'hidden_layer_sizes': (40, 20)}

accuracy:0.728
recall:0.728
precision:0.672

Classification Report
|                 | precision | recall | f1-score | support |
|-----------------|-----------|--------|----------|---------|
| artifact        | 1.00      | 0.90   | 0.95     | 10      |
| extrahls        | 0.50      | 1.00   | 0.67     | 5       |
| extrastole      | 0.00      | 0.00   | 0.00     | 12      |
| murmur          | 0.64      | 0.50   | 0.56     | 32      |
| normal          | 0.75      | 0.88   | 0.81     | 88      |
|-----------------|-----------|--------|----------|---------|
| accuracy        |           |        | 0.73     | 147     |
| macro avg       | 0.58      | 0.66   | 0.60     | 147     |
| weighted avg    | 0.67      | 0.73   | 0.69     | 147     |

## Support Vector Machine

SVC(C=0.1, gamma=0.01, kernel='linear', max_iter=10000, probability=True)

{'C': 0.1, 'gamma': 0.01, 'kernel': 'linear'}

accuracy:0.673
recall:0.673
precision:0.625

Classification Report
|                |precision  | recall | f1-score | support |
|----------------|-----------|--------|----------|---------|
| artifact       | 1.00      | 0.70   | 0.82     | 10      |
| extrahls       | 0.29      | 0.40   | 0.33     | 5       |
| extrastole     | 0.00      | 0.00   | 0.00     | 12      |
| murmur         | 0.64      | 0.28   | 0.39     | 32      |
| normal         | 0.68      | 0.92   | 0.78     | 88      |
|-----------------|-----------|--------|----------|---------|
| accuracy       |           |        | 0.67     | 147     |
| macro avg      | 0.52      | 0.46   | 0.47     | 147     |
| weighted avg   | 0.63      | 0.67   | 0.62     | 147     |

## Gradient Boosting Classifier

GradientBoostingClassifier(max_depth=1, n_estimators=50, random_state=42)

{'learning_rate': 0.1, 'max_depth': 1, 'n_estimators': 50}

accuracy:0.701
recall:0.701
precision:0.658
Classification Report

|              	| precision 	| recall 	| f1-score 	| support 	|
|--------------	|-----------	|--------	|----------	|---------	|
| artifact     	| 1.00      	| 0.70   	| 0.82     	| 10      	|
| extrahls     	| 0.57      	| 0.80   	| 0.67     	| 5       	|
| extrastole   	| 0.00      	| 0.00   	| 0.00     	| 12      	|
| murmur       	| 0.73      	| 0.25   	| 0.37     	| 32      	|
| normal       	| 0.69      	| 0.95   	| 0.80     	| 88      	|
|-----------------|-----------|--------|----------|---------|
| accuracy     	|           	|        	| 0.70     	| 147     	|
| macro avg    	| 0.60      	| 0.54   	| 0.53     	| 147     	|
| weighted avg 	| 0.66      	| 0.70   	| 0.64     	| 147     	|



# Works Cited


[1]	“Classifying Heart Sounds Challenge.” http://www.peterjbentley.com/heartchallenge/ (accessed Feb. 12, 2023).

[2]	Y. Zeinali and S. T. A. Niaki, “Heart sound classification using signal processing and machine learning algorithms,” Mach. Learn. Appl., vol. 7, p. 100206, Mar. 2022, doi: 10.1016/j.mlwa.2021.100206.

[3]	S. Nakagawa, L. Wang, and S. Ohtsuka, “Speaker Identification and Verification by Combining MFCC and Phase Information,” IEEE Trans. Audio Speech Lang. Process., vol. 20, no. 4, pp. 1085–1095, May 2012, doi: 10.1109/TASL.2011.2172422.

[4]	M. Hasan, M. Jamil, G. Rabbani, and Md. S. Rahman, “Speaker Identification Using Mel Frequency Cepstral Coefficients,” Proc. 3rd Int. Conf. Electr. Comput. Eng. ICECE 2004, Dec. 2004.






