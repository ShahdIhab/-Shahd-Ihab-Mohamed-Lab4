# -Shahd-Ihab-Mohamed-Lab4
## Lab 4: Bayesian Decision Surfaces

The project is Bayesian Decision Surfaces

![Capture](https://user-images.githubusercontent.com/92639654/216793802-d9fe6aee-35b0-45bb-b8b1-3e77963f8934.PNG)


## Naive Bayes classifiers 
Naive Bayes classifiers are a collection of classification algorithms based on Bayes’ Theorem. It is not a single algorithm but a family of algorithms where all of them share a common principle, i.e. every pair of features being classified is independent of each other.

## Libraries that is used
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import seaborn as sns

To start with, let us consider a dataset.

### Text data set
df = pd.read_csv('binclass.txt') 
df.columns = ["positive", "negative", "y"] 


[binclass.txt](https://github.com/ShahdIhab/-Shahd-Ihab-Mohamed-Lab4/files/10609927/binclass.txt)


[binclassv2.txt](https://github.com/ShahdIhab/-Shahd-Ihab-Mohamed-Lab4/files/10609928/binclassv2.txt)

## Data test
#### Binclass
![2](https://user-images.githubusercontent.com/92639654/216794798-a22cee3f-e608-44d0-a700-e7889d732e8d.PNG)

#### binclassv2
![image](https://user-images.githubusercontent.com/92639654/216795214-41e8f582-5199-43a1-ac6c-d14f5fba1e6c.png)


## Scatter Plot
#### Binclass
Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
xlabel='positive', ylabel='negative'
![Capture](https://user-images.githubusercontent.com/92639654/216795026-80d2cef0-ad16-4c2f-81dd-a63285594c3d.PNG)

#### binclassv2
![Capture](https://user-images.githubusercontent.com/92639654/216795255-d9944701-4fac-417d-a50f-2f471adf0d44.PNG)


## Accuracy 
The  Naive Bayes Classifier Accuracy Score is 0.9523809523809523
![image](https://user-images.githubusercontent.com/92639654/216795148-3eef889e-f635-4f18-9aec-145bb93de821.png)








