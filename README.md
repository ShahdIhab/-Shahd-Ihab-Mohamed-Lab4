# -Shahd-Ihab-Mohamed-Lab4
## Lab 4: Bayesian Decision Surfaces

The project is Bayesian Decision Surfaces

![Capture](https://user-images.githubusercontent.com/92639654/216793802-d9fe6aee-35b0-45bb-b8b1-3e77963f8934.PNG)

## steps
   1- read file : read txt file and calculate the some numerical values needed in the processing.
    2- gaussian estimation: calculate the sigma and mus for each class.
    3- distribution_calculation : return the boundary of each class on a scatter plot.
    5- prob_pos_cond_: calculate the conditional prob.
    6- boundary : draw the scatter plot with the decision boundary between each class.
    7- evaluation: calculate the accuracy of the classifier.
    

## Libraries that is used
#Imports Libraries

 matplotlib.pyplot 

 numpy 

scipy.stats

pandas 

sklearn seaborn 

To start with, let us consider a dataset.

### Text data set
df = pd.read_csv('binclass.txt') 

![9](https://user-images.githubusercontent.com/92639654/216848896-08f08f7d-e5c2-4c9b-b934-c6bb06596cdc.PNG)

![8](https://user-images.githubusercontent.com/92639654/216848888-2ea09268-721e-45a7-bc2e-1ab56e656a2a.PNG)




## Scatter Plot
#### Binclass
![download](https://user-images.githubusercontent.com/92639654/216848919-a35b40cd-9f6a-45f0-8dc5-cb9e38b1c562.png)

![2](https://user-images.githubusercontent.com/92639654/216848927-4d4c931f-1800-4e90-af86-abb0ca9bcd89.png)

![3](https://user-images.githubusercontent.com/92639654/216848941-610ddbea-2737-4d00-9bbd-8198aeea14d4.png)


#### binclassv2
![4](https://user-images.githubusercontent.com/92639654/216848949-a658a62a-7bc0-460a-9a05-a7c64e36b295.png)
![5](https://user-images.githubusercontent.com/92639654/216848958-af4da458-2625-483c-9fec-eb7284830659.png)
![6](https://user-images.githubusercontent.com/92639654/216848961-d8943168-6c5f-491a-8ec0-0296d9e1fe4f.png)






