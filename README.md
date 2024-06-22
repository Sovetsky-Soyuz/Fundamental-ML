# Mini-Project for Fundamentals of Machine Learning Course
![background](./materials/ai_wp.jpg)
This repository contains the code and data for a mini-project on facial expression recognition using machine learning algorithms.

## 📑 Project Policy
- Team: group should consist of 3-4 students.

    |No.| Student Name    	 | 	 Student ID |
    |---|------------------------|----------------- |
    |1  |Trần Nguyễn Minh Quang  	   21110160 |
    |2  |Trần Minh Huân 		   21110090 |
    |3  |Nguyễn Hoàng Hải 		   21110285 |
    |4  |Nguyễn Đức Cường 		   21110049 |

- The submission deadline is strict: **11:59 PM** on **June 22nd, 2024**. Commits pushed after this deadline will not be considered.

## 📦 Project Structure

The repository is organized into the following directories:

- **/data**: This directory contains the facial expression dataset. You'll need to download the dataset and place it here before running the notebooks. (Download link provided below)
- **/notebooks**: This directory contains the Jupyter notebook ```EDA.ipynb```. This notebook guides you through exploratory data analysis (EDA) and classification tasks.

## ⚙️ Usage

This project is designed to be completed in the following steps:

1. **Fork the Project**: Click on the ```Fork``` button on the top right corner of this repository, this will create a copy of the repository in your own GitHub account. Complete the table at the top by entering your team member names.

2. **Download the Dataset**: Download the facial expression dataset from the following [link](https://mega.nz/file/foM2wDaa#GPGyspdUB2WV-fATL-ZvYj3i4FqgbVKyct413gxg3rE) and place it in the **/data** directory:

3. **Complete the Tasks**: Open the ```notebooks/EDA.ipynb``` notebook in your Jupyter Notebook environment. The notebook is designed to guide you through various tasks, including:
    
    1. Prerequisite
    2. Principle Component Analysis
	- We can't visualize the data projected onto two principal component. This mean that we cannot effectively visualize the data, the visual 	representation fails to provide a meaningful separation or classification of the data points into their respective categories or classes.

	- One way to determine K is based on the amount of information to be retained. As mentioned, PCA is also known as the method of maximizing the total 	retained variance. So, we can consider the total retained variances as the amount of retained information. The larger the variance, the higher the 	data dispersion, indicating a larger amount of information. (read more in EDA.ipynb (explain))


    3. Image Classification
       - In this section, we got a problem with GridsearchCV. Therefore, we tried to implement RandomizedSearchCV and BayesSearchCV instead. because the 	dataset in this project is too large for free colab version, so we spent a lot of time for training model.
	 
	GridSearchCV and RandomizedSearchCV can take a long time to run on some models in our project due to the following reasons:

	- GridSearchCV, RandomizedSearchCV, and BayesSearchCV are hyperparameter tuning methods in machine learning, each with distinct characteristics and 	suitable use cases based on the dataset and computational resources available.

	- GridSearchCV performs an exhaustive search over a specified parameter grid. this method evaluates all possible combinations of hyperparameters and 	ensures all combinations are tested, guaranteeing the best set within the specified grid.

	- However, GridSearchCV Becomes impractical for large datasets or extensive hyperparameter grids Time-consuming, especially when the model training 	time is high. It is only suitable for Small to medium-sized datasets, scenarios where computational resources are not a limiting factor, problems 	where an exhaustive search is feasible and desired.


	- RandomizedSearchCV performs random sampling of hyperparameters from a specified distribution for a fixed number of iterations.Does not evaluate 	all possible combinations but samples a subset of them, so it more efficient than GridSearchCV can cover a larger hyperparameter space in fewer 	evaluations and Reduces computation time by limiting the number of iterations.

	- RandomizedSearchCV is optimal for large datasets where GridSearchCV would be too computationally expensive. In case, where a reasonably good 	solution is acceptable with less computational cost. Problems with a large number of hyperparameters.


	- Uses Bayesian optimization to model the performance of the hyperparameters and choose the next set of hyperparameters to evaluate based on past 	evaluations. It is efficiently narrows down the search space over iterations.

	- More sample-efficient than GridSearchCV and RandomizedSearchCV. BayesSearchCV Can find better hyperparameter sets with fewer evaluations by 	learning from previous iterations and be suitable for complex search spaces.

	- BayesSearchCV large and complex datasets where computational efficiency is critical. in scenarios, where a high-performing model is needed, and 	there is limited time for hyperparameter tuning, but it gonna get problems with high-dimensional hyperparameter spaces.

	***Summary
	- GridSearchCV is optimal for smaller datasets and when computational resources are abundant.
	- RandomizedSearchCV is optimal for larger datasets or when a good enough solution is needed quickly without evaluating all possibilities.
	- BayesSearchCV is optimal for large, complex datasets and when seeking the best possible model with efficient use of computational resources.
	
	-> Hence we used BayesSearchCV for some model in this project to save time for training model. Because when we use GridSearchCV, it might take a lot 	of times (over 7 hours with n_iter = 10; default kernel or leave it to parameters, with default CPU in colab) and RandomizedSearchCV (over 3 hours 	with n_iter=10; default kernel or leave it to param_distributions, with default CPU in colab) but can not done the task. So we will use Bayesian 	Optimization (BayesSearchCV, with n_iter=3, default kernel) instead.


    4. Evaluating Classification Performance
	- Read in EDA.ipynb

    5. Bonus model (EDA_2.ipynb)
	- We built another deep learning model (this is only extra model for researching more about deep learning).in this model following:
	
	- Instead of training with the original dataset, we balance the data to ensure there is no significant disparity between classes. We used 	RandomOverSampler method for oversampling the dataset.
	
	- The RandomOverSampler is a technique used in the context of imbalanced datasets to oversample the minority class, making the dataset more 	balanced. This can help improve the performance of machine learning models by preventing them from being biased towards the majority class.
	
	- After balancing, we created more diverse data through Data Augmentation. Data augmentation is a technique used to artificially increase the size 	and diversity of a dataset without actually collecting new data. By applying various transformations to existing data, models can generalize better 	and become more robust.
	
	-> This helps the model learn better and avoids overfitting to certain classes.

	About model: MLP in this file
	
	- We use MLP with PyTorch and ELU (Exponential Linear Unit) function, ELU is an activation function that helps neural networks learn complex patterns.
	
	This method take some advantages and disadvantages:
	- Advantages
	• High Representation Capability: Can learn and represent more complex features due to deeper hidden layers and non-linear activation functions (in this case we use 1 input layer, 4 hidden layers, 1 output layer and ELU activation function to give model learn more complex features).
	• Flexible Optimization: With PyTorch, we can customize the model in detail, experimenting with different architectures and activation functions.
	• Accuracy: Often achieves higher accuracy on complex tasks thanks to its deep learning capabilities.

	- Disadvantages:
	• Requires More Computational Resources: Training deep learning models usually requires GPUs and takes longer time.
	• Complex Setup: Requires more in-depth knowledge of network architecture and optimization techniques.
	• Overfitting: Without sufficient data or good regularization techniques, the model can easily overfit.


    Make sure to run all the code cells in the ```EDA.ipynb``` notebook and ensure they produce output before committing and pushing your changes.

5. **Commit and Push Your Changes**: Once you've completed the tasks outlined in the notebook, commit your changes to your local repository and push them to your forked repository on GitHub.


Feel free to modify and extend the notebook to explore further aspects of the data and experiment with different algorithms. Good luck.