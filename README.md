# My First Project in Machine Learning Basics Nanodegree
# Supervised Learning
## Project: Finding Donors for CharityML

### Table Of Contents:
- [Description](#description)<br>
    -[hey]()
- [Data](#data)<br>
- [Let's Begin](#lets-begin)<br>
- [Conclusion](#conclusion)<br>

----

### Description

#### About CharityML & Project

CharityML is a fictitious charity organization located in the heart of Silicon Valley that was established to provide financial support for people eager to learn machine learning. After nearly 32,000 letters sent to people in the community, CharityML determined that every donation they received came from someone that was making more than $50,000 annually. To expand their potential donor base, CharityML has decided to send letters to residents of California, but to only those most likely to donate to the charity. With nearly 15 million working Californians, CharityML has brought us on board to help build an algorithm to best identify potential donors and reduce overhead cost of sending mail. Our goal will be evaluate and optimize several different supervised learners to determine which algorithm will provide the highest donation yield while also reducing the total number of letters being sent.

#### What needs to be done?

In this project, we will apply supervised learning techniques and an analytical mind on data collected for the U.S. census to help CharityML (a fictitious charity organization) identify people most likely to donate to their cause. We will first explore the data to learn how the census data is recorded. Next, we will apply a series of transformations and preprocessing techniques to manipulate the data into a workable format. We will then evaluate several supervised learners of our choice on the data, and consider which is best suited for the solution. Afterwards, We will optimize the model that we have selected and present it as our solution to CharityML. Finally, We will explore the chosen model and its predictions under the hood, to see just how well it's performing when considering the data it's given.

#### Why this Project?

This project is designed to get us acquainted with the many supervised learning algorithms available in sklearn, and to also provide for a method of evaluating just how each model works and performs on a certain type of data. It is important in machine learning to understand exactly when and where a certain algorithm should be used, and when one should be avoided.

-----

### Data

#### Files

Template code is provided in the `Finding_Donors.ipynb` notebook file. We will also be required to use the included `visuals.py` Python file and the `census.csv` dataset file to complete this work. While some code has already been implemented to get us started, We will need to implement additional functionality when requested to successfully complete the project. 

> **Note: The code included in `visuals.py` is meant to be used out-of-the-box and not intended to be manipulate.** If you are interested in how the visualizations are created in the notebook, please feel free to explore this Python file.

#### Dataset file

The modified census dataset consists of approximately 32,000 data points, with each datapoint having 13 features. This dataset is a modified version of the dataset published in the paper *"Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",* by Ron Kohavi. You may find this paper [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf), with the original dataset hosted on [UCI](https://archive.ics.uci.edu/ml/datasets/Census+Income).

**Features**
- `age`: Age
- `workclass`: Working Class (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
- `education_level`: Level of Education (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool)
- `education-num`: Number of educational years completed
- `marital-status`: Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)
- `occupation`: Work Occupation (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)
- `relationship`: Relationship Status (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)
- `race`: Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
- `sex`: Sex (Female, Male)
- `capital-gain`: Monetary Capital Gains
- `capital-loss`: Monetary Capital Losses
- `hours-per-week`: Average Hours Per Week Worked
- `native-country`: Native Country (United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands)

 **Target Variable**
- `income`: Income Class (<=50K, >50K)

-----

### Lets Begin

#### Requirements

This project requires **Python 3.6.4** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute a [Jupyter Notebook](https://jupyter.org/install.html)

I recommend to install [Anaconda](https://www.anaconda.com/download/), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 


#### Execution

In a terminal or command window, navigate to the top-level project directory `Finding-Donors-CharityML/` (that contains this README) and run one of the following commands:

```bash
ipython notebook Finding_Donors.ipynb
```  
or
```bash
jupyter notebook Finding_Donors.ipynb
```

This will open the Jupyter/iPython Notebook software and project file in your browser.

-----

### Conclusion

#### Things we'll learn by completing this project:

- How to identify when preprocessing is needed, and how to apply it.
- How to establish a benchmark for a solution to the problem.
- What each of several supervised learning algorithms accomplishes given a specific dataset.
- How to investigate whether a candidate solution model is adequate for the problem.

#### Evaluation

This project was reviewed by a Udacity reviewer against the [Finding Donors for CharityML project rubric](https://review.udacity.com/#!/rubrics/406/view). All criteria found in the rubric must be *meeting specifications* for me to pass.

[My Project Review by an Udacity Reviewer](https://review.udacity.com/#!/reviews/1224180)

-----