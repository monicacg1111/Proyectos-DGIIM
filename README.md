# DGIIM Projects

Repository where I will be uploading some projects I have completed during my bachelor's. These projects are related to Machine Learning and Business Intelligence.

## Lazy Bayesian Rule Learning

This project focuses on developing a Lazy Bayesian Rule classifier to address **attribute interdependence** in Naive-Bayes models. This involves incorporating Bayesian rules that are constructed locally upon receiving test instances.

The project involved implementing the **LBR algorithm** (*Lazy Bayesian Rules*) in Python, as studied in this article: [https://link.springer.com/article/10.1023/A:1007613203719](https://link.springer.com/article/10.1023/A:1007613203719)

This approach combines two techniques: NBTREE and LAZYDT, aiming to avoid the disadvantages of both and enhance their strengths.

To evaluate the performance of the implemented algorithm, several datasets were used, including Iris and Wine. Additionally, LBR's metrics were compared with those of other algorithms, such as Naive Bayes and Random Forest. The results were more than satisfactory, with **metrics above 85%**.

This has been a very interesting project, as it allowed us to study an algorithm proposed by other researchers and compare its effectiveness with conventional Machine Learning algorithms.



## House Price Regression (Kaggle)

This project was carried out as part of the *Business Intelligence* course and involved participating in a [Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/). The objective of the competition was to **predict the price of a house** given a series of features, such as size, type of street, number of rooms, etc. Given a training dataset, the task was to train a **regressor** capable of accurately predicting the prices of houses in a new test dataset.

To tackle the problem, I first applied various **preprocessing** techniques extensively, including outlier removal, feature selection, missing value imputation, and encoding. Then, I used several **supervised learning algorithms**, including Decision Trees, Boosting, and Ensemble methods. Additionally, I performed **automatic tuning** to optimize the parameters of the chosen method to achieve the best possible accuracy.

As for the final results, the algorithm that performed best was Catboost, placing me in a relatively good position on the leaderboard for my first competition. Overall, it was a highly enriching and innovative experience.
![leaderboard2](https://github.com/monicacg1111/Proyectos-DGIIM/assets/55974676/ed5bd060-5df3-4d80-98ad-a7756888b2a8)



## Clustering CIS

This project was completed for the *Business Intelligence* course to draw conclusions about public opinion regarding the conflict between Israel and Palestine. It is based on a survey conducted by the **CIS** (Spanish Sociological Research Center) in November 2023, which collected the voting intentions of the Spanish population and their opinions on various current issues (Israel-Palestine conflict, inflation crisis, climate change, migration, etc.). The survey can be consulted [here](https://elpais.com/espana/2023-11-06/consulte-todos-los-datos-internos-de-la-encuesta-de-el-pais-cuestionarios-cruces-y-respuestas-individuales.html).

In addition to applying several **clustering algorithms** (such as K-means, BIRCH, Mean Shift, and DBSCAN), I performed preprocessing using normalization and imputation techniques.

The project involved establishing several case studies where I selected various features (survey responses) and examined the relationships between them. Among these case studies, I analyzed the relationship between people's **ideology**, their **voting intentions**, and their **opinions** on various actions taken by both Israel and Hamas. To interpret the algorithm results, I used various **visualization** tools, including Heatmaps, Scatter Matrix, MDS, etc.




