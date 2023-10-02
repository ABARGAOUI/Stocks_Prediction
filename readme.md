First I want to thank you for your time, consideration and letting me take part of this assessment.
I will try in this readme file to illustrate my solution to the python assessment:
Here I expose two types of regressions:
    -Linear regression used to predict the future price of the index based on a first predifined amount
        using the random walk principle the features I used consists of index performance lagged by a number of days(default to 3)
    -Logistic regression used to predict in which direction the index is going (decreasing or increasing)
        
for each type of regression a hit ratio is computed to evaluate the predictions to the real values.
In most of the cases we end up with a hit ratio above 50%.

You can use the test examples to execute the API.

I joined two screenshot: on of a logistic regression and one of a linear regression.
As we can see the logistic regression performs well in predicting market movement directions and linear regression 
performed poorly in predicting sharp movements. 

To resolve this issue we can use features that represent better the sharp movements 
as we did when we choose momentum values as feature. We could use technics as recursive feature elimination 

Given more time for the exercise we can consider more advanced machine learning algorithms as gradient boost or 
deep learning models (neural network.)

