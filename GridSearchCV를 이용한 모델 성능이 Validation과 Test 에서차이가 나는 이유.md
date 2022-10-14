## 시작하며
**질이 좋지 않은 Dataset**의 모델링을 하기 위해 Grid Search를 사용하여 Hyper Parameter Tuning을 했다.    
여느 때와 같이 가장 높은 Validation Score를 보이는 hyperparameter 조합을 사용해 최종 모델을 생성하여    
train_test_split을 이용해 사전에 분리해뒀던 Test set에 적용해 보았는데,    
```
validation R2 score     0.48  
test R2 score           0.29  
```

validation과 test 모두 모델에 누설되지 않은 데이터인데 왜 모델 결정력에서 차이가 나는 것인가에 대한 의문이 들었다.
(validation 또한 test set과 비슷한 성향을 띄니 성능 또한 비슷해야 하는 것이 아닌가? 하는..)


본격 시작!


## 탐구

Scikit learn의 GridSearchCV 에는 Parameter 최적화와 더불어 Cross validation을 할 수 있는 기능이 내장되어 있다.
**Cross validation**이란
데이터셋을 여러 split(fold)로 나누어 학습에 사용하여 모델이 다양한 train set 을 경험할 수 있도록 하는 기법이다. 

![cv](https://user-images.githubusercontent.com/111988648/195750082-89008197-6c59-491a-b145-949bd1b74e9c.png)
CV기법에 따라 자세한 split 방식에는 차이가 있지만, 일반적으로는 이렇다고 한다.

고정된 train set과 test set으로 학습과 평가를 하며 반복적으로 모델을 튜닝하다보면 모델이 과적합되어버리는 결과가 생긴다. 
CV를 사용하여 데이터를 train/validation set로 여러 번 나눈 것의 평균적인 성능을 계산하면, 한 번 나누어서 학습하는 것에 비해 일반화된 성능을 얻을 수 있다.  



나 또한, 이 모델의 성능을 효과적으로 확인하기 위해 GridSearchCV에 CV 옵션을 사용하여 Hyperparameter tuning을 진행했다.



### GridSearchCV

사용한 CV 방식은 아래와 같다.  
**ShuffleSplit**  
Split 수: 10  
train size : test size = 0.8 : 0.2   
model: XGBoost  

```python
def fit_xgb_model(X, y):
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
    cv_sets.get_n_splits(X)

    regressor = XGBRegressor(random_state=0)

    params = {'n_estimators':range(10, 200, 10), 'learning_rate': [lr * 0.1 for lr in range(1, 6)]}

    scoring_fnc = make_scorer(performance_metric)

    grid = GridSearchCV(regressor, params, scoring = scoring_fnc, cv=cv_sets)    
    grid = grid.fit(X, y)

    return grid
``` 

```python
xgb_model = fit_xgb_model(x_train, y_train)

print("Parameter 'lr' is {}, 'n_estimators' is {} for the optimal model.".format(xgb_model.best_estimator_.get_params()['learning_rate'], xgb_model.best_estimator_.get_params()['n_estimators']))
print("Best Score is {:.2f}".format(xgb_model.best_score_))
```

output
```
Parameter 'lr' is 0.2, 'n_estimators' is 190 for the optimal model.
Best Score is 0.48
```

GridSearchCV의 결과를 확인해 보면, 
XGBoost  
Learning rate: 0.2  
n_estimators: 190 일 때 가장 높은 **validation R2 score 0.48** 을 기록했다고 한다.  

데이터 수가 적고, 데이터셋 또한 깨끗하지 않았기 때문에 이 정도면 나름 준수한 성능이라 생각했고,  
나는 싱글벙글 웃으며 최적화된 hyper parameter를 사용하여 Test set을 예측하여 성능을 확인해 보았다.  


```python
from sklearn.metrics import r2_score

r2_score(xgb_model.best_estimator_.predict(x_test), y_test)
```
```
0.29508569327215195
```

test 성능과 validation 성능이 큰 차이를 보인다.  
train 성능과 test 성능이 각각 0.5 0.3이었다면 고개를 끄덕였을지도 모른다. 근데 이건 둘 다 누설되지 않은 새로운 데이터이기 때문에 비슷한 성능을 보여야 하는 것이 아닌가??

그 이유에 대해 생각해 보았다.
가장 먼저 떠오른 것은 **과적합**이라는 단어였다.
GridSearchCV에서 데이터가 과적합되어 Test set에서 그만한 성능을 보이지 못하는 게 아닐까? 하는 의문이 첫번째였다.
근데 이건 
