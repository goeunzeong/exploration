## 시작하며
**질이 좋지 않은 Dataset**의 모델링을 하기 위해 Grid Search를 사용하여 Hyper Parameter Tuning을 했다.    
여느 때와 같이 가장 높은 Validation Score를 보이는 hyperparameter 조합을 사용해 최종 모델을 생성하여    
train_test_split을 이용해 사전에 분리해뒀던 Test set에 적용해 보았는데,    
<br/> 
```
validation R2 score     0.48  
test R2 score           0.29  
```
<br/> 
validation과 test 모두 모델에 누설되지 않은 데이터인데 왜 모델 결정력에서 차이가 나는 것인가에 대한 의문이 들었다.
(validation 또한 test set과 비슷한 성향을 띄니 성능 또한 비슷해야 하는 것이 아닌가? 하는..)
<br/> <br/> <br/> 

본격 탐구 시작!
  
<br/> <br/>   
## 데이터셋
<br/> 
사용한 것은 파장 별 흡광도로, 스펙트럼을 나타내는 데이터이다. 목표변수 Y는 특정한 정수값. (자세히 말할 수 없는 것에 심심한 사과를..)  
<br/> 
![spectrum](https://user-images.githubusercontent.com/111988648/195767602-2a710187-5d0d-4a1b-9beb-d48d927c59a7.png)
(https://www.edinst.com/blog/what-are-absorption-excitation-and-emission-spectra/)  
시각화를 한다면 이런 형태로 나올 것이다.
<br/> 
질이 좋지 않다고 이야기한 이유는, 
```python
df.iloc[:, 0:58]
```
![image](https://user-images.githubusercontent.com/111988648/195768081-908d9c8a-d267-4e44-9521-40a548eee4f3.png)
<br/> 
저파장의 영역에선 모든 데이터의 값이 0을 띄는 희소한 데이터이며  
특성이 1024개로 매우 많았고  
데이터 수는 66개로 매우 적었기 때문이다.  
<br/> 
물론, 이런 점들을 모두 보완하는 모델을 만드는 것이 나의 역할이지만, 이 글의 목적인 `Validation 성능과 Test 성능이 차이가 나는 이유 탐구` 와는 조금 거리가 있기 때문에 전처리하고 특성공학을 적용하고 모델의 성능을 높이려는 노력은 잠시 뒤로 제쳐두겠다!!!  
<br/> <br/> 
## 탐구
<br/> 
Scikit learn의 GridSearchCV 에는 Parameter 최적화와 더불어 Cross validation을 할 수 있는 기능이 내장되어 있다.  
<br/> <br/> 
**Cross validation**이란  
데이터셋을 여러 split(fold)로 나누어 학습에 사용하여 모델이 다양한 train set 을 경험할 수 있도록 하는 기법이다.   
<br/> 
![cv](https://user-images.githubusercontent.com/111988648/195750082-89008197-6c59-491a-b145-949bd1b74e9c.png)  
CV기법에 따라 자세한 split 방식에는 차이가 있지만, 일반적으로는 이렇다고 한다.  
<br/> 
고정된 train set과 test set으로 학습과 평가를 하며 반복적으로 모델을 튜닝하다보면 모델이 과적합되어버리는 결과가 생긴다.  
CV를 사용하여 데이터를 train/validation set로 여러 번 나눈 것의 평균적인 성능을 계산하면, 한 번 나누어서 학습하는 것에 비해 일반화된 성능을 얻을 수 있다.  
<br/> <br/> 


나 또한, 이 모델의 성능을 효과적으로 확인하기 위해 GridSearchCV에 CV 옵션을 사용하여 Hyperparameter tuning을 진행했다.  
<br/> <br/> <br/> 


### GridSearchCV
<br/> 
사용한 CV 방식은 아래와 같다.   
<br/> 
```
**ShuffleSplit**  
Split 수: 10  
train size : test size = 0.8 : 0.2   
model: XGBoost  
```
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
<br/> 
output
```
Parameter 'lr' is 0.2, 'n_estimators' is 190 for the optimal model.
Best Score is 0.48
```
<br/> 
GridSearchCV의 결과를 확인해 보면, 
<br/> 
XGBoost  
Learning rate: 0.2  
n_estimators: 190 일 때 가장 높은 **validation R2 score 0.48** 을 기록했다고 한다.  
<br/> 
데이터 수가 적고, 데이터셋 또한 깨끗하지 않았기 때문에 이 정도면 나름 준수한 성능이라 생각했고,  
나는 싱글벙글 웃으며 최적화된 hyper parameter를 사용하여 Test set을 예측하여 성능을 확인해 보았다.  
<br/> <br/> 

```python
from sklearn.metrics import r2_score

r2_score(xgb_model.best_estimator_.predict(x_test), y_test)
```
```
0.29508569327215195
```
<br/> 
test 성능과 validation 성능이 큰 차이를 보인다.  
train 성능과 test 성능이 각각 0.5 0.3이었다면 고개를 끄덕였을지도 모른다. 근데 이건 둘 다 누설되지 않은 새로운 데이터이기 때문에 비슷한 성능을 보여야 하는 것이 아닌가??  

<br/> <br/> 
가장 먼저 떠오른 것은 **과적합**이라는 단어였다.  
GridSearchCV에서 데이터가 과적합되어 Test set에서 그만한 성능을 보이지 못하는 게 아닐까? 하는 의문이 첫 번째였다.  
단순히 test 성능과 val 성능이 차이가 난다는 점에서 기인한 생각이었다.  
그렇지만 그건 아니라는 걸 깨닫고,, GridSearch가 아니라 CV에 대해 더 자세히 공부해보기 시작했다.  
<br/> 
그 결과, 왜 나의 XGBoost 모델의 Train 성능과 val 성능에 차이가 나는지 조금이나마 깨달을 수 있었다  
<br/> <br/> 
### 분석 결과
<br/> 
GridSearchCV를 통한 Cross Validation의 결과는 `.cv_results_` 메서드를 사용해 확인할 수 있다.  
<br/> 
```python
score = pd.Dataframe(xgb_model.cv_results_)
score[['rank_test_score', 'params', 'mean_test_score',
       'split0_test_score', 'split1_test_score', 'split2_test_score',
       'split3_test_score', 'split4_test_score', 'split5_test_score',
       'split6_test_score', 'split7_test_score', 'split8_test_score',
       'split9_test_score']].sort_values(by = 'mean_test_score', ascending=False)
```
<br/> 
![scoredataframe](https://user-images.githubusercontent.com/111988648/195765220-de422d8c-10bb-45a1-acbf-20352577eca7.png)
<br/> 

`n_split = 10`으로 설정했기 때문에, 각각의 hyper parameter 조합마다 10개의 split을 생성하고 평가하며 그 값들의 평균치로 전체적인 CV를 통한 성능을 도출할 것이다.  
그런데, 내 CV의 결과를 보면, 같은 파라미터 조합임에도 몇 번째 split이냐에 따라 test score가 `-0.7 ~ 0.8` 값으로 너무너무 천차만별임을 확인할 수 있다.  
<br/> <br/> 
그렇다.   
각 split에 어떠한 데이터가 들어가느냐에 따라 val score가 영향을 받는 것이다.   
무작위로 나누어진 split의 val set과 train set이 비슷한 양상을 띄면 (즉, val set이 train set을 잘 대표하면) val score 또한 높게 나오는 것이고, 그렇지 않으면 낮게 나오는 것이다.  
원본 데이터셋 자체가 좋지 않고 데이터 수 또한 적은데, split까지 여러 개로 나누다 보니 각 split의 데이터가 중구난방으로 일관성있지 않게 설정된 것이 문제인 것 같다.  
<br/> 
그래서 선택한 parameter 조합이 잘 설명하는 데이터와 test set으로 사전에 나누어 둔 데이터가 달랐기 때문에 이러한 사태가 발생하지 않았나 싶다.  
<br/> <br/> 
이 모든 건
```
1. 데이터의 질 영향이 크지 않았나 싶다.
```
<br/> <br/> 
## 정리!
Q. GridSearchCV를 이용한 모델 성능이 Validation과 Test에서 차이가 나는 이유가 무엇인가요??
A. 양이 적고 좋지 않은 Dataset에 CV의 성질이 적용되었기 때문입니다! 데이터를 더 많이 쌓아 데이터 간 적당한 규칙이 존재할 수 있도록 만들거나, 데이터를 잘 정제하는 과정을 거친다면 자연히 validation score 와 test score가 비슷해질 것이라 생각한다.  
<br/> <br/> 
## 고찰
어찌 보면 당연한 말이다. ...   
데이터를 분배해서 검증하는 것을 통해 전체적으로 잘 일반화된 모델을 찾을 수 있도록 하는 것이 gridsearch 와 Cross validation의 본질이니까.  
기존에 잘 알려진 Titanic, Boston Housing, Iris과 같은 질 좋은 Dataset을 사용하면 gridSearchCV는 물론 잘 돌아가고 test score까지 괜찮게 나왔고,
그런 데이터에 익숙해진 탓에 CV의 기본 개념을 잊고 데이터만 보면 코드부터 일단 냅다 들이밀 게 된 것 같다.  
실제 데이터는 그런 이상적인 형태가 아닌데...  ㅠㅠ
너무나 당연한 문제를 갖고 고민했던 게 조금은 부끄러우면서도 개념을 다시 정리할 수 있던 계기가 되어주어 감사하다.  



