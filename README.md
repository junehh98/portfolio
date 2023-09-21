# portfolio


## 프로젝트 개요

***1. ESG_folium***

**프로젝트 목표:** 이 프로젝트는 서울시 구별 미세먼지 데이터를 시각화하는 것을 목표로 합니다.

**프로젝트 설명:** 

1. **데이터 수집**: 2016년부터 2021년까지의 서울 대기질 데이터를 두 개의 Excel 파일에서 읽어옵니다. 이 데이터는 서울시의 구별로 측정된 미세먼지 정보를 포함하고 있습니다.

2. **데이터 가공**: 데이터를 수집한 후, 데이터 가공 작업을 수행합니다. 누락된 값은 NaN으로 처리하고, 분석에 필요한 형태로 데이터를 변환합니다.

3. **데이터 시각화**: Folium 라이브러리를 활용하여 서울시 지도 위에 미세먼지 데이터를 시각화합니다. 이를 통해 서울의 다양한 지역에서의 대기질 상태를 한 눈에 확인할 수 있습니다.

**결과** 
- 서울시 구별로 미세먼지를 시각화 하고 미세먼지 심각성과 미세먼지 농도를 개선하기 위한 개선책을 수립할 필요가 있음.



## 프로젝트 개요

***2. 주식뉴스 크롤링 후 afinn 감성분석***

**프로젝트 목표:** 이 프로젝트는 2021년 동안의 삼성전자 주식 가격과 뉴스 기사 간의 관계를 분석하여 주식 시장의 트렌드와 뉴스 간의 상관 관계를 조사하는 것을 목표로 합니다.

**프로젝트 설명:** 

1. **주식 데이터 수집**: FinanceDataReader 라이브러리를 사용하여 2021년 1월 1일부터 2021년 12월 31일까지의 삼성전자 주식 데이터를 수집합니다. 불필요한 열을 삭제하고 주식 종가 데이터를 활용합니다.

2. **뉴스 크롤링**: 네이버 금융 뉴스 웹페이지에서 주어진 기간 동안의 삼성전자 주식 관련 뉴스 기사를 크롤링합니다. 크롤링된 기사의 제목과 날짜를 추출하고, 중복된 기사를 제거합니다.

3. **데이터 분석**: 주식 데이터와 뉴스 데이터를 통합하고, 주식 가격 변동과 뉴스 간의 상관 관계를 분석합니다. 뉴스 제목을 한영 번역한 후 감성 점수를 계산하여 긍정적 또는 부정적인 뉴스와 주식 가격 변동 간의 관계를 조사합니다.

4. **결과 도출**: 주식 가격 변동과 뉴스 간의 상관 관계를 시각화하고, 감성 점수와 주식 가격 변동의 관계를 평가합니다. 이를 통해 주식 시장에 영향을 미치는 뉴스의 트렌드와 패턴을 발견하고 분석합니다.

**결과 및 한계점 :** 이 프로젝트를 통해 사용자는 삼성전자 주식과 관련된 뉴스와 주식 가격 변동 간의 관계를 이해하고, 뉴스의 감성 요소가 주식 시장에 미치는 영향을 파악할 수 있습니다. 이를 통해 투자 전략을 개발하거나 주식 시장 동향을 예측하는 데 도움을 얻을 수 있습니다. 하지만 한국어 감성사전을 활용하기 어려워 영문으로 번역후 감성분석을 하였기 때문에 정확성이 떨어질 수 있는 한계점이 있습니다.



## 프로젝트 개요
이 프로젝트는 이안류 탐지를 위한 딥러닝 모델을 개발하고 테스트하는 것을 목표로 합니다. 이안류는 매년 여름 해수욕장에서 발생하며 신속한 구조와 예측이 중요합니다. 

***3. 이안류 분류를 위한 cnn모델*** 

### 프로젝트 설명

1. **데이터 수집 및 전처리**: 학습을 위한 이미지 데이터와 해당 이미지에 대한 이안류 여부를 나타내는 라벨 데이터를 수집하고 전처리합니다. 이미지 크기를 조정하고 데이터를 텐서 형식으로 변환합니다.

2. **모델 구축**: 사전 훈련된 VGG16 모델을 기반으로 한 딥러닝 모델을 구축합니다. 모델은 이미지 분류를 위한 컨볼루션 레이어와 긍정 또는 부정 클래스를 분류하기 위한 출력 레이어로 구성됩니다.

3. **학습 및 클래스 가중치 적용**: 모델을 학습하고 클래스 불균형을 해결하기 위해 클래스 가중치를 적용합니다. 이를 통해 모델이 불균형한 데이터에 대해 더 효과적으로 학습됩니다.

4. **테스트 및 결과 시각화**: 학습된 모델을 사용하여 테스트 이미지에 대한 이안류 탐지를 수행하고 결과를 시각화합니다. 이안류가 탐지된 경우 해당 부분에 바운딩 박스를 그립니다.

### 기술 스택
프로젝트에서 사용된 기술 스택

- Python
- TensorFlow 및 Keras 라이브러리
- PIL (Python Imaging Library)
- OpenCV
- JSON 파일 처리
- 데이터 전처리 및 가중치 계산

### 프로젝트 결과
학습된 딥러닝 모델을 사용하여 정확한 이안류를 탐지는 실패하였지만 추후에 개선한다면 해수욕장 cctv에 사용하여 좋은 모델이 될 수 있다고 생각합니다.

