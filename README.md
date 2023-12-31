## 프로젝트 1: ESG_folium


**프로젝트 목표:** 서울시 구별 미세먼지 데이터 시각화

**프로젝트 설명:** 

1. **데이터 수집**: 2016년부터 2021년까지의 서울 대기질 데이터를 두 개의 Excel 파일에서 읽어옵니다. 이 데이터는 서울시의 구별로 측정된 미세먼지 정보를 포함하고 있습니다.

2. **데이터 가공**: 데이터를 수집한 후, 데이터 가공 작업을 수행합니다. 누락된 값은 NaN으로 처리하고, 분석에 필요한 형태로 데이터를 변환합니다.

3. **데이터 시각화**: Folium 라이브러리를 활용하여 서울시 지도 위에 미세먼지 데이터를 시각화합니다. 이를 통해 서울의 다양한 지역에서의 대기질 상태를 한 눈에 확인할 수 있습니다.

**사용된 기술 스택:**
- Python
- Pandas 및 NumPy: 데이터 수집 및 가공
- Folium: 지도 시각화
- Excel 파일 처리 (데이터 수집)
- 데이터 가공 및 시각화

## 프로젝트 2: 주식뉴스 크롤링 후 afinn 감성분석


**프로젝트 목표:** 2021년 동안의 삼성전자 주식 가격과 뉴스 기사 간의 관계를 분석하여 주식 시장의 트렌드와 뉴스 간의 상관 관계를 조사

**프로젝트 설명:** 

1. **주식 데이터 수집**: FinanceDataReader 라이브러리를 사용하여 2021년 1월 1일부터 2021년 12월 31일까지의 삼성전자 주식 데이터를 수집합니다. 불필요한 열을 삭제하고 주식 종가 데이터를 활용합니다.

2. **뉴스 크롤링**: 네이버 금융 뉴스 웹페이지에서 주어진 기간 동안의 삼성전자 주식 관련 뉴스 기사를 크롤링합니다. 크롤링된 기사의 제목과 날짜를 추출하고, 중복된 기사를 제거합니다.

3. **데이터 분석**: 주식 데이터와 뉴스 데이터를 통합하고, 주식 가격 변동과 뉴스 간의 상관 관계를 분석합니다. 뉴스 제목을 한영 번역한 후 감성 점수를 계산하여 긍정적 또는 부정적인 뉴스와 주식 가격 변동 간의 관계를 조사합니다.

**사용된 기술 스택:**
- Python
- Pandas 및 NumPy: 데이터 수집 및 가공
- FinanceDataReader: 주식 데이터 수집
- 웹 스크래핑 (뉴스 데이터 수집)
- AFINN 감성 분석 라이브러리
- 데이터 분석 및 시각화

## 프로젝트 3: 이안류 분류를 위한 CNN 모델

**프로젝트 목표:** CNN을 통한 이안류 분류 모델 생성


**프로젝트 설명:**

1. **데이터 수집 및 전처리**: 학습을 위한 이미지 데이터와 해당 이미지에 대한 이안류 여부를 나타내는 라벨 데이터를 수집하고 전처리합니다. 이미지 크기를 조정하고 데이터를 텐서 형식으로 변환합니다.

2. **모델 구축**: 사전 훈련된 VGG16 모델을 기반으로 한 딥러닝 모델을 구축합니다. 모델은 이미지 분류를 위한 컨볼루션 레이어와 긍정 또는 부정 클래스를 분류하기 위한 출력 레이어로 구성됩니다.

3. **학습 및 클래스 가중치 적용**: 모델을 학습하고 클래스 불균형을 해결하기 위해 클래스 가중치를 적용합니다. 이를 통해 모델이 불균형한 데이터에 대해 더 효과적으로 학습됩니다.

4. **테스트 및 결과 시각화**: 학습된 모델을 사용하여 테스트 이미지에 대한 이안류 탐지를 수행하고 결과를 시각화합니다. 이안류가 탐지된 경우 해당 부분에 바운딩 박스를 그립니다.

**사용된 기술 스택:**
- Python
- TensorFlow 및 Keras 라이브러리
- PIL (Python Imaging Library)
- OpenCV
- JSON 파일 처리
- 데이터 전처리 및 가중치 계산



## 프로젝트 4: 부천시 교통안전지수 및 노인보호구역 위치 추천

**프로젝트 목표:** 부천시 여러 교통요인을 통한 새로운 노인보호구역 위치 추천

**프로젝트 설명:**

1. **데이터 수집 및 전처리**: 프로젝트에서 사용된 데이터는 부천시에 관한 데이터입니다.
   - 노인 보호 구역 및 사고 다발 지역의 좌표 정보를 활용하기 위해 'pyproj' 패키지를 사용하여 좌표 변환을 수행합니다.
   - 지리 정보를 다루기 위해 'geopandas' 라이브러리를 활용하여 지오데이터로 변환합니다.
   - 사고 발생 정보를 계산하여 위험 요소와 '위험 지수'를 생성합니다.

2. **모델링**: XGBoost 모델을 사용하여 교통 안전과 사고 위험을 종합적으로 평가하는 '위험 지수'를 계산합니다.
   - 모델의 학습 및 평가를 위해 데이터를 학습 및 테스트 세트로 분할하고, 모델의 성능 지표를 확인합니다.

3. **데이터 시각화**: 'folium' 라이브러리를 활용하여 위험 지수 및 사고 다발 지역을 지도 상에 시각화합니다.
   - Choropleth 맵을 사용하여 격자로 나눈 지역에 위험 지수를 표시하고, 실제 노인 보호 구역을 마커로 나타냅니다.

**사용된 기술 스택:**
- Python
- 'pyproj' 패키지 (좌표 변환)
- 'geopandas' 라이브러리 (지리 정보 처리)
- 'folium' 라이브러리 (지도 시각화)
- XGBoost (머신 러닝 모델)

**프로젝트 결과:** 부천시는 이 지도를 통해 교통 안전과 사고 위험을 확인하고, 새로운 노인 보호 구역을 추천받을 수 있습니다.



## 프로젝트 5:   소상공인의 효과적인 VOC 분석을 위한 Doc2Vec 활용
 
** 폴더 내 한글 논문 자료**
