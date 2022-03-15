# Anomaly Detection - FPC

Feture extractor + PCA + Cluster

- **작업 기간**
2022.01~2022.01 (1개월)

- **인력 구성(기여도)**
feature extractor 2명 (50%), post processing 2명 (50%), 총 3명.

- **프로젝트 개요**
간 데이터를 이용해 암을 찾아내는 anomaly detection project (독성병리학).
정상 데이터만으로 feature extractor을 fine-tuning, varidation 데이터로 cluster 학습 후, test data로 예측을 진행하는 semi-supervised model.

- **평가 방식**
train(정상) / varidation(정상, 비정상), test(no label) 형식으로 데이터를 분리한다. 
그 후 Confusion matrix로 평가를 진행한다.

- **제한 사항**
    - 지도 학습 불가.
    - 클래스 불균형 (class imbalance).
    - 모든 label 제공 불가.
    - recall 1.
    - 빠른 inference 속도.
    - 데이터 공개 불가.

---

# 데이터 설명

- 총 데이터 개수 100개, 정상 90개, 비정상 10개.
- 데이터 형식 : Whole Slide Image (WSI), .mrxs 파일 ( + metadata, .dat 파일 )
- 데이터 크기 : 평균 (77000, 185000, 4), bitmap 기준 대략 56GB

- Label 설명
    - 데이터 이미지에서 비정상 영역에 xml 파일로 boundary 및 병명이 기제 되어 있다.
    - 한 비정상 데이터당 label 개수 : 2~5개, 20~30개 등으로 다양하다.
    
    [Anomaly feature (데이터 상세)](https://www.notion.so/Anomaly-feature-3992a4f1e25c4d10acde53ec7d81eae5)
    

---

# 결과

전처리는 [Cell Based Model](https://www.notion.so/Anomaly-Detection-Cell-Based-Model-dc4f87510468429b8f0f607be7eb64dd)과 동일하게 진행했습니다.

### Backbone(Feature extractor) 실험 결과

![Untitled](Anomaly%20De%20771e1/Untitled.png)

[PCA & Cluster](https://www.notion.so/d3f11d2e67d1486897585d686f6391f3)

[Result](https://www.notion.so/bedc6fc40348441c860a4f6a38f466b5)

📁Github

[GitHub - essential2189/FPC](https://github.com/essential2189/FPC)