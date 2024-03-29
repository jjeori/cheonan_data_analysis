# cheonan_data_analysis

한국평가데이터에서 주최한 공모전에서 천안을 효과적으로 홍보하는 아이디어 제안 주제로 프로젝트를 수행했습니다.
<br/>2022년 9월 ~ 12월 동안 진행되었고, 이 공모전에서 3위를 수상하게 되었습니다.

-----
### **문제 정의**
* 천안은 다양한 사람들에게 효과적으로 홍보할 만한 매력적인 요소가 없다

### **문제 해결 방안 및 분석 목표**
* 매력적인 요소를 찾는 것이 아닌, 연간 천안 독립기념관에 방문하는 11만 이상의 군인들을 핵심 타깃으로 선정
* 타깃 분석을 통해 천안을 효과적으로 알릴 수 있는 아이디어 제안하기

### **분석 과정**
* 군인 데이터 수집을 위해 크롤링을 활용하여 '군인', '휴가', '독립기념관' 해시태그 데이터 수집
* 메인 해시태그를 중심으로 연관 태그 분석을 진행
* 분석 결과 군인들은 독립기념관 방문 시점인 휴가를 나왔을 때 **여행 연관 키워드가 많다는 점**과 독립기념관 방문 시 **누군가와 함께 방문한다는 점**을 도출 

### **아이디어 제안**
* 군인들에게 천안은 누군가와 함께 다양한 여행을 할 수 있는 곳으로 포지셔닝

### **아이디어 구체화**
* 20대 남성 여행시간 활용 목적 데이터를 통해 여행 유형을 4가지로 분류
* 전국, 충청남도 여행키워드 데이터 전처리 후, 천안에서 경험할 수 있는 여행 키워드 선별
* 여행 키워드 별 데이터 수집을 위해 천안의 흥 홈페이지 크롤링을 통해 데이터 수집
* 여행 유형 및 여행 키워드에 맞게 데이터를 정리 후 Tableau 대시보드 제작
<img width="1088" alt="스크린샷 2023-11-06 오후 4 55 03" src="https://github.com/jjeori/cheonan_data_analysis/assets/99062088/b3d416fc-343c-408a-bc60-5a78ff520cd6">

### **대시보드 활용 아이디어 제안**
* 군인들이 독립기념관을 방문할 때 사용하는 현충시설기념관 안내 어플에 대시보드 도입 방안 제안
<img width="1138" alt="스크린샷 2023-11-06 오후 5 01 02" src="https://github.com/jjeori/cheonan_data_analysis/assets/99062088/f0d94ae8-a657-4dc5-a30b-8564775a3986">

### **대시보드 시연 영상**
* https://www.youtube.com/watch?v=4cbMrLXMoek

### **의의 및 한계**
* 의의: 군인이라는 타깃의 차별성을 두었다는 점과 수집한 데이터에서 근거를 찾아 논리성을 확보했다는 점
* 한계: 국방부에서 제공하는 공공데이터에서 활용할 데이터가 없었기에 크롤링 결과 타깃 데이터의 수가 적었다는 점과 추가적으로 비즈니스 측면으로 어떤 이점이 있을까에 대한 고민을 하지 못한 점이 아쉬움으로 남음
