# 시간의존형 용량 제약 최대 커버리지 모델 (TD-CMCLP)

## 1. 모델 개요 (Overview)

본 모델은 대전시 트램 공사 국면에서 **0.5초 이내 실시간 관제**를 달성하기 위해 2-Echelon 물류망 중 라스트마일(AMR) 영역에 집중한 **Spatial-Temporal Decomposition** 모델입니다.

변수 폭발을 막기 위해 **1.5km 이내 유효 배송쌍만 사전 생성하는 Sparse Indexing** 기법을 적용하였으며, XGBoost 정체 예측 연동, 턴어라운드 병목, 동적 배터리 소모를 모두 포함합니다.

---

## 2. 집합 및 인덱스 (Sets and Indices)

| 기호 | 정의 | 규모 |
|---|---|---|
| $i \in I$ | 배송 수요지 (소상공인 상가, 그리드 집계) | ~7,500개 |
| $j \in J$ | 가상 조업 거점 후보지 (사전 필터링) | ~100–500개 |
| $t, \tau \in T$ | 당일 운영 타임슬롯 (10분 단위 8시간) | $1, \dots, 48$ |
| $\Omega \subseteq I \times J \times T$ | **유효 가능 조합 인덱스 집합 (Sparse Set)** | ~150만 조합 |

$$\Omega = \{(i, j, t) \mid d_{ij} \le 1.5\text{km},\; t + tt_{ij} \in [E_i, L_i]\}$$

---

## 3. 주요 파라미터 및 자동 산출식 (Parameters)

### 3.1 수요 및 목적함수 계수

| 기호 | 정의 | 비고 |
|---|---|---|
| $w^{eff}_{ij}$ | 거리 쇠퇴 및 위험도가 반영된 배송 유효 수요 가중치 | 네트워크BC × DTW × XGB multiplier 반영 |
| $f_j$ | 거점 $j$의 합법성 및 인센티브 점수 | $0 \le f_j \le 1$ |
| $q_i$ | 상가 $i$의 일일 배송 물동량 | 데이터 부재 시 업종별 가중치 $\text{CategoryWeight}(cat_i)$ 대입 |
| $\lambda$ | 자동 산출 스케일링 계수 | 아래 수식으로 산출 |

$$\lambda = \frac{\displaystyle\sum_{(i,j,t) \in \Omega} w^{eff}_{ij}}{\displaystyle\sum_{j \in J} f_j + \epsilon}$$

### 3.2 물리 제약 및 자원 동역학

| 기호 | 정의 | 기본값 |
|---|---|---|
| $P$ | 최대 허용 거점 수 | 10–30 |
| $V_{total}$ | 총 투입 가능 로봇 수 | 설정 파라미터 |
| $C_j$ | 거점 $j$의 로봇 최대 수용 대수 | 주차면수 기반 |
| $Q$ | 로봇 1대당 1회 최대 적재량 | 설정 파라미터 |
| $N$ | 로봇 1대 일일 최대 왕복 횟수 | 설정 파라미터 |
| $B_{max}$ | 로봇 1대 가용 배터리 총량 | 설정 파라미터 |
| $\gamma_1$ | 주행 거리 비례 배터리 소모 계수 | 설정 파라미터 |
| $\gamma_2$ | 적재 중량 비례 배터리 소모 계수 | 설정 파라미터 |
| $\beta_{cap}$ | 과적 방지 풀링 안전계수 | 0.8 |
| $\beta_{bat}$ | 방전 방지 풀링 안전계수 | 0.8 |
| $tt_{ij}$ | 거점 $j \to$ 상가 $i$ 로봇 편도 소요 시간 (타임슬롯 단위) | Haversine/네트워크 기반 |
| $\overline{tt}_{j}^{turn}$ | 거점 $j$의 평균 왕복 준비 시간 | $\text{Mean}_{i}(2 \cdot tt_{ij}) + t_{service}$ |
| $[E_i, L_i]$ | 상가 $i$의 수령 가능 타임 윈도우 | 외생 데이터 |
| $t_{handling}$ | 탑차 도착 후 하역 소요 시간 | 설정 파라미터 |
| $U_{jt} \in \{0,1\}$ | 외생 탑차 도착 스케줄 | 상위 시스템에서 확정 전달 |

### 3.3 XGBoost 연동 시변 교통체증 파라미터

| 기호 | 정의 | 산출 |
|---|---|---|
| $S^{truck}_{jt} \in \{0,1\}$ | 시간 $t$에 거점 $j$ 주변 차도 심각 정체 여부 | XGBoost `prob_severe ≥ 0.25` |
| $S^{AMR}_{ijt} \in \{0,1\}$ | 시간 $t$에 $j \to i$ 인도/횡단보도 지연 여부 | XGBoost + 보호구역 밀집도 |
| $R_{jt} \in \{0,1\}$ | 시간 $t$에 거점 $j$에 하역 완료 재고 존재 여부 (사전 계산) | 아래 정의 |

$$R_{jt} = \begin{cases} 1 & \text{if } \exists\, \tau \le t - t_{handling} \text{ s.t. } U_{j\tau} = 1 \text{ and } S^{truck}_{j\tau} = 0 \\ 0 & \text{otherwise} \end{cases}$$

---

## 4. 의사결정 변수 (Decision Variables)

| 변수 | 정의 | 도메인 |
|---|---|---|
| $x_j$ | 거점 $j$ 오픈 여부 | $\{0, 1\}$ |
| $v_j$ | 거점 $j$에 배치할 로봇 대수 | $\mathbb{Z}_{\ge 0}$ |
| $y_{ijt}$ | 상가 $i$를 시간 $t$에 거점 $j$에서 출발시켜 배송 | $\{0, 1\},\; \forall (i,j,t) \in \Omega$ |

> **Sparse Variable**: $y_{ijt}$는 전체 $I \times J \times T$가 아닌 유효 집합 $\Omega$ 위에서만 정의됩니다.

---

## 5. 목적 함수 (Objective Function)

$$\max Z = \sum_{(i,j,t) \in \Omega} w^{eff}_{ij}\, y_{ijt} \;+\; \lambda \sum_{j \in J} f_j\, x_j$$

| 항 | 의미 |
|---|---|
| 제1항 | 교통 혼잡 속에서도 최대한 많은 상인에게 안전하고 짧은 경로로 배송 |
| 제2항 | 불법 주정차 단속을 피하고 하역 공간이 넓은 우수 거점을 선택하도록 유인 |
| $\lambda$ | 수만 단위의 물동량과 한 자릿수의 거점 점수가 동등하게 겨루도록 저울 영점 조정 (자동 산출) |

---

## 6. 수학적 제약식 (Mathematical Constraints)

### 6.1 위상 및 공간 할당 제약

#### ① 거점 총량 통제

$$\sum_{j \in J} x_j \le P$$

> 지자체에서 허가한 임시 조업 주차 구역의 최대 개수($P$)를 초과하여 거점을 오픈하는 것을 차단합니다.

#### ② 논리적 배정 종속성

$$y_{ijt} \le x_j \quad \forall (i,j,t) \in \Omega$$

> 미오픈 거점($x_j=0$)에서의 로봇 출발을 원천 차단합니다.

#### ③ 수요지 단일 할당 원칙

$$\sum_{(j,t):\,(i,j,t) \in \Omega} y_{ijt} \le 1 \quad \forall i \in I$$

> 상가당 하루 1회 단일 배송 원칙을 강제하여 중복 배차를 방지합니다.

#### ④ 거점 카니발라이제이션 방지 (분산 강제)

$$x_j + x_k \le 1 \quad \forall (j,k) \text{ where } dist(j,k) \le D_{min}$$

> 최소 이격 거리($D_{min}$, 기본 500m) 이내에 거점이 중복 오픈되는 것을 금지하여 공공 물류망의 분산 배치를 강제합니다.

---

### 6.2 하드 제약 사전 필터링 (Sparse Set 구성 시 적용)

> 아래 제약들은 $\Omega$ 구성 단계에서 사전 필터링으로 처리됩니다.

| 필터 | 조건 | 효과 |
|---|---|---|
| 소방시설 5m | $h_j = 0$ if 소화전 반경 5m 이내 | $j \notin J$ (후보 제거) |
| 보도 접근불가 | $a_{ij} = 0$ if 단차/보도폭 미달 | $(i,j,\cdot) \notin \Omega$ |
| 타임 윈도우 | $t + tt_{ij} \notin [E_i, L_i]$ | $(i,j,t) \notin \Omega$ |
| 커버리지 반경 | $d_{ij} > 1.5\text{km}$ | $(i,j,\cdot) \notin \Omega$ |

---

### 6.3 자원 한계 및 용량 제약 (Safe Pooling)

#### ⑤ 전체 로봇 가용 예산 한계

$$\sum_{j \in J} v_j \le V_{total}$$

> 투입 가능한 로봇의 총대수를 초과 배치하는 것을 방지합니다.

#### ⑥ 거점 수용 및 최소 운영 효율

$$3 \cdot x_j \le v_j \le C_j \cdot x_j \quad \forall j \in J$$

> **상한**: 주차장 면적 한계($C_j$)를 초과하는 과밀 배치를 차단합니다.
> **하한**: 오픈 시 최소 3대 이상 배치를 강제하여 현장 관리 인건비 대비 채산성을 확보합니다.

#### ⑦ 적재 용량 한계 (안전계수 반영)

$$\sum_{(i,t):\,(i,j,t) \in \Omega} q_i\, y_{ijt} \le \beta_{cap} \cdot Q \cdot N \cdot v_j \quad \forall j \in J$$

> 풀링 안전계수 $\beta_{cap}=0.8$을 적용하여 피크 타임 과적을 예방합니다. $q_i$는 업종별 가중치로 대입됩니다.

#### ⑧ 동적 하중-배터리 소모 한계 (안전계수 반영)

$$\sum_{(i,t):\,(i,j,t) \in \Omega} \left( \gamma_1 d_{ij} + \gamma_2 q_i d_{ij} \right) y_{ijt} \le \beta_{bat} \cdot B_{max} \cdot v_j \quad \forall j \in J$$

> 화물 중량 비례 배터리 소모 역학을 반영합니다. 무거운 화물은 근거리 거점에, 가벼운 화물은 원거리 거점에 배정되도록 유도합니다.

---

### 6.4 시공간 동기화 제약 (Tight Bounds Applied)

#### ⑨ 턴어라운드 로봇 병목 통제

$$\sum_{\tau = \max(1,\, t - \overline{tt}_{j}^{turn})}^{t} \;\sum_{i:\,(i,j,\tau) \in \Omega} y_{ij\tau} \;\le\; v_j \quad \forall j \in J,\; \forall t \in T$$

> 거점 $j$의 평균 복귀 시간($\overline{tt}_j^{turn}$) 윈도우 동안 출발한 로봇의 합이 보유 대수($v_j$)를 초과할 수 없습니다. VRP를 풀지 않고도 동시간대 로봇 부족 병목을 0.5초 내에 연산합니다.
>
> **해석**: "시점 $t$ 기준으로 아직 복귀하지 못한(운행 중인) 로봇의 총합 ≤ 거점 보유 대수"

#### ⑩ 탑차-AMR 크로스도킹 동기화 (Tight Big-M)

$$\sum_{i:\,(i,j,t) \in \Omega} y_{ijt} \;\le\; C_j \cdot R_{jt} \quad \forall j \in J,\; \forall t \in T$$

여기서 $R_{jt}$는 사전 계산 파라미터:

$$R_{jt} = \begin{cases} 1 & \text{if } \exists\, \tau \le t - t_{handling} \text{ s.t. } U_{j\tau} = 1 \text{ and } S^{truck}_{j\tau} = 0 \\ 0 & \text{otherwise} \end{cases}$$

> 탑차가 도착하여 하역($t_{handling}$)을 완료해야만 AMR 배차가 풀립니다. 차도 정체($S^{truck}=1$) 발생 시 탑차 미도착으로 처리되어 AMR도 정지합니다. Big-M을 $C_j$(거점 수용 한계)로 tight하게 설정하여 LP relaxation gap을 최소화합니다.

#### ⑪ AMR 보행로 정체 회피

$$y_{ijt} \le 1 - S^{AMR}_{ijt} \quad \forall (i,j,t) \in \Omega$$

> XGBoost가 예측한 인도/횡단보도 지연 시간대($S^{AMR}=1$)에는 해당 경로의 배정을 강제 차단하여 AMR 안전 운행을 보장합니다.

---

### 6.5 공공성 하한선 보장 (Coverage Floor)

#### ⑫ 최소 커버리지 방어선

$$\sum_{(i,j,t) \in \Omega} y_{ijt} \ge \alpha \cdot |I|$$

> 대전시 전체 배송 수요지의 최소 비율($\alpha$, 기본 0.8) 이상을 보장하여 외곽 지역 소외를 방지합니다. Infeasible 시 $\alpha$를 0.05 단위로 자동 완화합니다.

---

## 7. 계산 복잡도 분석

| 항목 | Full ILP (naive) | Sparse Indexing (본 모델) |
|---|---|---|
| $y_{ijt}$ 변수 수 | $7,500 \times 500 \times 48 = 1.8$억 | $\|\Omega\| \approx 150$만 |
| 제약식 수 | 수십억 | ~300만 |
| 예상 솔버 시간 | 풀 수 없음 | **1~5분 (CBC)** |
| Warm-start 적용 시 | — | **< 30초** |

---

## 8. 데이터-파라미터 연결표

| 파라미터 | 데이터 소스 | 현재 상태 |
|---|---|---|
| $w^{eff}_{ij}$ | `outputs/segment_priority.csv` × DTW × XGB | ✅ 구현 완료 |
| $f_j$ | 주차면수 정규화 + 주정차 합법 보너스 | ✅ 구현 완료 |
| $q_i$ | 업종별 가중치 (음식점 5, 소매 3, 서비스 1) | △ 가정값 사용 |
| $C_j$ | `data/주차장 정보 표준데이터/` 주차구획수 | ✅ |
| $tt_{ij}$ | Haversine 거리 / AMR 속도 10km/h | ✅ |
| $S^{truck}_{jt}$ | XGBoost `prob_severe` ≥ 0.25 | ✅ reference tables 보유 |
| $S^{AMR}_{ijt}$ | 보호구역 + 횡단보도 밀집도 | ✅ 데이터 보유 |
| $U_{jt}$ | 탑차 경로 시스템 (외생) | △ 시뮬레이션용 가정 필요 |
| $[E_i, L_i]$ | 상가 영업시간 | △ 일괄 09:00~18:00 가정 |

---

## 9. 모듈 간 데이터 플로우

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  네트워크 BC │     │   Prophet    │     │  DTW 군집     │
│  (모듈 3)   │     │  (모듈 1)    │     │  (모듈 4)     │
└──────┬──────┘     └──────┬───────┘     └───────┬───────┘
       │                   │                     │
       │   betweenness     │  y_hat_t30          │  cluster_multiplier
       │   lane_ratio      │  y_hat_lower        │  rapid_multiplier
       ▼                   ▼                     │
┌──────────────────────────────────┐             │
│         XGBoost (모듈 2)          │             │
│  → prob_severe, S^truck, S^AMR   │             │
└──────────────┬───────────────────┘             │
               │                                 │
               ▼                                 ▼
┌────────────────────────────────────────────────────┐
│              TD-CMCLP (본 모듈)                      │
│                                                      │
│  입력:                                               │
│    w_eff_ij ← BC priority × DTW mult × XGB mult     │
│    S^truck_jt, S^AMR_ijt ← XGBoost 예측             │
│    C_j, tt_ij ← 주차장 데이터 + 네트워크             │
│                                                      │
│  출력:                                               │
│    x_j (거점 선택), v_j (로봇 배치), y_ijt (배차)     │
└────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────┐
│  대시보드 / 관제 시스템    │
│  거점 지도 + 실시간 경보   │
└──────────────────────────┘
```

---

## 10. 한계 및 향후 보완

| 항목 | 현재 상태 | 보완 방향 |
|---|---|---|
| $q_i$ 실측 물동량 | 업종별 가정값 | 택배사 API 또는 카드매출 데이터 연동 |
| $U_{jt}$ 탑차 스케줄 | 외생 가정 | 상위 VRP 모듈 구축 또는 물류사 연계 |
| 0.5초 실시간 | CBC 1~5분 | Warm-start + Gurobi 또는 Heuristic 초기해 |
| DEM 경사도 | 미반영 | 국토지리정보원 DEM 확보 시 $\Omega$ 필터에 추가 |
| 9공구 착공 후 데이터 | ITS 미관측 (7/7 착공 > 7/1 데이터 마감) | ITS 축적 후 자동 갱신 |
| Prophet 18/90 구간 | 저성능 보류 | fallback 예측 또는 DTW 정보만으로 대체 |
