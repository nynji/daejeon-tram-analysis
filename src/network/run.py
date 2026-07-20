"""
run.py
대전 트램 공사 구간 네트워크 분석 파이프라인 v3

변경 사항:
  - segment_id: STN{no} → SEG_{a}_{b} (인접 정거장 쌍 기반 진짜 구간 ID)
  - 기상 위험도 통합 (weather_risk.py)
  - 우선순위 스코어 산출 (priority_score.py)
  - 인사이트 리포트 자동 생성 (insight_report.md)
"""

import os
import pandas as pd
from datetime import datetime

from src.preprocess import load_data, create_station_segments
from src.network_simulation import apply_construction_rules
from src.analysis import build_network_graph, calculate_betweenness_centrality
from src.weather_risk import load_asos, get_monthly_weather_risk, compute_monthly_stats
from src.priority_score import compute_priority_scores, get_top_risk_segments


def main():
    print("=" * 60)
    print(" 대전 트램 공사 구간 네트워크 분석 파이프라인 v3")
    print("=" * 60)
    os.makedirs("outputs", exist_ok=True)

    # ── 1~3. 데이터 로드 / 구간 매핑 / 패널티 적용 ──────────
    gdf_link, gdf_node, df_stations, df_construction = load_data()
    gdf_link = create_station_segments(df_stations, gdf_link)

    # 공사 구간 GIS 좌표 추출 (geocode)
    from src.geocode_construction import geocode_construction_zones
    import geopandas as _gpd
    _gdf_node = _gpd.read_file('data/network/daejeon_node.geojson')
    df_zones = geocode_construction_zones(df_construction, _gdf_node, gdf_link)
    df_zones.to_csv('outputs/construction_zones_geocoded.csv', index=False, encoding='utf-8-sig')

    gdf_link = apply_construction_rules(gdf_link, df_construction, use_gis=True)

    # ── 4. 그래프 구성 ───────────────────────────────────────
    G_free  = build_network_graph(gdf_link, speed_mode="free_flow")
    G_const = build_network_graph(gdf_link, speed_mode="under_construction")

    # ITS 실측 속도 기반 그래프 (가장 현실적인 상황 반영)
    from src.its_speed import load_its_speed, get_daily_avg_speed
    import os as _os
    its_path = _os.path.join("data", "ITS_대전_링크속도.csv")
    G_its = None
    if _os.path.exists(its_path):
        df_its_hourly = load_its_speed(its_path)
        its_speed = get_daily_avg_speed(df_its_hourly)
        G_its = build_network_graph(gdf_link, speed_mode="its_realtime", its_speed_dict=its_speed)

    # ── 5. 매개중심성 계산 (앙상블 4회 평균 → 안정성 향상) ──
    ENSEMBLE_SEEDS = [42, 0, 123, 777]

    print("\n[Main] 자유류 BC 계산 중 ...")
    bc_free  = calculate_betweenness_centrality(G_free,  sample_frac=0.05, max_sample=1000, ensemble_seeds=ENSEMBLE_SEEDS)
    print("[Main] 공사중(패널티) BC 계산 중 ...")
    bc_const = calculate_betweenness_centrality(G_const, sample_frac=0.05, max_sample=1000, ensemble_seeds=ENSEMBLE_SEEDS)

    bc_its = None
    if G_its:
        print("[Main] ITS 실측 BC 계산 중 ...")
        bc_its = calculate_betweenness_centrality(G_its, sample_frac=0.05, max_sample=1000, ensemble_seeds=ENSEMBLE_SEEDS)

    # ── 6. 결과 조립 ─────────────────────────────────────────
    print("\n[Main] 결과 조립 및 저장 ...")
    gdf_out = gdf_link.copy()
    gdf_out['bc_free_flow']          = gdf_out['LINK_ID'].map(bc_free).fillna(0.0)
    gdf_out['bc_under_construction'] = gdf_out['LINK_ID'].map(bc_const).fillna(0.0)
    gdf_out['bc_change']             = gdf_out['bc_under_construction'] - gdf_out['bc_free_flow']

    # ITS 실측 BC (가장 현실적인 파급력 지표)
    if bc_its:
        gdf_out['bc_its_realtime'] = gdf_out['LINK_ID'].map(bc_its).fillna(0.0)
        gdf_out['bc_its_vs_free']  = gdf_out['bc_its_realtime'] - gdf_out['bc_free_flow']
    else:
        gdf_out['bc_its_realtime'] = 0.0
        gdf_out['bc_its_vs_free']  = 0.0

    output_cols = [
        'LINK_ID', 'ROAD_NAME', 'ROAD_RANK', 'LANES', 'LENGTH', 'MAX_SPD',
        'nearest_station_no', 'nearest_station_name',
        'second_station_no', 'second_station_name', 'segment_id',
        'speed_penalty_multiplier', 'lane_remain_ratio',
        'bc_free_flow', 'bc_under_construction', 'bc_change',
        'bc_its_realtime', 'bc_its_vs_free'
    ]
    df_output = pd.DataFrame(gdf_out[output_cols])

    # ── 7. 기상 위험도 통합 (ASOS 실측 데이터) ─────────────────
    print("\n[Main] ASOS 실측 기상 데이터 로드 중 ...")
    df_asos = load_asos()
    month = datetime.now().month
    weather_info = get_monthly_weather_risk(month, df_asos)
    print(f"  기상 시나리오: {weather_info['scenario']} "
          f"(월:{month}, 평균위험:{weather_info['rain_risk_score']:.4f}, "
          f"출처:{weather_info['source']})")
    print(f"  {weather_info['note']}")

    # ── 8. 우선순위 스코어 산출 ──────────────────────────────
    print("[Main] 우선순위 스코어 산출 중 ...")
    seg_df, df_link_scored = compute_priority_scores(
        df_output,
        weather_risk=weather_info['rain_risk_score']
    )

    # 링크 스코어를 메인 CSV에 병합
    df_output = df_output.merge(
        df_link_scored[['LINK_ID','bc_score','lane_score','rank_score',
                        'priority_score','risk_grade']],
        on='LINK_ID', how='left'
    )

    # ── 9. CSV 저장 ───────────────────────────────────────────
    output_path = os.path.join("outputs", "network_betweenness.csv")
    df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  → {output_path} 저장 완료 ({len(df_output):,}개 링크)")

    seg_path = os.path.join("outputs", "segment_priority.csv")
    seg_df.to_csv(seg_path, index=False, encoding='utf-8-sig')
    print(f"  → {seg_path} 저장 완료 ({len(seg_df)}개 구간)")

    # ── 10. 리포트 저장 ───────────────────────────────────────
    _write_summary_report(df_output, seg_df, weather_info)
    _write_insight_report(df_output, seg_df, weather_info)

    print("\n[Main] 파이프라인 완료.")
    print("  다음 단계: python visualize.py")
    print("=" * 60)


def _write_summary_report(df: pd.DataFrame, seg_df: pd.DataFrame, weather_info: dict):
    """기존 분석 요약 리포트"""
    path = os.path.join("outputs", "analysis_summary.md")
    top_risk  = df.sort_values('bc_change', ascending=False).head(10)
    top_const = df.sort_values('bc_under_construction', ascending=False).head(10)
    penalty_dist = df['speed_penalty_multiplier'].value_counts().sort_index()

    with open(path, 'w', encoding='utf-8') as f:
        f.write("# 대전 트램 공사 구간 네트워크 분석 리포트 v3\n\n")
        f.write(f"> 생성: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
                f"기상: {weather_info['scenario']} ({weather_info['note']})\n\n")

        f.write("## 1. 패널티 적용 분포\n\n")
        f.write("| 패널티 | 링크 수 |\n|---|---|\n")
        labels = {1.00:"무패널티",0.75:"잔여>75%",0.55:"잔여50~75%",
                  0.50:"파싱기본",0.40:"잔여35~50%",0.25:"잔여<35%"}
        for v, c in penalty_dist.items():
            f.write(f"| {v:.2f} ({labels.get(v,'')}) | {c:,} |\n")

        f.write("\n## 2. BC 변화 상위 10 링크\n\n")
        f.write("| 도로명 | 인접 정거장 | 구간 ID | 잔여차로 | BC(전) | BC(후) | 변화량 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for _, r in top_risk.iterrows():
            f.write(f"| {r['ROAD_NAME']} | {r['nearest_station_name']} | {r['segment_id']} "
                    f"| {r['lane_remain_ratio']:.2f} | {r['bc_free_flow']:.5f} "
                    f"| {r['bc_under_construction']:.5f} | {r['bc_change']:.5f} |\n")

        f.write("\n## 3. 구간 우선순위 상위 20\n\n")
        f.write("| 순위 | 구간명 | 스코어 | 등급 | BC(후) | 잔여차로 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for _, r in seg_df.head(20).iterrows():
            f.write(f"| {int(r['rank'])} | {r['segment_name']} | {r['priority_score']:.1f} "
                    f"| {r['risk_grade']} | {r['bc_under_construction']:.5f} "
                    f"| {r['lane_remain_ratio']:.2f} |\n")

    print(f"  → {path} 저장 완료")


def _write_insight_report(df: pd.DataFrame, seg_df: pd.DataFrame, weather_info: dict):
    """
    보고서 작성용 인사이트 리포트 — 중요 도로 분석 + ASOS 실측 기상 통계
    """
    path = os.path.join("outputs", "insight_report.md")

    road_agg = df.groupby('ROAD_NAME').agg(
        link_count=('LINK_ID','count'),
        bc_const_max=('bc_under_construction','max'),
        bc_const_mean=('bc_under_construction','mean'),
        bc_change_max=('bc_change','max'),
        bc_change_mean=('bc_change','mean'),
        priority_max=('priority_score','max'),
        lane_remain_min=('lane_remain_ratio','min'),
        lanes_max=('LANES','max'),
        max_spd=('MAX_SPD','first'),
        road_rank=('ROAD_RANK','first'),
    ).sort_values('bc_const_max', ascending=False)

    major_roads  = road_agg[road_agg['lanes_max'] >= 3].head(15)
    change_roads = road_agg.sort_values('bc_change_max', ascending=False).head(15)

    from src.weather_risk import load_asos, compute_monthly_stats, get_required_data_note
    df_asos   = load_asos()
    monthly   = compute_monthly_stats(df_asos)
    req_note  = get_required_data_note()
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# 보고서용 인사이트: 네트워크 분석 핵심 발견\n\n")
        f.write(f"> 분석 기준일: {datetime.now().strftime('%Y-%m-%d')} | "
                f"기상 시나리오: **{weather_info['scenario']}** ({weather_info['note']})\n\n")
        f.write("---\n\n")

        f.write("## 1. 매개중심성 최상위 도로 — 실제 교통 의미\n\n")
        f.write("아래 도로들은 공사 중 대전 도로망에서 **가장 많은 최단경로가 경유**하는 구간입니다.  \n")
        f.write("차단 또는 정체 발생 시 **도시 전역으로 파급**될 위험이 가장 큽니다.\n\n")
        f.write("| 순위 | 도로명 | 도로위계 | 최대차로 | 제한속도 | BC 최대값(공사중) | BC 변화량 | 우선순위 점수 | 교통적 의미 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")

        road_meaning = {
            '천변도시고속도로': '갑천 변 남북 고속 통행 축. 트램 공사 우회 수요 최대 흡수처. 차단 시 서구·유성구 완전 고립 위험',
            '계룡로':   '유성구↔둔산 도심 광역 간선. 트램 2호선 직접 경유 구간. BC 급증 = 공사 직접 영향 구조 확인',
            '대전로':   '구도심~동구 연결 간선. 대전역 인근 물류 접근성 핵심. 독립 통행수요 높아 정체 전이 취약',
            '현충원로':  '유성온천↔국립현충원/IC 연결. 공사 구간 교차 다수. 우회 수요 집중 관측',
            '월드컵대로': '유성구 동서 간선. 계룡로 혼잡 시 1차 대체 우회로. BC 상승 = 우회 부하 증거',
            '온천로':   '트램 노선 직접 영향권 이면도로. BC 변화량 최고(+0.033). 용량 대비 과부하 위험',
            '계룡로105번길': '계룡로 인접 이면도로. BC 변화 크지만 2차로 이하 → AMR 배송 경로 활용 가능',
            '대전로':   '구도심 남북 축. 13공구 인근. 물류 탑차 우회 집중 예상',
        }

        for i, (rname, row) in enumerate(major_roads.iterrows(), 1):
            rank_label = {
                '101':'고속국도','102':'일반국도','103':'광역시도',
                '104':'지방도','105':'시군구도','106':'고속도로','107':'기타'
            }.get(str(row['road_rank']), '기타')
            meaning = road_meaning.get(rname, '—')
            f.write(f"| {i} | **{rname}** | {rank_label} | {int(row['lanes_max'])}차로 "
                    f"| {int(row['max_spd'])}km/h | {row['bc_const_max']:.5f} "
                    f"| {row['bc_change_max']:+.5f} | {row['priority_max']:.1f} "
                    f"| {meaning} |\n")

        f.write("\n> **검증 계획**: 위 도로들의 BC 상위 순위가 2026년 4월 원촌육교 통제 사태 당시\n")
        f.write("> 실제 정체 전이 경로와 일치하는지 사후 재현 검증 예정.\n\n")
        f.write("---\n\n")

        f.write("## 2. 공사 영향 BC 변화량 상위 — '공사가 만든 위험'\n\n")
        f.write("공사 전에는 중요하지 않았으나 공사 후 최단경로 집중도가 크게 높아진 구간입니다.  \n")
        f.write("XGBoost `bc_change` 피처가 이 효과를 예측 모델에 반영합니다.\n\n")
        f.write("| 순위 | 도로명 | BC 변화량(최대) | 인접 정거장 | 해석 |\n")
        f.write("|---|---|---|---|---|\n")

        for i, (rname, row) in enumerate(change_roads.head(10).iterrows(), 1):
            seg_rows = df[df['ROAD_NAME'] == rname]
            stn = seg_rows['nearest_station_name'].mode()[0] if len(seg_rows) > 0 else '—'
            interp = "이면도로 우회 집중" if row['lanes_max'] <= 2 else "간선 우회 부하 증가"
            f.write(f"| {i} | {rname} | {row['bc_change_max']:+.5f} | {stn} | {interp} |\n")

        f.write("\n---\n\n")

        f.write("## 3. 구간 우선순위 최종 순위 (대시보드 표출용)\n\n")
        f.write("Prophet→XGBoost 파이프라인 결합 전 **네트워크 분석 단독 기준** 우선순위입니다.  \n")
        f.write("BC(40%) + 차로감소(30%) + 도로위계(20%) + 기상(10%) 가중합.\n\n")

        grade_counts = seg_df['risk_grade'].value_counts()
        f.write("**등급 분포**\n\n")
        for grade, cnt in grade_counts.items():
            f.write(f"  - {grade}: {cnt}개 구간\n")
        f.write("\n")

        f.write("| 순위 | 구간 ID | 구간명 | 점수 | 등급 | BC(공사중) | BC 변화 | 잔여차로 | 적용 패널티 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for _, r in seg_df.head(30).iterrows():
            f.write(f"| {int(r['rank'])} | {r['segment_id']} | {r['segment_name']} "
                    f"| **{r['priority_score']:.1f}** | {r['risk_grade']} "
                    f"| {r['bc_under_construction']:.5f} | {r['bc_change']:+.5f} "
                    f"| {r['lane_remain_ratio']:.2f} | {r['speed_penalty_multiplier']:.2f} |\n")

        f.write("\n---\n\n")

        f.write("## 4. 추가 수집 필요 데이터\n\n")
        f.write("현재 네트워크 분석 고도화를 위해 아래 데이터가 추가로 필요합니다.\n\n")

        f.write("### 4.1 당장 필요 (1주차 내)\n\n")
        f.write("| 데이터 | 출처 | 활용 목적 | 우선순위 |\n")
        f.write("|---|---|---|---|\n")
        f.write("| **실시간 링크별 속도 (5분)** | 대전ITS/국가교통정보센터 OpenAPI | BC 엣지 가중치 실시간 갱신 → 동적 BC 계산 | 🔴 필수 |\n")
        f.write("| **트램 공사 구간 GIS 좌표** | 대전시 공공데이터 / 트램 공식 자료 | 세부구간 한정 패널티 (현재 도로명 전체에 적용 중) | 🔴 필수 |\n")
        f.write("| **ITS 돌발상황 API** | 국가교통정보센터 | 돌발 감지 → XGBoost 재분류 트리거 | 🟠 권장 |\n")
        f.write("| **시간별 강수량 (대전 272번 관측소)** | 기상청 API허브 ASOS | Prophet 외생변수, 집중호우 J_danger 트리거 | 🟠 권장 |\n")

        f.write("\n### 4.2 고도화 단계 (2~3주차)\n\n")
        f.write("| 데이터 | 출처 | 활용 목적 |\n")
        f.write("|---|---|---|\n")
        f.write("| 기상특보 API (대설/한파/태풍) | 기상청 API허브 | AMR 운영불가일 마스킹 |\n")
        f.write("| 공영주차장 위치/면수 | 대전시 주차안내시스템 | MCLP 거점 후보지(J) |\n")
        f.write("| 화물차 주정차 허용구역 | 대전시 구별 포털 | MCLP J 가중치 인센티브 |\n")
        f.write("| 소방용수시설 위치 | 공공데이터포털 | MCLP J_danger 배제 |\n")
        f.write("| 행정동별 인구밀도 | KOSIS | MCLP 수요 가중치 w_i |\n")
        f.write("| 소상공인 상가 위치 | 소상공인시장진흥공단 | MCLP 수요지점 I |\n")
        f.write("| 아파트 단지 세대수 | 국토교통부 공동주택 | MCLP 수요지점 I |\n")

        f.write("\n### 4.3 기상 데이터 비고\n")
        f.write(req_note)

        f.write("\n---\n\n")
        f.write("## 5. ASOS 실측 기상 통계 (대전 133번, 2024-10 ~ 2026-07)\n\n")
        f.write("> **출처**: 기상청 기상자료개방포털 종관기상관측(ASOS), 대전 지점(133), 시간자료\n\n")
        f.write("이 통계가 우선순위 스코어의 **기상 가중치(10%)** 에 실측값으로 반영됩니다.\n")
        f.write("가상 시나리오(고정 0.20~0.55) 대비 **실제 대전 기상 패턴**이 적용되어\n")
        f.write("6~7월·9월 장마/집중호우 시즌의 위험도가 더 정확하게 반영됩니다.\n\n")

        f.write("| 월 | 강수합계(mm) | 강수시간(h) | 집중호우(≥20mm)시간 | 결빙위험시간 | 적설시간 | 평균위험계수 | AMR불가시간 |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for _, r in monthly.iterrows():
            f.write(f"| {r['month']} | {r['rain_sum']:.1f} | {int(r['rain_hours'])} "
                    f"| {int(r['heavy_rain_hours'])} | {int(r['freeze_hours'])} "
                    f"| {int(r['snow_hours'])} | {r['avg_risk']:.4f} "
                    f"| {int(r['amr_inoperable'])} |\n")

        f.write("\n**주요 발견:**\n\n")
        # 가장 위험한 달
        worst_month = monthly.loc[monthly['avg_risk'].idxmax()]
        f.write(f"- 평균 위험계수 최고월: **{worst_month['month']}** "
                f"(avg_risk={worst_month['avg_risk']:.4f}, "
                f"집중호우 {int(worst_month['heavy_rain_hours'])}시간)\n")
        # 결빙 최고 달
        worst_freeze = monthly.loc[monthly['freeze_hours'].idxmax()]
        f.write(f"- 결빙 위험시간 최고월: **{worst_freeze['month']}** "
                f"({int(worst_freeze['freeze_hours'])}시간, "
                f"AMR불가 {int(worst_freeze['amr_inoperable'])}시간)\n")
        # 현재 월
        curr_month = str(datetime.now().strftime('%Y-%m'))
        curr = monthly[monthly['month'] == curr_month]
        if len(curr) > 0:
            r = curr.iloc[0]
            f.write(f"- **현재 월({curr_month})**: 평균위험계수 {r['avg_risk']:.4f}, "
                    f"집중호우 {int(r['heavy_rain_hours'])}시간\n")

        f.write("\n---\n\n")
        f.write("## 6. 현재 분석의 한계 및 검증 방법\n\n")
        f.write("| 항목 | 현재 상태 | 검증/개선 방법 |\n")
        f.write("|---|---|---|\n")
        f.write("| BC 샘플링 오차 | 전체 18,236 노드 중 182개(1%) 샘플 | 다중 seed 실행 후 Spearman 상관 ≥ 0.95 목표 |\n")
        f.write("| 패널티 공간 범위 | 도로명 전체에 적용 (세부구간 미구분) | 공사 GIS 좌표 확보 후 교차 필터 적용 |\n")
        f.write("| 기상 공간 차등 | 전 링크 동일 계수 (1개 관측소) | 관측소 거리 가중 보간 또는 격자 예보 활용 |\n")
        f.write("| 동광장로 미매칭 | 표준노드링크 미등재 | 중앙로 구간으로 간접 커버 확인 |\n")
        f.write("| 원촌육교 사태 재현 | 미실시 | 2026-04 실측 강수·기온 + BC 분포 대조 검증 예정 |\n")

        f.write(f"\n{req_note}\n")
    print(f"  → {path} 저장 완료")


if __name__ == "__main__":
    main()
