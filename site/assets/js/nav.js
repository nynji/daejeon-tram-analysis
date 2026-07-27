// 공통 좌측 네비게이션 + 계정유형 랜딩 분기(REQ-H1)
(function () {
  const NAV_ITEMS = [
    { id: "home", label: "🏠 Home", href: "BASE/index.html", group: "" },
    { id: "1-1", label: "1-1. 예측 & 임계도달", href: "BASE/analysis/1-1.html", group: "📊 분석 뷰" },
    { id: "1-2", label: "1-2. 네트워크 구조 분석", href: "BASE/analysis/1-2.html", group: "📊 분석 뷰" },
    { id: "1-3", label: "1-3. 정체 패턴 군집", href: "BASE/analysis/1-3.html", group: "📊 분석 뷰" },
    { id: "2-1", label: "2-1. 실시간 병목 경보", href: "BASE/control/2-1.html", group: "🚨 관제 뷰" },
    { id: "2-2", label: "2-2. 우회 경로 추천", href: "BASE/control/2-2.html", group: "🚨 관제 뷰" },
    { id: "2-3", label: "2-3. AMR 거점 배치 관제", href: "BASE/control/2-3.html", group: "🚨 관제 뷰" },
    { id: "report", label: "🧭 공구별 리포트", href: "BASE/report/report.html", group: "" },
  ];

  function currentRole() {
    return localStorage.getItem("dashboard_role") || "관제요원";
  }

  function setRole(role) {
    localStorage.setItem("dashboard_role", role);
  }

  function renderNav(activeId, base) {
    const root = document.getElementById("sidebar-root");
    if (!root) return;

    let groupsHtml = "";
    let lastGroup = "__init__";
    for (const item of NAV_ITEMS) {
      if (item.group !== lastGroup && item.group) {
        groupsHtml += `<div class="nav-group-label">${item.group}</div>`;
        lastGroup = item.group;
      }
      const href = item.href.replace("BASE", base);
      const activeCls = item.id === activeId ? " active" : "";
      groupsHtml += `<a class="nav-item${activeCls}" href="${href}">${item.label}</a>`;
    }

    root.innerHTML = `
      <div class="sidebar">
        <div class="brand">트램 공사 통합 관제<small>대전 도시철도 트램 2호선</small></div>
        ${groupsHtml}
        <div class="role-toggle">
          현재 로그인 계정 유형(시뮬레이션)
          <select id="role-select">
            <option value="관제요원">관제 요원</option>
            <option value="관리자">관리자</option>
          </select>
        </div>
      </div>`;

    const select = document.getElementById("role-select");
    select.value = currentRole();
    select.addEventListener("change", (e) => {
      setRole(e.target.value);
      if (e.target.value === "관제요원") {
        window.location.href = `${base}/control/2-1.html`;
      } else {
        window.location.href = `${base}/index.html`;
      }
    });
  }

  // REQ-H1: Home 진입 시 관제요원 계정이면 2-1로 즉시 리다이렉트
  function applyLandingRedirect(base) {
    const isHome = /\/index\.html$/.test(window.location.pathname) || window.location.pathname.endsWith("/site/") || window.location.pathname.endsWith("/site");
    if (isHome && currentRole() === "관제요원" && !window.location.search.includes("stay=1")) {
      window.location.replace(`${base}/control/2-1.html`);
    }
  }

  window.AppNav = { renderNav, applyLandingRedirect, currentRole, setRole };
})();
