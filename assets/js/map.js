// 공통 Leaflet 지도 유틸 (구간 색칠, 공구 롤업, 팝업, 적색 자동확대)
(function () {
  const RISK_COLOR = { "정상": "#2ecc71", "주의": "#f1c40f", "심각": "#e74c3c" };
  const DAEJEON_CENTER = [36.3504, 127.3845];

  function initMap(containerId, opts) {
    const map = L.map(containerId, { scrollWheelZoom: true }).setView(
      (opts && opts.center) || DAEJEON_CENTER,
      (opts && opts.zoom) || 12
    );
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
      maxZoom: 19,
    }).addTo(map);
    return map;
  }

  function riskColor(cls) {
    return RISK_COLOR[cls] || "#7f8c8d";
  }

  // segments: segments.json 의 segments 배열, snapshotByKey: segment_key -> snapshot row
  // onClick(segment, worstState) 콜백
  function drawSegments(map, segments, snapshotByKey, onClick) {
    const layers = {};
    segments.forEach((seg) => {
      const states = Object.values(seg.directions)
        .map((key) => snapshotByKey[key])
        .filter(Boolean);
      const worst = states.reduce((acc, s) => {
        const rank = { "정상": 0, "주의": 1, "심각": 2 };
        return !acc || rank[s.predicted_class] > rank[acc.predicted_class] ? s : acc;
      }, null);
      const cls = worst ? worst.predicted_class : "정상";
      const latlngs = seg.geometry.coordinates.map((line) => line.map(([lng, lat]) => [lat, lng]));
      const layer = L.polyline(latlngs, {
        color: riskColor(cls),
        weight: cls === "심각" ? 6 : cls === "주의" ? 5 : 3.5,
        opacity: 0.9,
      }).addTo(map);
      layer.on("click", () => onClick && onClick(seg, worst));
      layers[seg.segment_id] = { layer, cls };
    });
    return layers;
  }

  function zoomToSevere(map, layerMap) {
    const severeLayers = Object.values(layerMap).filter((l) => l.cls === "심각").map((l) => l.layer);
    if (severeLayers.length === 0) return;
    const group = L.featureGroup(severeLayers);
    map.fitBounds(group.getBounds().pad(0.3));
  }

  window.AppMap = { initMap, riskColor, drawSegments, zoomToSevere, DAEJEON_CENTER };
})();
