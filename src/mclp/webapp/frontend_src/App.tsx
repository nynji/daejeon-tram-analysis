import React, { useState, useEffect, useCallback, useRef } from 'react'
import Map, { Marker, Source, Layer, NavigationControl, MapRef } from 'react-map-gl/maplibre'
import 'maplibre-gl/dist/maplibre-gl.css'

interface Anchor {
  idx: number
  name: string
  lat: number
  lon: number
  capacity: number
  assigned_demands: number
  avg_distance_m: number
}

interface OptResult {
  status: string
  time_ms: number
  p_selected: number
  wcr: number
  ucr: number
  obj: number
  anchors: any[]
  covered: number[][]
  uncovered: number[][]
  routes: any[]
  truck_routes: any[]
  total: number
  covered_n: number
  districts: Record<string, number>
}

interface ConstructionZone {
  name: string
  start_lat: number | null
  start_lon: number | null
  end_lat: number | null
  end_lon: number | null
}

interface TramStation {
  no: number
  name: string
  lat: number
  lon: number
}

const DAEJEON_CENTER = { latitude: 36.35, longitude: 127.385, zoom: 11.5 }

export default function App() {
  const mapRef = useRef<MapRef>(null)

  const [p, setP] = useState(10)
  const [radius, setRadius] = useState(2500)
  const [beta, setBeta] = useState(0.10)
  const [dMin, setDMin] = useState(300)
  const [weightMode, setWeightMode] = useState('기준')
  const [amrSpeed, setAmrSpeed] = useState(10)
  const [weather, setWeather] = useState('정상')
  const [timeStart] = useState('09:00')
  const [timeEnd] = useState('18:00')
  const [incidentMode, setIncidentMode] = useState(false)
  const [incidentLat, setIncidentLat] = useState<number | null>(null)
  const [incidentLon, setIncidentLon] = useState<number | null>(null)
  const [incidentRadius, setIncidentRadius] = useState(500)

  const [result, setResult] = useState<OptResult | null>(null)
  const [zones, setZones] = useState<any>(null)
  const [stations, setStations] = useState<TramStation[]>([])
  const [tramLine, setTramLine] = useState<any>(null)
  const [roadNetwork, setRoadNetwork] = useState<any>(null)
  const [boundary, setBoundary] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [showCovered, setShowCovered] = useState(true)
  const [showUncovered, setShowUncovered] = useState(false)
  const [showZones, setShowZones] = useState(true)
  const [showTram, setShowTram] = useState(true)
  const [showRoads, setShowRoads] = useState(true)
  const [showRoutes, setShowRoutes] = useState(true)
  const [showTrucks, setShowTrucks] = useState(true)
  const [warehouses, setWarehouses] = useState<any[]>([])

  // Load static data
  useEffect(() => {
    fetch('/api/construction_zones').then(r => r.json()).then(setZones).catch(() => {})
    fetch('/api/tram_stations').then(r => r.json()).then(setStations).catch(() => {})
    fetch('/api/tram_line').then(r => r.json()).then(setTramLine).catch(() => {})
    fetch('/api/road_network?rank=major').then(r => r.json()).then(setRoadNetwork).catch(() => {})
    fetch('/api/boundary').then(r => r.json()).then(setBoundary).catch(() => {})
    fetch('/api/warehouses').then(r => r.json()).then(setWarehouses).catch(() => {})
  }, [])

  const runOptimize = useCallback(async () => {
    setLoading(true)
    try {
      const body: any = {
        p, radius_m: radius, beta, d_min_m: dMin,
        weight_mode: weightMode, amr_speed_kmh: amrSpeed,
        weather, time_start: timeStart, time_end: timeEnd,
        incident_radius_m: incidentRadius,
        forced_on: [], forced_off: [], disabled_zones: [],
      }
      if (incidentLat !== null && incidentLon !== null) {
        body.incident_lat = incidentLat
        body.incident_lon = incidentLon
      }
      const res = await fetch('/api/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await res.json()
      setResult(data)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }, [p, radius, beta, dMin, weightMode, amrSpeed, weather, timeStart, timeEnd, incidentLat, incidentLon, incidentRadius])

  // Don't auto-optimize on mount — show precomputed results or wait for user click
  useEffect(() => {
    // Load precomputed results from teammate's output as default view
    fetch('/api/precomputed_results')
      .then(r => r.json())
      .then(data => {
        if (data.scenarios && data.scenarios.length > 0) {
          // Find the BASE scenario with P=15
          const base = data.scenarios.find((s: any) => 
            s.scenario_name && s.scenario_name.includes('기본조건') && s.anchor_limit_p === 15 && s.status === 'Optimal'
          )
          if (base) {
            // Just show status, user clicks Run to get live results
          }
        }
      })
      .catch(() => {})
  }, [])

  // Map click for incident
  const handleMapClick = (e: any) => {
    if (incidentMode && e.lngLat) {
      setIncidentLat(e.lngLat.lat)
      setIncidentLon(e.lngLat.lng)
    }
  }

  // GeoJSON for covered demands
  const coveredGeoJSON = result ? {
    type: 'FeatureCollection' as const,
    features: result.covered.slice(0, 800).map(([lat, lon]) => ({
      type: 'Feature' as const,
      geometry: { type: 'Point' as const, coordinates: [lon, lat] },
      properties: {},
    }))
  } : null

  const uncoveredGeoJSON = result ? {
    type: 'FeatureCollection' as const,
    features: result.uncovered.slice(0, 300).map(([lat, lon]) => ({
      type: 'Feature' as const,
      geometry: { type: 'Point' as const, coordinates: [lon, lat] },
      properties: {},
    }))
  } : null

  // Construction zone lines — now loaded as GeoJSON from API directly

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>대전 트램 물류 관제</h1>
          <p>TD-Risk-CMCLP Real-time Optimization</p>
        </div>

        {result && (
          <div className="section">
            <div className="section-title">Performance Metrics</div>
            <div className="kpi-grid">
              <div className="kpi-card">
                <div className="label">Weighted Coverage</div>
                <div className={`value ${result.wcr >= 0.8 ? 'success' : result.wcr >= 0.6 ? 'warning' : 'danger'}`}>
                  {(result.wcr * 100).toFixed(1)}%
                </div>
              </div>
              <div className="kpi-card">
                <div className="label">Unweighted Coverage</div>
                <div className={`value ${result.ucr >= 0.8 ? 'success' : result.ucr >= 0.6 ? 'warning' : 'danger'}`}>
                  {(result.ucr * 100).toFixed(1)}%
                </div>
              </div>
              <div className="kpi-card">
                <div className="label">Active Anchors</div>
                <div className="value">{result.p_selected}</div>
              </div>
              <div className="kpi-card">
                <div className="label">Solve Time</div>
                <div className="value success">{result.time_ms < 1000 ? `${result.time_ms.toFixed(0)}ms` : `${(result.time_ms/1000).toFixed(1)}s`}</div>
              </div>
            </div>
          </div>
        )}

        <div className="section">
          <div className="section-title">Optimization Parameters</div>
          <div className="control-row">
            <span className="control-label">Anchor count (p)</span>
            <input type="range" min={1} max={20} value={p} onChange={e => setP(+e.target.value)} />
            <span className="control-value">{p}</span>
          </div>
          <div className="control-row">
            <span className="control-label">Coverage radius</span>
            <select value={radius} onChange={e => setRadius(+e.target.value)}>
              <option value={2000}>2,000m</option>
              <option value={2500}>2,500m</option>
              <option value={3000}>3,000m</option>
            </select>
          </div>
          <div className="control-row">
            <span className="control-label">Risk penalty</span>
            <input type="range" min={0} max={30} value={beta * 100} onChange={e => setBeta(+e.target.value / 100)} />
            <span className="control-value">{beta.toFixed(2)}</span>
          </div>
          <div className="control-row">
            <span className="control-label">Min spacing</span>
            <input type="range" min={0} max={800} step={50} value={dMin} onChange={e => setDMin(+e.target.value)} />
            <span className="control-value">{dMin}m</span>
          </div>
          <div className="control-row">
            <span className="control-label">AMR speed</span>
            <input type="range" min={3} max={15} value={amrSpeed} onChange={e => setAmrSpeed(+e.target.value)} />
            <span className="control-value">{amrSpeed}km/h</span>
          </div>
          <div className="control-row">
            <span className="control-label">Weight mode</span>
            <select value={weightMode} onChange={e => setWeightMode(e.target.value)}>
              <option value="보수">Conservative</option>
              <option value="기준">Baseline</option>
              <option value="강화">Enhanced</option>
            </select>
          </div>
          <div className="control-row">
            <span className="control-label">Weather</span>
            <select value={weather} onChange={e => setWeather(e.target.value)}>
              <option value="정상">Normal</option>
              <option value="강우">Rain</option>
              <option value="적설">Snow (Halt)</option>
              <option value="한파">Cold (Halt)</option>
            </select>
          </div>
          <button className="btn-optimize" onClick={runOptimize} disabled={loading}>
            {loading ? 'Optimizing...' : 'Run Optimization'}
          </button>
        </div>

        <div className="section">
          <div className="section-title">Incident Simulation</div>
          <div
            className={`incident-toggle ${incidentMode ? 'active' : ''}`}
            onClick={() => {
              setIncidentMode(!incidentMode)
              if (incidentMode) { setIncidentLat(null); setIncidentLon(null) }
            }}
          >
            <span style={{width:8,height:8,borderRadius:'50%',background: incidentMode ? '#ef4444' : '#6b7280', flexShrink:0}}></span>
            {incidentMode ? 'Incident mode active — click map to place' : 'Enable incident simulation'}
          </div>
          {incidentMode && (
            <div className="control-row" style={{marginTop: 8}}>
              <span className="control-label">Impact radius</span>
              <input type="range" min={100} max={2000} step={50} value={incidentRadius} onChange={e => setIncidentRadius(+e.target.value)} />
              <span className="control-value">{incidentRadius}m</span>
            </div>
          )}
          {incidentLat && (
            <p style={{fontSize:11,color:'var(--text-secondary)',marginTop:4}}>
              Incident at ({incidentLat.toFixed(4)}, {incidentLon?.toFixed(4)})
            </p>
          )}
        </div>

        <div className="section">
          <div className="section-title">Map Layers</div>
          <label style={{display:'flex',alignItems:'center',gap:8,fontSize:13,color:'var(--text-secondary)',marginBottom:6,cursor:'pointer'}}>
            <input type="checkbox" checked={showTram} onChange={e => setShowTram(e.target.checked)} />
            Tram line / stations
          </label>
          <label style={{display:'flex',alignItems:'center',gap:8,fontSize:13,color:'var(--text-secondary)',marginBottom:6,cursor:'pointer'}}>
            <input type="checkbox" checked={showRoads} onChange={e => setShowRoads(e.target.checked)} />
            Major roads (arterial)
          </label>
          <label style={{display:'flex',alignItems:'center',gap:8,fontSize:13,color:'var(--text-secondary)',marginBottom:6,cursor:'pointer'}}>
            <input type="checkbox" checked={showRoutes} onChange={e => setShowRoutes(e.target.checked)} />
            AMR delivery routes
          </label>
          <label style={{display:'flex',alignItems:'center',gap:8,fontSize:13,color:'var(--text-secondary)',marginBottom:6,cursor:'pointer'}}>
            <input type="checkbox" checked={showTrucks} onChange={e => setShowTrucks(e.target.checked)} />
            Truck routes (warehouse to anchor)
          </label>
          <label style={{display:'flex',alignItems:'center',gap:8,fontSize:13,color:'var(--text-secondary)',marginBottom:6,cursor:'pointer'}}>
            <input type="checkbox" checked={showCovered} onChange={e => setShowCovered(e.target.checked)} />
            Covered demands
          </label>
          <label style={{display:'flex',alignItems:'center',gap:8,fontSize:13,color:'var(--text-secondary)',marginBottom:6,cursor:'pointer'}}>
            <input type="checkbox" checked={showUncovered} onChange={e => setShowUncovered(e.target.checked)} />
            Uncovered demands
          </label>
          <label style={{display:'flex',alignItems:'center',gap:8,fontSize:13,color:'var(--text-secondary)',cursor:'pointer'}}>
            <input type="checkbox" checked={showZones} onChange={e => setShowZones(e.target.checked)} />
            Construction zones
          </label>
        </div>

        {result && result.anchors.length > 0 && (
          <div className="section">
            <div className="section-title">Selected Anchors ({result.anchors.length})</div>
            <ul className="anchor-list">
              {result.anchors.sort((a: any, b: any) => b.assigned_demands - a.assigned_demands).map((anchor: any, i: number) => (
                <li key={anchor.candidate_id || i} className="anchor-item">
                  <span className="anchor-rank">{i + 1}</span>
                  <div className="anchor-info">
                    <div className="anchor-name">{anchor.name}</div>
                    <div className="anchor-meta">
                      {anchor.assigned_demands} assigned / {anchor.district}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}
      </aside>

      <main className="map-container">
        <div className="status-bar">
          <span className={`status-dot ${loading ? 'loading' : ''}`}></span>
          <span>{loading ? 'Computing...' : result ? `${result.status} / ${result.time_ms.toFixed(0)}ms` : 'Ready'}</span>
        </div>

        <Map
          ref={mapRef}
          initialViewState={DAEJEON_CENTER}
          style={{ width: '100%', height: '100%' }}
          mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
          onClick={handleMapClick}
        >
          <NavigationControl position="bottom-right" />

          {/* Daejeon city boundary */}
          {boundary && (
            <Source id="boundary" type="geojson" data={boundary}>
              <Layer id="boundary-line" type="line" paint={{
                'line-color': '#6366f1',
                'line-width': 2,
                'line-opacity': 0.6,
                'line-dasharray': [4, 2],
              }} />
            </Source>
          )}

          {/* Major road network */}
          {showRoads && roadNetwork && (
            <Source id="roads" type="geojson" data={roadNetwork}>
              <Layer id="roads-layer" type="line" paint={{
                'line-color': '#374151',
                'line-width': 1.5,
                'line-opacity': 0.6,
              }} />
            </Source>
          )}

          {/* Tram line */}
          {showTram && tramLine && (
            <Source id="tram-line" type="geojson" data={tramLine}>
              <Layer id="tram-line-layer" type="line" paint={{
                'line-color': '#8b5cf6',
                'line-width': 3,
                'line-opacity': 0.9,
              }} />
            </Source>
          )}

          {/* Tram stations */}
          {showTram && stations.map(st => (
            <Marker key={st.no} latitude={st.lat} longitude={st.lon} anchor="center">
              <div title={`${st.no} ${st.name}`} style={{
                width: 10, height: 10, borderRadius: '50%',
                background: '#8b5cf6', border: '1.5px solid #c4b5fd',
                cursor: 'pointer',
              }} />
            </Marker>
          ))}

          {/* Covered demands */}
          {showCovered && coveredGeoJSON && (
            <Source id="covered" type="geojson" data={coveredGeoJSON}>
              <Layer id="covered-layer" type="circle" paint={{
                'circle-radius': 4,
                'circle-color': '#3b82f6',
                'circle-opacity': 0.6,
              }} />
            </Source>
          )}

          {/* Uncovered demands */}
          {showUncovered && uncoveredGeoJSON && (
            <Source id="uncovered" type="geojson" data={uncoveredGeoJSON}>
              <Layer id="uncovered-layer" type="circle" paint={{
                'circle-radius': 3,
                'circle-color': '#6b7280',
                'circle-opacity': 0.4,
              }} />
            </Source>
          )}

          {/* Construction zones — actual road geometry */}
          {showZones && zones && (
            <Source id="zones" type="geojson" data={zones}>
              <Layer id="zones-line-layer" type="line" paint={{
                'line-color': '#f59e0b',
                'line-width': 3.5,
                'line-opacity': 0.75,
              }} />
            </Source>
          )}

          {/* Anchor markers + radius circles */}
          {result && result.anchors.map((anchor: any, i: number) => (
            <React.Fragment key={anchor.candidate_id || i}>
              <Marker latitude={anchor.lat} longitude={anchor.lon} anchor="center">
                <div style={{
                  width: 28, height: 28, borderRadius: '50%',
                  background: '#ef4444', border: '2px solid #fff',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 11, fontWeight: 700, color: '#fff',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.4)',
                }}>
                  {i + 1}
                </div>
              </Marker>
            </React.Fragment>
          ))}

          {/* Coverage radius circles (using geojson) */}
          {result && (
            <Source id="radius-circles" type="geojson" data={{
              type: 'FeatureCollection',
              features: result.anchors.map((a: any) => ({
                type: 'Feature' as const,
                geometry: {
                  type: 'Point' as const,
                  coordinates: [a.lon, a.lat],
                },
                properties: {},
              }))
            }}>
              <Layer id="radius-layer" type="circle" paint={{
                'circle-radius': ['interpolate', ['linear'], ['zoom'], 10, 20, 14, 120],
                'circle-color': '#ef4444',
                'circle-opacity': 0.08,
                'circle-stroke-color': '#ef4444',
                'circle-stroke-width': 1,
                'circle-stroke-opacity': 0.3,
              }} />
            </Source>
          )}

          {/* AMR delivery routes (anchor → demand) */}
          {showRoutes && result && result.routes && result.routes.length > 0 && (
            <Source id="amr-routes" type="geojson" data={{
              type: 'FeatureCollection',
              features: result.routes.map((route: any, idx: number) => ({
                type: 'Feature' as const,
                geometry: { type: 'LineString' as const, coordinates: [route.from, route.to] },
                properties: { idx },
              }))
            }}>
              <Layer id="amr-routes-layer" type="line" paint={{
                'line-color': '#10b981',
                'line-width': 1,
                'line-opacity': 0.3,
              }} />
            </Source>
          )}

          {/* Truck routes (warehouse → selected anchor) */}
          {showTrucks && result && result.truck_routes && result.truck_routes.length > 0 && (
            <Source id="truck-routes" type="geojson" data={{
              type: 'FeatureCollection',
              features: result.truck_routes.map((route: any, idx: number) => ({
                type: 'Feature' as const,
                geometry: { type: 'LineString' as const, coordinates: [route.from, route.to] },
                properties: { warehouse: route.warehouse },
              }))
            }}>
              <Layer id="truck-routes-layer" type="line" paint={{
                'line-color': '#f97316',
                'line-width': 2.5,
                'line-opacity': 0.7,
                'line-dasharray': [6, 3],
              }} />
            </Source>
          )}

          {/* Warehouse markers */}
          {showTrucks && warehouses.map((wh: any, i: number) => (
            <Marker key={`wh-${i}`} latitude={wh.lat} longitude={wh.lon} anchor="center">
              <div title={wh.name} style={{
                width: 22, height: 22, borderRadius: 4,
                background: '#f97316', border: '2px solid #fff',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 10, fontWeight: 700, color: '#fff',
                boxShadow: '0 2px 6px rgba(0,0,0,0.4)',
              }}>W</div>
            </Marker>
          ))}

          {/* Incident marker */}
          {incidentLat && incidentLon && (
            <Marker latitude={incidentLat} longitude={incidentLon} anchor="center">
              <div style={{
                width: 20, height: 20, borderRadius: '50%',
                background: '#ef4444', border: '3px solid #fff',
                boxShadow: '0 0 12px rgba(239,68,68,0.6)',
                animation: 'pulse 1.5s infinite',
              }} />
            </Marker>
          )}
        </Map>
      </main>
    </div>
  )
}
