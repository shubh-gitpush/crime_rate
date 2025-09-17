import React, { useState } from "react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

delete L.Icon.Default.prototype._getIconUrl;

L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});


function App() {
  const [result, setResult] = useState(null);
  const [location, setLocation] = useState({ 
    lat: 32.0, 
    lon: 109.0, 
    hour: 12,
    prediction_level: "district" 
  });
  const [predictionType, setPredictionType] = useState("district");

  async function checkSafety() {
    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(location),
    });
    const data = await res.json();
    setResult(data);
  }

  return (
    <div style={{ textAlign: "center", padding: "20px", background: "#f6f8fb", minHeight: "100vh" }}>
      <style>
        {`
          .panel {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 20px;
          }
          .controls {
            display: grid;
            gap: 20px;
            margin-bottom: 20px;
          }
          .control {
            text-align: left;
          }
          .control label {
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
            color: #666;
          }
          .slider-row {
            display: flex;
            align-items: center;
            gap: 10px;
          }
          .slider-row input {
            flex: 1;
          }
          .coord {
            display: flex;
            gap: 20px;
            color: #666;
          }
          .btn {
            padding: 8px 16px;
            border: 1px solid #ddd;
            border-radius: 6px;
            background: white;
            cursor: pointer;
            transition: all 0.2s;
          }
          .btn:hover {
            background: #f5f5f5;
          }
          .btn.active {
            background: #e0e0e0;
            border-color: #ccc;
          }
          .btn.primary {
            background: #007bff;
            color: white;
            border: none;
          }
          .btn.primary:hover {
            background: #0069d9;
          }
          .summary {
            text-align: left;
            padding-top: 20px;
            border-top: 1px solid #eee;
          }
          .badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: bold;
            margin-bottom: 10px;
          }
          .badge.safe { background: #d4edda; color: #155724; }
          .badge.average { background: #fff3cd; color: #856404; }
          .badge.danger { background: #f8d7da; color: #721c24; }
          .meta {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 15px;
          }
          .chip {
            background: #e9ecef;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
            color: #495057;
          }
          .chip.danger { background: #f8d7da; color: #721c24; }
          .chip.warning { background: #fff3cd; color: #856404; }
          .chip.safe { background: #d4edda; color: #155724; }
          .progress {
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 15px;
          }
          .progress .bar {
            height: 100%;
            background: #28a745;
            transition: width 0.3s ease;
          }
          .progress .bar.warning { background: #ffc107; }
          .progress .bar.danger { background: #dc3545; }
          .recommendations {
            margin-top: 15px;
            color: #666;
          }
          .recommendation {
            margin: 5px 0;
            font-size: 0.9em;
          }
          .warning-message {
            margin-top: 10px;
            padding: 8px;
            background: #fff3cd;
            border: 1px solid #ffeeba;
            border-radius: 4px;
            color: #856404;
            font-size: 0.9em;
          }
        `}
      </style>
      <h1 style={{ marginTop: 0 }}>Smart Tourist Safety</h1>

      <div className="panel">
        <div className="controls">
          <div className="control">
            <label>Time of Day</label>
            <div className="slider-row">
              <input
                type="range"
                min="0"
                max="23"
                value={location.hour}
                onChange={(e) =>
                  setLocation((prev) => ({ ...prev, hour: parseInt(e.target.value, 10) }))
                }
              />
              <span className="value">{location.hour}:00</span>
            </div>
          </div>

          <div className="control">
            <label>Prediction Level</label>
            <div style={{ display: 'flex', gap: '10px', marginTop: '5px' }}>
              <button
                className={`btn ${predictionType === 'state' ? 'active' : ''}`}
                onClick={() => {
                  setPredictionType('state');
                  setLocation(prev => ({ ...prev, prediction_level: 'state' }));
                }}
                style={{ flex: 1 }}
              >
                State
              </button>
              <button
                className={`btn ${predictionType === 'district' ? 'active' : ''}`}
                onClick={() => {
                  setPredictionType('district');
                  setLocation(prev => ({ ...prev, prediction_level: 'district' }));
                }}
                style={{ flex: 1 }}
              >
                District
              </button>
            </div>
          </div>

          <div className="control readonly">
            <label>Coordinates</label>
            <div className="coord">
              <span>Lat: {location.lat.toFixed(4)}</span>
              <span>Lon: {location.lon.toFixed(4)}</span>
            </div>
          </div>

          <button className="btn primary" onClick={checkSafety}>Check Safety</button>
        </div>

        {result && (
          <div className="summary">
            <span className={`badge ${result.status === "risky" ? "danger" : result.status === "average" ? "average" : "safe"}`}>
              {result.status.toUpperCase()}
            </span>
            <div className="meta">
              <span className="chip">State: {result.state}</span>
              {result.district && <span className="chip">District: {result.district}</span>}
              <span className="chip">Score: {result.score}</span>
              {result.time_risk && (
                <span className={`chip ${result.time_risk === 'high' ? 'danger' : result.time_risk === 'moderate' ? 'warning' : 'safe'}`}>
                  Time Risk: {result.time_risk}
                </span>
              )}
            </div>
            <div className="progress">
              <div 
                className={`bar ${result.score > 70 ? 'danger' : result.score > 40 ? 'warning' : 'safe'}`} 
                style={{ width: `${result.score}%` }} 
              />
            </div>
            {result.recommendations && (
              <div className="recommendations">
                {result.recommendations.filter(Boolean).map((rec, idx) => (
                  <div key={idx} className="recommendation">
                    • {rec}
                  </div>
                ))}
              </div>
            )}
            {result.warning && (
              <div className="warning-message">
                {result.warning}
              </div>
            )}
          </div>
        )}
      </div>

      <div style={{ marginTop: "20px" }}>
        <MapContainer
          center={[location.lat, location.lon]}
          zoom={13}
          style={{ height: "520px", width: "100%", borderRadius: 12, overflow: "hidden", boxShadow: "0 6px 18px rgba(0,0,0,0.08)" }}
          whenReady={(map) => {
            map.target.setView([location.lat, location.lon]);
          }}
          onclick={(e) => {
            setLocation((prev) => ({ ...prev, lat: e.latlng.lat, lon: e.latlng.lng }));
          }}
        >
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          />
          <Marker
            position={[location.lat, location.lon]}
            draggable={true}
            eventHandlers={{
              dragend: (e) => {
                const { lat, lng } = e.target.getLatLng();
                setLocation((prev) => ({ ...prev, lat: lat, lon: lng }));
              },
            }}
          >
            <Popup>
              <div style={{ fontSize: '14px' }}>
                <strong>Location Details</strong>
                <div style={{ margin: '8px 0' }}>
                  <div>📍 {result?.state || 'Unknown State'}</div>
                  {result?.district && <div>🏘️ District: {result.district}</div>}
                  <div>🕒 Time: {location.hour}:00</div>
                  <div>📌 Lat: {location.lat.toFixed(5)}, Lon: {location.lon.toFixed(5)}</div>
                </div>
                {result ? (
                  <div style={{ marginTop: '8px' }}>
                    <strong>Safety Analysis</strong>
                    <div style={{
                      padding: '4px 8px',
                      margin: '4px 0',
                      borderRadius: '4px',
                      background: result.status === 'risky' ? '#f8d7da' : 
                                result.status === 'average' ? '#fff3cd' : '#d4edda',
                      color: result.status === 'risky' ? '#721c24' : 
                             result.status === 'average' ? '#856404' : '#155724'
                    }}>
                      Status: {result.status.toUpperCase()}
                    </div>
                    <div>Safety Score: {result.score}%</div>
                    {result.time_risk && <div>Time Risk: {result.time_risk}</div>}
                  </div>
                ) : (
                  <div style={{ color: '#666' }}>
                    Click "Check Safety" to analyze this location
                  </div>
                )}
              </div>
            </Popup>
          </Marker>
        </MapContainer>
      </div>
    </div>
  );
}

export default App;
