import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function analyze(event) {
    event.preventDefault();
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    const formData = new FormData();
    formData.append("image", file);
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Prediction failed");
      setResult(payload);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <header>
        <p className="eyebrow">URBAN SKIN / FIRST IDEA</p>
        <h1>Visual health check for infrastructure</h1>
        <p className="subtitle">
          Upload an inspection image and translate visible damage into an
          explainable condition score.
        </p>
      </header>
      <section className="card">
        <form onSubmit={analyze}>
          <label className="dropzone">
            <span>{file ? file.name : "Choose a road, bridge, or facade image"}</span>
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              onChange={(event) => setFile(event.target.files?.[0] || null)}
            />
          </label>
          <button type="submit" disabled={!file || loading}>
            {loading ? "Analyzing..." : "Analyze image"}
          </button>
        </form>
        {error && <p className="error">{error}</p>}
        {result && (
          <div className="result">
            <div>
              <span className="label">Condition score</span>
              <strong>{result.condition_score}/100</strong>
            </div>
            <div>
              <span className="label">Priority</span>
              <strong className={`priority ${result.priority}`}>{result.priority}</strong>
            </div>
            <div>
              <span className="label">Inference latency</span>
              <strong>{result.latency_ms} ms</strong>
            </div>
          </div>
        )}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
