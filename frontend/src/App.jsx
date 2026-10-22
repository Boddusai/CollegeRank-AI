
import React, { useState } from "react";

const API = "http://127.0.0.1:8000";

const EXAMS = [
  "JEE Main",
  "JEE Advanced",
  "NEET UG",
  "TG EAPCET",
  "AP EAPCET",
  "MHT CET",
  "KCET",
  "COMEDK UGET",
  "BITSAT",
  "WBJEE",
];

const CATEGORIES = ["OPEN", "EWS", "OBC-NCL", "SC", "ST"];

const initialForm = {
  exam: "JEE Main",
  rank: "",
  category: "OPEN",
  gender: "Gender-Neutral",
  state: "Telangana",
  quota: "AI",
  preferred_branch: "Computer Science",
  round: 2,
};

export default function App() {
  const [form, setForm] = useState(initialForm);
  const [results, setResults] = useState([]);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [searched, setSearched] = useState(false);

  const supported = ["JEE Main", "JEE Advanced"].includes(form.exam);

  function update(event) {
    const { name, value } = event.target;

    setForm((previous) => ({
      ...previous,
      [name]: value,
      ...(name === "exam" ? { category: "OPEN" } : {}),
    }));

    if (name === "exam") {
      setResults([]);
      setMessage("");
      setError("");
      setSearched(false);
    }
  }

  async function predict(event) {
    event.preventDefault();
    setError("");
    setMessage("");
    setResults([]);
    setSearched(true);

    if (!supported) {
      setMessage(
        `${form.exam} is not implemented yet. Choose JEE Main or JEE Advanced.`
      );
      return;
    }

    const rank = Number(form.rank);

    if (!Number.isInteger(rank) || rank < 1) {
      setError("Enter a valid positive integer rank.");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(`${API}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form,
          rank,
          round: 2,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "The prediction request could not be completed."
        );
      }

      setResults(data.predictions || []);
      setMessage(data.message || "Prediction request completed.");

      if (!data.model_available) {
        setError(data.message || "The selected model is unavailable.");
      }
    } catch (err) {
      setError(
        err.message?.includes("fetch")
          ? "Cannot connect to the backend. Start FastAPI at http://127.0.0.1:8000."
          : err.message || "Something went wrong. Please try again."
      );
    } finally {
      setLoading(false);
    }
  }

  function resetForm() {
    setForm({ ...initialForm });
    setResults([]);
    setMessage("");
    setError("");
    setSearched(false);
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div className="hero-inner">
          <div className="brand-mark">CR</div>
          <div className="eyebrow">AI-ASSISTED COLLEGE EXPLORATION</div>
          <h1>
            Your rank.
            <br />
            <span>Your next possibility.</span>
          </h1>
          <p className="hero-description">
            Explore engineering college options using historical counselling
            data and machine-learning estimates.
          </p>

          <div className="hero-tags">
            <span>Historical data</span>
            <span>ML estimates</span>
            <span>Transparent limitations</span>
          </div>
        </div>
      </header>

      <main className="dashboard">
        <section className="panel form-panel">
          <div className="section-heading">
            <div>
              <div className="eyebrow">STEP 01</div>
              <h2>Your preferences</h2>
              <p>Enter your exam and rank to explore options.</p>
            </div>
          </div>

          <form onSubmit={predict}>
            <label>
              Entrance examination
              <select name="exam" value={form.exam} onChange={update}>
                {EXAMS.map((exam) => (
                  <option key={exam} value={exam}>
                    {exam}
                    {!["JEE Main", "JEE Advanced"].includes(exam)
                      ? " — Coming soon"
                      : ""}
                  </option>
                ))}
              </select>
            </label>

            {!supported && (
              <div className="notice">
                This exam is listed for future development. Prediction is
                currently available only for JEE Main and JEE Advanced.
              </div>
            )}

            <label>
              Your rank
              <input
                name="rank"
                type="number"
                min="1"
                step="1"
                placeholder="e.g. 5000"
                value={form.rank}
                onChange={update}
                required
              />
              <span className="field-help">
                Enter your rank, not your marks or percentile.
              </span>
            </label>

            <label>
              Seat category
              <select
                name="category"
                value={form.category}
                onChange={update}
              >
                {(form.exam === "JEE Advanced" ? ["OPEN"] : CATEGORIES).map(
                  (category) => (
                    <option key={category} value={category}>
                      {category}
                    </option>
                  )
                )}
              </select>
              {form.exam === "JEE Advanced" && (
                <span className="field-help">
                  Current JEE Advanced dataset covers OPEN category records.
                </span>
              )}
            </label>

            <label>
              Gender / seat pool
              <select name="gender" value={form.gender} onChange={update}>
                <option value="Gender-Neutral">Gender-neutral</option>
                <option value="Female-only">
                  Female-only / supernumerary
                </option>
              </select>
            </label>

            <label>
              State / domicile
              <input
                name="state"
                value={form.state}
                onChange={update}
                placeholder="e.g. Telangana"
              />
              <span className="field-help">
                State is collected as user input. The quota selection below
                filters the available JEE Main records; it does not verify
                Home State eligibility.
              </span>
            </label>

            {form.exam === "JEE Main" && (
              <label>
                Counselling quota
                <select name="quota" value={form.quota} onChange={update}>
                  <option value="AI">All India (AI)</option>
                  <option value="HS">Home State (HS)</option>
                  <option value="OS">Other State (OS)</option>
                </select>
                <span className="field-help">
                  Select a quota to compare matching records. Official
                  eligibility depends on the institute and counselling rules.
                </span>
              </label>
            )}

            <label>
              Preferred branch
              <input
                name="preferred_branch"
                value={form.preferred_branch}
                onChange={update}
                placeholder="e.g. Computer Science"
              />
              <span className="field-help">
                Current datasets focus on CSE-related programmes.
              </span>
            </label>

            <div className="round-field">
              <span>Counselling round</span>
              <strong>Round 2</strong>
              <span className="field-help">
                The current historical datasets use Round 2 records.
              </span>
            </div>

            {error && (
              <div className="error-message" role="alert">
                {error}
              </div>
            )}

            <button type="submit" disabled={loading || !supported}>
              {loading ? (
                <>
                  <span className="spinner" />
                  Analysing historical data...
                </>
              ) : (
                "Explore college options →"
              )}
            </button>

            <button
              className="secondary-button"
              type="button"
              onClick={resetForm}
              disabled={loading}
            >
              Reset form
            </button>
          </form>
        </section>

        <section className="results-section">
          <div className="results-heading">
            <div>
              <div className="eyebrow">STEP 02</div>
              <h2>Prediction results</h2>
              <p>Review estimated cutoffs and each college's details.</p>
            </div>
            {results.length > 0 && (
              <div className="result-count">{results.length} options</div>
            )}
          </div>

          {loading && (
            <div className="empty-state">
              <div className="large-spinner" />
              <h3>Working on your results</h3>
              <p>Comparing your inputs with historical admission records.</p>
            </div>
          )}

          {!loading && error && (
            <div className="empty-state error-state">
              <div className="empty-icon">!</div>
              <h3>Unable to show predictions</h3>
              <p>{error}</p>
            </div>
          )}

          {!loading && !error && results.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon">↗</div>
              <h3>
                {searched
                  ? "No matching colleges found"
                  : "Your college shortlist starts here"}
              </h3>
              <p>
                {searched
                  ? "Try reviewing your category, gender, branch, quota, or rank. Results depend on the records available in the current dataset."
                  : "Enter your details and select Explore college options to view the available model estimates."}
              </p>
            </div>
          )}

          {!loading && results.length > 0 && (
            <>
              {message && (
                <div className="notice results-notice">{message}</div>
              )}

              <div className="results-grid">
                {results.map((result, index) => {
                  const cutoff = Number(result.predicted_cutoff);
                  const rank = Number(form.rank);
                  const comparable =
                    Number.isFinite(cutoff) && cutoff > 0 && rank > 0;
                  const withinCutoff = comparable && rank <= cutoff;

                  return (
                    <article
                      className="result-card"
                      key={`${result.college}-${index}`}
                    >
                      <div className="result-card-top">
                        <span className="rank-label">
                          OPTION {String(index + 1).padStart(2, "0")}
                        </span>
                        <span
                          className={`status-pill ${
                            withinCutoff ? "status-within" : "status-review"
                          }`}
                        >
                          {comparable
                            ? withinCutoff
                              ? "Within estimated cutoff"
                              : "Beyond estimated cutoff"
                            : "Estimate unavailable"}
                        </span>
                      </div>

                      <h3>{result.college}</h3>
                      <p className="branch-name">{result.branch}</p>

                      <div className="cutoff-box">
                        <span>Estimated closing rank</span>
                        <strong>
                          {comparable
                            ? Math.round(cutoff).toLocaleString("en-IN")
                            : "N/A"}
                        </strong>
                        <small>
                          Model estimate, not an admission guarantee
                        </small>
                      </div>

                      <div className="metrics-grid">
                        <div>
                          <span>Historical cutoff</span>
                          <strong>
                            {result.historical_cutoff != null
                              ? Number(result.historical_cutoff).toLocaleString(
                                  "en-IN",
                                  { maximumFractionDigits: 0 }
                                )
                              : "N/A"}
                          </strong>
                        </div>
                        <div>
                          <span>Previous-year cutoff</span>
                          <strong>
                            {result.previous_year_cutoff != null
                              ? Number(
                                  result.previous_year_cutoff
                                ).toLocaleString("en-IN", {
                                  maximumFractionDigits: 0,
                                })
                              : "N/A"}
                          </strong>
                        </div>
                      </div>

                      <div className="trend-row">
                        <span>Cutoff trend</span>
                        <strong>{result.trend || "Not available"}</strong>
                      </div>

                      <p className="explanation">
                        {result.explanation ||
                          "This estimate is based on the available historical records."}
                      </p>

                      <p className="comparison-note">
                        {comparable
                          ? withinCutoff
                            ? "Your entered rank is numerically at or better than this estimated cutoff. Admission is not guaranteed."
                            : "Your entered rank is numerically beyond this estimated cutoff. This comparison does not determine eligibility."
                          : "There is insufficient cutoff data to compare your rank."}
                      </p>
                    </article>
                  );
                })}
              </div>
            </>
          )}

          <div className="disclaimer">
            <strong>Important:</strong> These are experimental historical
            machine-learning estimates, not admission probabilities or
            guarantees. Actual allotments depend on official counselling
            rules, seat availability, category, quota, gender pool, round,
            preferences, and current-year cutoffs. Always verify details using
            the official counselling authority.
          </div>
        </section>
      </main>

      <footer className="site-footer">
        <div className="footer-brand">CollegeRank AI</div>
        <p>Explore possibilities. Verify with official counselling data.</p>
        <span>Machine-learning estimates for educational guidance.</span>
      </footer>
    </div>
  );
}
