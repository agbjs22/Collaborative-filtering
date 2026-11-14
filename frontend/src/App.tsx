import { FormEvent, useEffect, useState } from 'react'

type Movie = { item_id: number; title: string; score?: number; explanation?: string }
type Metric = Record<string, string | number | null>

type AnalyticsSummary = {
  total_users: number
  total_movies: number
  total_ratings: number
  average_rating: number
  full_sparsity_pct: number
  train_sparsity_pct: number
  sparsity_formula: string
  rating_distribution: Array<{ rating: number; count: number; percentage: number }>
  most_rated_movies: Array<{ item_id: number; title: string; count: number; avg_rating: number }>
}

async function readResponse(response: Response) {
  const contentType = response.headers.get('content-type') ?? ''
  const body = contentType.includes('application/json') ? await response.json() : await response.text()
  if (!response.ok) {
    const detail = typeof body === 'object' && body !== null && 'detail' in body ? String(body.detail) : ''
    throw new Error(detail || `Request failed (${response.status}). Is the API server running on port 8000?`)
  }
  if (!contentType.includes('application/json')) {
    throw new Error('The API returned an unexpected response. Start the FastAPI server on port 8000 and reload this page.')
  }
  return body
}

const api = (path: string) => fetch(`/api${path}`).then(readResponse)

function Stars({ score }: { score?: number }) {
  if (score === undefined) return null
  return <span className="score">★ {score.toFixed(2)}</span>
}

function MovieList({ items, empty = 'Choose a profile to see films.' }: { items: Movie[]; empty?: string }) {
  return (
    <div className="movie-list">
      {items.length ? (
        items.map((movie, i) => (
          <article className="movie" key={`${movie.item_id}-${i}`}>
            <span className="rank">{String(i + 1).padStart(2, '0')}</span>
            <div>
              <h3>{movie.title} <span className="item-id-badge">ID: {movie.item_id}</span></h3>
              {movie.explanation && <p>{movie.explanation}</p>}
            </div>
            {movie.score !== undefined && <Stars score={movie.score} />}
          </article>
        ))
      ) : (
        <p className="empty">{empty}</p>
      )}
    </div>
  )
}

export default function App() {
  const [tab, setTab] = useState<'discover' | 'demo' | 'analytics' | 'compare' | 'export' | 'new'>('discover')
  
  // Discover & Profile State
  const [userId, setUserId] = useState('1')
  const [model, setModel] = useState('svd')
  const [recommendations, setRecommendations] = useState<Movie[]>([])
  const [history, setHistory] = useState<Movie[]>([])
  const [notice, setNotice] = useState('')
  const [loading, setLoading] = useState(false)

  // Live Demo State
  const [demoUserId, setDemoUserId] = useState('1')
  const [demoModel, setDemoModel] = useState('svd')
  const [demoCount, setDemoCount] = useState(10)
  const [demoLoading, setDemoLoading] = useState(false)
  const [currentRecs, setCurrentRecs] = useState<Movie[]>([])
  const [previousRecs, setPreviousRecs] = useState<Movie[]>([])
  const [prevModelName, setPrevModelName] = useState<string>('')
  const [prevUserId, setPrevUserId] = useState<string>('')

  // Analytics & Metrics State
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null)
  const [metrics, setMetrics] = useState<Metric[]>([])
  const [analyticsLoading, setAnalyticsLoading] = useState(false)

  // New Profile Builder State
  const [query, setQuery] = useState('')
  const [searches, setSearches] = useState<Movie[]>([])
  const [selected, setSelected] = useState<{ [key: number]: number }>({})

  // Fetch Discover recommendations
  const discover = async (event?: FormEvent) => {
    event?.preventDefault()
    setNotice('')
    setLoading(true)
    try {
      const [rec, hist] = await Promise.all([
        api(`/recommend/${userId}?model=${model}&n=10`),
        api(`/users/${userId}/history`)
      ])
      setRecommendations(rec.items)
      setHistory(hist.items)
    } catch (e) {
      setNotice((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  // Fetch Live Demo recommendations with delta comparison tracking
  const runLiveDemo = async (event?: FormEvent) => {
    event?.preventDefault()
    setNotice('')
    setDemoLoading(true)
    try {
      const rec = await api(`/recommend/${demoUserId}?model=${demoModel}&n=${demoCount}`)
      if (currentRecs.length > 0) {
        setPreviousRecs(currentRecs)
        setPrevModelName(demoModel)
        setPrevUserId(demoUserId)
      }
      setCurrentRecs(rec.items)
    } catch (e) {
      setNotice((e as Error).message)
    } finally {
      setDemoLoading(false)
    }
  }

  // Initial load
  useEffect(() => {
    discover()
  }, [])

  // Lazy load Analytics
  useEffect(() => {
    if (tab === 'analytics' && !analytics) {
      setAnalyticsLoading(true)
      api('/analytics/summary')
        .then(data => setAnalytics(data))
        .catch(e => setNotice(e.message))
        .finally(() => setAnalyticsLoading(false))
    }
    if ((tab === 'analytics' || tab === 'compare') && !metrics.length) {
      api('/metrics')
        .then(x => setMetrics(x.items))
        .catch(e => setNotice(e.message))
    }
  }, [tab])

  const findMovies = async (event: FormEvent) => {
    event.preventDefault()
    const result = await api(`/movies?query=${encodeURIComponent(query)}&limit=25`)
    setSearches(result.items)
  }

  const newUser = async () => {
    if (!Object.keys(selected).length) return setNotice('Rate at least one film first.')
    try {
      const data = await fetch('/api/recommend/new-user', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ratings: Object.entries(selected).map(([item_id, rating]) => ({ item_id: Number(item_id), rating })),
          model: 'item_cf',
          n: 10
        })
      }).then(readResponse)
      setRecommendations(data.items)
      setTab('discover')
      setNotice('Your personalized recommendations are ready below!')
    } catch (e) {
      setNotice((e as Error).message)
    }
  }

  // Calculate Delta between Current and Previous run in Live Demo
  const currentIds = new Set(currentRecs.map(m => m.item_id))
  const previousIds = new Set(previousRecs.map(m => m.item_id))
  const retainedMovies = currentRecs.filter(m => previousIds.has(m.item_id))
  const newMovies = currentRecs.filter(m => !previousIds.has(m.item_id))
  const droppedMovies = previousRecs.filter(m => !currentIds.has(m.item_id))
  const jaccardSimilarity = previousRecs.length + currentRecs.length > 0 
    ? (retainedMovies.length / (new Set([...currentIds, ...previousIds]).size || 1)) 
    : 0

  return (
    <main>
      <header>
        <a className="brand" onClick={() => setTab('discover')}>
          movie<span>mingle</span>
        </a>
        <nav>
          {[
            ['discover', 'Discover'],
            ['demo', 'Live Demo'],
            ['analytics', 'Analytics'],
            ['compare', 'Model Lab'],
            ['export', 'Export Center'],
          ].map(([id, label]) => (
            <button
              className={tab === id ? 'active' : ''}
              onClick={() => setTab(id as any)}
              key={id}
            >
              {label}
            </button>
          ))}
        </nav>
        <div className="chip">MovieLens 100K Benchmark</div>
      </header>

      {notice && (
        <div className="notice">
          <span>{notice}</span>
          <button onClick={() => setNotice('')}>×</button>
        </div>
      )}

      {/* TAB 1: DISCOVER */}
      {tab === 'discover' && (
        <>
          <section className="hero">
            <div>
              <p className="eyebrow">Data Mining Techniques Project</p>
              <h1>Find a film that<br /><em>feels right tonight.</em></h1>
              <p className="lead">
                Explore real collaborative filtering recommenders across 943 viewers and 100,000 ratings.
              </p>
            </div>
            <div className="hero-orb">
              <span>✦</span>
              <b>MovieLens<br />100K</b>
            </div>
          </section>

          <section className="controls">
            <form onSubmit={discover}>
              <label>
                Viewer ID (1 - 943)
                <input
                  value={userId}
                  onChange={e => setUserId(e.target.value)}
                  type="number"
                  min="1"
                  max="943"
                />
              </label>
              <label>
                Algorithm
                <select value={model} onChange={e => setModel(e.target.value)}>
                  <option value="svd">SVD — Matrix Factorization (Latent Factors)</option>
                  <option value="item_cf_cosine">Item CF — Cosine (Similar Films)</option>
                  <option value="item_cf_adj">Item CF — Adjusted Cosine (Normalized)</option>
                  <option value="user_cf_pearson">User CF — Pearson (Similar Viewers)</option>
                  <option value="user_cf_cosine">User CF — Cosine (Raw Profile Angle)</option>
                  <option value="popularity">Popularity Baseline (Cold-Start Fallback)</option>
                </select>
              </label>
              <button className="primary" disabled={loading}>
                {loading ? 'Computing…' : 'Refresh picks →'}
              </button>
            </form>
          </section>

          <section className="grid">
            <div className="panel large">
              <div className="panel-head">
                <div>
                  <p className="eyebrow">Personalized Recommendations</p>
                  <h2>Viewer {userId} Selection</h2>
                </div>
                <span>{recommendations.length} picks</span>
              </div>
              <MovieList items={recommendations} />
            </div>
            <aside className="panel">
              <p className="eyebrow">Rating History</p>
              <h2>Films Rated by Viewer {userId}</h2>
              <MovieList items={history} empty="Loading rated films…" />
              <button className="text-button" onClick={() => setTab('new')}>
                ★ Rate movies to create a new profile →
              </button>
            </aside>
          </section>
        </>
      )}

      {/* TAB 2: LIVE DEMO */}
      {tab === 'demo' && (
        <section className="demo-section">
          <div className="section-header">
            <p className="eyebrow">Interactive Verification</p>
            <h1>Live Recommendation <em>Demonstrator</em></h1>
            <p className="lead">
              Evaluate how switching users or collaborative filtering algorithms dynamically alters 
              predicted ratings and top-N ranking composition in real-time.
            </p>
          </div>

          <div className="demo-controls-card">
            <form onSubmit={runLiveDemo} className="demo-form">
              <div className="form-group">
                <label>User Profile</label>
                <div className="quick-picks">
                  {['1', '100', '455', '876'].map(id => (
                    <button
                      type="button"
                      key={id}
                      className={demoUserId === id ? 'chip-btn active' : 'chip-btn'}
                      onClick={() => setDemoUserId(id)}
                    >
                      User {id}
                    </button>
                  ))}
                  <input
                    type="number"
                    min="1"
                    max="943"
                    value={demoUserId}
                    onChange={e => setDemoUserId(e.target.value)}
                    className="custom-user-input"
                    title="User ID between 1 and 943"
                  />
                </div>
              </div>

              <div className="form-group">
                <label>Collaborative Filtering Algorithm</label>
                <select value={demoModel} onChange={e => setDemoModel(e.target.value)}>
                  <option value="svd">SVD — Matrix Factorization (Surprise SGD)</option>
                  <option value="item_cf_cosine">Item-Based CF — Cosine Similarity</option>
                  <option value="item_cf_adj">Item-Based CF — Adjusted Cosine (Mean-Centered)</option>
                  <option value="user_cf_pearson">User-Based CF — Pearson Correlation</option>
                  <option value="user_cf_cosine">User-Based CF — Cosine Similarity</option>
                  <option value="popularity">Popularity Baseline — Frequency & Mean</option>
                </select>
              </div>

              <div className="form-group count-group">
                <label>Count: {demoCount}</label>
                <input
                  type="range"
                  min="5"
                  max="20"
                  value={demoCount}
                  onChange={e => setDemoCount(Number(e.target.value))}
                />
              </div>

              <button className="primary demo-submit" disabled={demoLoading}>
                {demoLoading ? 'Querying Model…' : 'Generate Recommendations ⚡'}
              </button>
            </form>
          </div>

          {/* REAL-TIME DELTA COMPARISON */}
          {previousRecs.length > 0 && currentRecs.length > 0 && (
            <div className="comparison-banner">
              <div className="comparison-header">
                <h3>Model Delta Analysis</h3>
                <span className="jaccard-stat">
                  Jaccard Similarity: <b>{(jaccardSimilarity * 100).toFixed(1)}%</b>
                </span>
              </div>
              <p className="comparison-meta">
                Comparing current run (<b>User {demoUserId}</b> with <b>{demoModel}</b>) against 
                previous run (<b>User {prevUserId}</b> with <b>{prevModelName}</b>):
              </p>
              <div className="comparison-grid">
                <div className="delta-card retained">
                  <h4>Common Movies ({retainedMovies.length})</h4>
                  <p>Retained in both recommendation lists</p>
                  <ul>
                    {retainedMovies.length ? (
                      retainedMovies.map(m => <li key={m.item_id}>• {m.title}</li>)
                    ) : (
                      <li className="dim">None — lists completely diverged</li>
                    )}
                  </ul>
                </div>

                <div className="delta-card added">
                  <h4>Newly Recommended ({newMovies.length})</h4>
                  <p>Surfaced specifically by {demoModel}</p>
                  <ul>
                    {newMovies.map(m => (
                      <li key={m.item_id}>+ {m.title}</li>
                    ))}
                  </ul>
                </div>

                <div className="delta-card dropped">
                  <h4>Dropped Movies ({droppedMovies.length})</h4>
                  <p>Present in {prevModelName} but replaced</p>
                  <ul>
                    {droppedMovies.map(m => (
                      <li key={m.item_id}>- {m.title}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}

          {/* ACTIVE DEMO RECOMMENDATIONS */}
          <div className="panel large demo-results">
            <div className="panel-head">
              <div>
                <p className="eyebrow">Live Model Output</p>
                <h2>User {demoUserId} Recommendations ({demoModel})</h2>
              </div>
              <a
                href={`/api/export/recommendations?user_id=${demoUserId}&model=${demoModel}&n=${demoCount}`}
                download
                className="export-link"
              >
                ⬇ Download CSV
              </a>
            </div>
            {demoLoading ? (
              <div className="loader">Running Collaborative Filtering inference in memory…</div>
            ) : currentRecs.length ? (
              <MovieList items={currentRecs} />
            ) : (
              <div className="empty-prompt">
                <p>Click "Generate Recommendations ⚡" to run the live collaborative filtering model.</p>
              </div>
            )}
          </div>
        </section>
      )}

      {/* TAB 3: ANALYTICS */}
      {tab === 'analytics' && (
        <section className="analytics-section">
          <div className="section-header">
            <p className="eyebrow">Empirical Ground Truth</p>
            <h1>Dataset & Benchmark <em>Analytics</em></h1>
            <p className="lead">
              Verified dataset statistics, interaction sparsity analysis, rating distributions, and model performance.
            </p>
          </div>

          {analyticsLoading ? (
            <div className="loader">Aggregating dataset statistics…</div>
          ) : analytics ? (
            <>
              {/* SUMMARY CARDS */}
              <div className="stats-row">
                <div className="stat-card">
                  <span className="stat-label">Total Users</span>
                  <span className="stat-value">{analytics.total_users.toLocaleString()}</span>
                  <span className="stat-sub">Each rated ≥ 20 films</span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">Total Movies</span>
                  <span className="stat-value">{analytics.total_movies.toLocaleString()}</span>
                  <span className="stat-sub">Catalog items</span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">Total Ratings</span>
                  <span className="stat-value">{analytics.total_ratings.toLocaleString()}</span>
                  <span className="stat-sub">Explicit 1–5 scale</span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">Average Rating</span>
                  <span className="stat-value">{analytics.average_rating} ★</span>
                  <span className="stat-sub">Global rating mean</span>
                </div>
              </div>

              {/* SPARSITY CARD */}
              <div className="panel sparsity-card">
                <p className="eyebrow">Matrix Sparsity Formula</p>
                <h2>Interaction Density vs Sparsity</h2>
                <div className="sparsity-details">
                  <div className="formula-block">
                    <code>
                      Sparsity = 1.0 - (|R| / (|U| × |I|)) = 1.0 - (100,000 / (943 × 1,682)) = {analytics.full_sparsity_pct}%
                    </code>
                  </div>
                  <div className="sparsity-bar-container">
                    <div className="sparsity-bar" style={{ width: `${100 - analytics.full_sparsity_pct}%` }}>
                      <span>Observed: {(100 - analytics.full_sparsity_pct).toFixed(2)}%</span>
                    </div>
                    <div className="unobserved-bar" style={{ width: `${analytics.full_sparsity_pct}%` }}>
                      <span>Empty / Unobserved: {analytics.full_sparsity_pct}%</span>
                    </div>
                  </div>
                  <p className="sparsity-note">
                    Training split matrix sparsity is <b>{analytics.train_sparsity_pct}%</b> ($80,000$ training ratings). 
                    Only <b>{(100 - analytics.full_sparsity_pct).toFixed(2)}%</b> of all possible user-movie interactions are known.
                  </p>
                </div>
              </div>

              {/* RATING DISTRIBUTION & MOST RATED */}
              <div className="grid analytics-grid">
                <div className="panel">
                  <p className="eyebrow">Rating Frequency</p>
                  <h2>Rating Distribution</h2>
                  <div className="rating-dist-list">
                    {analytics.rating_distribution.map(d => (
                      <div className="dist-row" key={d.rating}>
                        <span className="star-tag">{d.rating} ★</span>
                        <div className="dist-track">
                          <div className="dist-fill" style={{ width: `${d.percentage * 2}%` }} />
                        </div>
                        <span className="dist-count">{d.count.toLocaleString()} ({d.percentage}%)</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="panel">
                  <p className="eyebrow">Most-Rated Titles</p>
                  <h2>Top 10 Most Rated Movies</h2>
                  <div className="top-movies-list">
                    {analytics.most_rated_movies.map((m, idx) => (
                      <div className="top-movie-item" key={m.item_id}>
                        <span className="top-idx">{idx + 1}</span>
                        <div className="top-title">
                          <b>{m.title}</b>
                          <span>Avg: {m.avg_rating} ★</span>
                        </div>
                        <span className="top-votes">{m.count} votes</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* MODEL BENCHMARK CHARTS */}
              <div className="panel">
                <p className="eyebrow">Offline Benchmark Results</p>
                <h2>Model Evaluation Charts (80/20 Test Split)</h2>
                <div className="chart-preview-grid">
                  <div className="chart-box">
                    <h4>Rating Prediction Error (Lower is better)</h4>
                    <p>Item-Based CF (Cosine) achieves the lowest error (RMSE: 0.9178, MAE: 0.7200).</p>
                    <div className="visual-bar-group">
                      {metrics.filter(m => m.RMSE !== null).map((m, i) => (
                        <div className="vbar-row" key={i}>
                          <span className="vbar-label">{String(m.Model)}</span>
                          <div className="vbar-track">
                            <div
                              className="vbar-fill"
                              style={{ width: `${((Number(m.RMSE) - 0.88) / 0.1) * 100}%` }}
                            />
                          </div>
                          <span className="vbar-val">RMSE: {Number(m.RMSE).toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="chart-box">
                    <h4>Top-10 Ranking Quality NDCG@10 (Higher is better)</h4>
                    <p>SVD delivers the highest personalized ranking quality (NDCG@10: 0.0900).</p>
                    <div className="visual-bar-group">
                      {metrics.map((m, i) => (
                        <div className="vbar-row" key={i}>
                          <span className="vbar-label">{String(m.Model)}</span>
                          <div className="vbar-track">
                            <div
                              className="vbar-fill blue"
                              style={{ width: `${(Number(m['NDCG@10']) / 0.18) * 100}%` }}
                            />
                          </div>
                          <span className="vbar-val">NDCG: {Number(m['NDCG@10']).toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <p className="empty">Unable to load analytics data.</p>
          )}
        </section>
      )}

      {/* TAB 4: MODEL LAB */}
      {tab === 'compare' && (
        <section className="lab">
          <p className="eyebrow">Academic Benchmark Protocol</p>
          <h1>How the models <em>compare.</em></h1>
          <p className="lead">
            Evaluated on an identical 80/20 train/test split (random seed 42) with 20,000 held-out ratings. 
            Relevance threshold for top-N ranking metrics is $r \ge 4.0$.
          </p>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  {['Model', 'RMSE ↓', 'MAE ↓', 'Precision@10 ↑', 'Recall@10 ↑', 'F1@10 ↑', 'NDCG@10 ↑'].map(h => (
                    <th key={h}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {metrics.map((row, i) => {
                  const isBestRmse = row.Model === 'Item-based CF (cosine)'
                  const isBestSvd = row.Model === 'SVD'
                  const isPop = row.Model === 'Popularity baseline'
                  return (
                    <tr key={i} className={isBestRmse ? 'highlight-row' : ''}>
                      <td>
                        <b>{String(row.Model)}</b>
                        {isBestRmse && <span className="best-badge">Best Rating Accuracy</span>}
                        {isBestSvd && <span className="best-badge blue">Best Personalized Ranking</span>}
                        {isPop && <span className="best-badge yellow">Cold-Start Baseline</span>}
                      </td>
                      <td>{row.RMSE !== null ? Number(row.RMSE).toFixed(4) : <span className="dim">N/A</span>}</td>
                      <td>{row.MAE !== null ? Number(row.MAE).toFixed(4) : <span className="dim">N/A</span>}</td>
                      <td>{Number(row['Precision@10']).toFixed(4)}</td>
                      <td>{Number(row['Recall@10']).toFixed(4)}</td>
                      <td>{row['F1@10'] !== null && row['F1@10'] !== undefined ? Number(row['F1@10']).toFixed(4) : '—'}</td>
                      <td>{Number(row['NDCG@10']).toFixed(4)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          <div className="lab-insights">
            <div className="insight-card">
              <h4>🎯 Rating Error vs. Top-N Ranking Trade-Off</h4>
              <p>
                No single algorithm is universally superior across all dimensions. <b>Item-Based CF</b> achieves the lowest 
                rating error (RMSE 0.9178) because item similarities are more stable over co-rated sets. However, <b>SVD</b> 
                achieves significantly higher personalized NDCG@10 (0.0900) by projecting interactions into 50 latent dimensions.
              </p>
            </div>
            <div className="insight-card">
              <h4>❄️ Cold-Start Handling Strategy</h4>
              <p>
                When a user has rated fewer than 5 items, collaborative filtering suffers from insufficient neighborhood evidence. 
                The <b>Popularity Baseline</b> provides a robust fallback (NDCG@10 of 0.1763) until the viewer establishes a taste profile.
              </p>
            </div>
          </div>
        </section>
      )}

      {/* TAB 5: EXPORT CENTER */}
      {tab === 'export' && (
        <section className="export-section">
          <div className="section-header">
            <p className="eyebrow">Data Export Center</p>
            <h1>Download Project <em>Data & Metrics</em></h1>
            <p className="lead">
              Export verified CSV datasets, model evaluation results, and personalized recommendation lists.
            </p>
          </div>

          <div className="export-grid">
            <div className="export-card">
              <div className="export-icon">📊</div>
              <h3>Model Benchmark Metrics</h3>
              <p>Complete evaluation table with RMSE, MAE, Precision@10, Recall@10, F1@10, and NDCG@10 for all 6 models.</p>
              <a href="/api/export/metrics" download className="primary export-btn">
                Download Metrics (CSV)
              </a>
            </div>

            <div className="export-card">
              <div className="export-icon">🎬</div>
              <h3>Full Ratings Dataset</h3>
              <p>The complete MovieLens 100K ratings dataset (100,000 explicit ratings from 943 users over 1,682 films).</p>
              <a href="/api/export/ratings" download className="primary export-btn">
                Download Ratings (CSV)
              </a>
            </div>

            <div className="export-card">
              <div className="export-icon">📈</div>
              <h3>Dataset Analytics Summary</h3>
              <p>Aggregated dataset distributions, rating frequency breakdown, sparsity formulas, and most-rated titles.</p>
              <a href="/api/export/analytics" download className="primary export-btn">
                Download Summary (CSV)
              </a>
            </div>

            <div className="export-card">
              <div className="export-icon">⚡</div>
              <h3>Live Recommendations</h3>
              <p>Personalized top-10 recommendation list for User {demoUserId} generated using {demoModel}.</p>
              <a
                href={`/api/export/recommendations?user_id=${demoUserId}&model=${demoModel}&n=${demoCount}`}
                download
                className="primary export-btn"
              >
                Download User {demoUserId} Picks (CSV)
              </a>
            </div>
          </div>
        </section>
      )}

      {/* TAB 6: NEW PROFILE (BUILD MY TASTE) */}
      {tab === 'new' && (
        <section className="builder">
          <p className="eyebrow">New Profile Onboarding</p>
          <h1>Tell us a little<br /><em>about your taste.</em></h1>
          <p className="lead">
            Search for films you know and assign a 1–5 star rating to simulate cold-start bootstrapping.
          </p>
          <form className="search" onSubmit={findMovies}>
            <input
              placeholder="Try: Toy Story, Star Wars, Titanic, Fargo…"
              value={query}
              onChange={e => setQuery(e.target.value)}
            />
            <button className="primary">Search</button>
          </form>
          <div className="search-results">
            {searches.map(movie => (
              <div className="search-row" key={movie.item_id}>
                <span>{movie.title}</span>
                <div className="rating">
                  {[1, 2, 3, 4, 5].map(n => (
                    <button
                      type="button"
                      onClick={() => setSelected({ ...selected, [movie.item_id]: n })}
                      className={selected[movie.item_id] === n ? 'rated' : ''}
                      key={n}
                    >
                      ★
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div className="sticky-action">
            <span>{Object.keys(selected).length} films rated</span>
            <button className="primary" onClick={newUser}>
              Get my recommendations →
            </button>
          </div>
        </section>
      )}

      <footer>
        MovieMingle · Collaborative Filtering Explorer · Built with MovieLens 100K for Data Mining Techniques
      </footer>
    </main>
  )
}
