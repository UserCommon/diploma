<script>
  import { goto } from '$app/navigation';
</script>

<div class="home">
  <h1>Car Tuning AI</h1>
  <p>Upload a car photo, describe what you want changed, and let the AI handle the rest.</p>
  <div class="actions">
    <button class="btn-primary" on:click={() => goto('/studio')}>Open Studio</button>
    <button class="btn-ghost" on:click={() => goto('/parts')}>Browse Parts</button>
  </div>

  <div class="features">
    <div class="card">
      <div class="icon">🔍</div>
      <h3>Semantic Search</h3>
      <p>Finds the right parts from the library using natural language.</p>
    </div>
    <div class="card">
      <div class="icon">✂️</div>
      <h3>Auto Segmentation</h3>
      <p>SAM2 + GroundingDINO precisely isolates the area to modify.</p>
    </div>
    <div class="card">
      <div class="icon">🎨</div>
      <h3>AI Inpainting</h3>
      <p>FLUX generates photorealistic results with the new part applied.</p>
    </div>
  </div>
</div>

<style>
  .home {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 80px 24px 40px;
    gap: 24px;
  }

  h1 {
    font-size: 48px;
    font-weight: 800;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, #fff 40%, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  p { color: var(--text-muted); font-size: 16px; max-width: 480px; }

  .actions { display: flex; gap: 12px; }
  .actions button { padding: 10px 24px; font-size: 15px; }

  .features {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    width: 100%;
    max-width: 720px;
    margin-top: 24px;
  }

  .card { text-align: left; }
  .icon { font-size: 24px; margin-bottom: 8px; }
  h3 { font-size: 15px; font-weight: 600; margin-bottom: 4px; }
  .card p { font-size: 13px; }
</style>
