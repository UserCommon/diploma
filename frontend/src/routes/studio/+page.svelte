<script>
  import { createSession, streamSession, resultImageUrl } from '$lib/api.js';
  import EventLog from '$lib/components/EventLog.svelte';

  let prompt = '';
  let images = [];
  let previews = [];
  let nVariants = 2;
  let inpaintProvider = 'gpt_image';

  const providers = [
    { value: 'klein', label: 'Flux 2 Klein' },
    { value: 'kontext', label: 'Flux Kontext' },
    { value: 'genapi', label: 'FLUX Inpaint' },
    { value: 'gpt_image', label: 'GPT Image' },
    { value: 'polza', label: 'FLUX.2 Pro' },
    { value: 'sdxl', label: 'SDXL (local)' },
    { value: 'diffusers', label: 'FLUX Fill (local)' },
  ];

  let sessionId = null;
  let status = 'idle'; // idle | running | complete | failed
  let events = [];
  let resultUrls = [];
  let error = '';

  let stopStream = null;
  let selectedResult = null;

  function handleFiles(files) {
    images = [...images, ...Array.from(files)];
    previews = images.map(f => URL.createObjectURL(f));
  }

  function removeImage(i) {
    images = images.filter((_, idx) => idx !== i);
    previews = previews.filter((_, idx) => idx !== i);
  }

  async function start() {
    if (!prompt.trim()) return;
    events = [];
    resultUrls = [];
    error = '';
    selectedResult = null;
    status = 'running';

    try {
      const fd = new FormData();
      fd.append('user_prompt', prompt);
      fd.append('n', String(nVariants));
      fd.append('inpaint_provider', inpaintProvider);
      for (const img of images) fd.append('images', img);

      const { session_id } = await createSession(fd);
      sessionId = session_id;

      stopStream = streamSession(
        session_id,
        (event) => {
          events = [...events, event];
          if (event.type === 'result' && event.result_image_urls?.length) {
            resultUrls = event.result_image_urls;
            selectedResult = resultUrls[0];
          }
        },
        (finalStatus) => {
          status = finalStatus;
          stopStream = null;
        }
      );
    } catch (e) {
      error = e.message;
      status = 'failed';
    }
  }

  function reset() {
    if (stopStream) stopStream();
    stopStream = null;
    status = 'idle';
    events = [];
    resultUrls = [];
    selectedResult = null;
    error = '';
    sessionId = null;
  }
</script>

<div class="studio">
  <div class="left">
    <h2>Studio</h2>

    <!-- Image upload -->
    <div class="section">
      <p class="section-label">Car Images</p>
      <div class="dropzone"
        on:dragover|preventDefault
        on:drop|preventDefault={(e) => handleFiles(e.dataTransfer.files)}>
        {#if previews.length === 0}
          <span class="hint">Drop car photos here or</span>
          <label class="file-btn">
            browse
            <input type="file" accept="image/*" multiple
              on:change={(e) => handleFiles(e.target.files)} />
          </label>
        {:else}
          <div class="thumb-row">
            {#each previews as src, i}
              <div class="thumb">
                <img {src} alt="car {i+1}" />
                <button class="remove" on:click={() => removeImage(i)}>×</button>
              </div>
            {/each}
            <label class="add-more">
              +
              <input type="file" accept="image/*" multiple
                on:change={(e) => handleFiles(e.target.files)} />
            </label>
          </div>
        {/if}
      </div>
    </div>

    <!-- Prompt -->
    <div class="section">
      <p class="section-label">Tuning Request</p>
      <textarea
        bind:value={prompt}
        rows="3"
        placeholder="e.g. Add sport wheels and a carbon rear spoiler"
        disabled={status === 'running'}
      />
    </div>

    <!-- Provider -->
    <div class="section">
      <p class="section-label">Inpaint Model</p>
      <div class="provider-row">
        {#each providers as p}
          <button
            class="provider-btn"
            class:active={inpaintProvider === p.value}
            disabled={status === 'running'}
            on:click={() => inpaintProvider = p.value}
          >{p.label}</button>
        {/each}
      </div>
    </div>

    <!-- Variants -->
    <div class="section">
      <p class="section-label">Variants</p>
      <div class="variants-row">
        {#each [1, 2, 3, 4] as n}
          <button
            class="variant-btn"
            class:active={nVariants === n}
            disabled={status === 'running'}
            on:click={() => nVariants = n}
          >{n}</button>
        {/each}
      </div>
    </div>

    <!-- Actions -->
    <div class="actions">
      {#if status === 'idle' || status === 'complete' || status === 'failed'}
        {#if status !== 'idle'}
          <button class="btn-ghost" on:click={reset}>Reset</button>
        {/if}
        <button class="btn-primary" disabled={!prompt.trim()} on:click={start}>
          Tune ✦
        </button>
      {:else}
        <button class="btn-ghost" on:click={reset}>Cancel</button>
        <div class="running-badge">
          <span class="dot"></span> Agent running…
        </div>
      {/if}
    </div>

    {#if error}<p class="err">{error}</p>{/if}

    <!-- Event log -->
    {#if events.length > 0 || status === 'running'}
      <div class="log-wrap card">
        <div class="log-header">
          <span>Agent Log</span>
          {#if sessionId}
            <span class="session-id">{sessionId.slice(0, 8)}…</span>
          {/if}
        </div>
        <EventLog {events} />
      </div>
    {/if}
  </div>

  <!-- Results -->
  <div class="right">
    <p class="section-label">Results</p>

    {#if resultUrls.length === 0}
      <div class="placeholder">
        <span>Result images will appear here</span>
      </div>
    {:else}
      <div class="result-main">
        {#if selectedResult}
          <img src={resultImageUrl(selectedResult)} alt="result" class="main-img" />
        {/if}
      </div>
      <div class="result-thumbs">
        {#each resultUrls as url}
          <button
            class="result-thumb"
            class:active={url === selectedResult}
            on:click={() => selectedResult = url}
          >
            <img src={resultImageUrl(url)} alt="variant" />
          </button>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .studio {
    display: grid;
    grid-template-columns: 380px 1fr;
    gap: 24px;
    align-items: start;
  }

  h2 { font-size: 20px; font-weight: 700; margin-bottom: 20px; }

  .left { display: flex; flex-direction: column; gap: 16px; }

  .section { display: flex; flex-direction: column; gap: 6px; }
  .section-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-muted);
  }

  .dropzone {
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 20px;
    text-align: center;
    color: var(--text-muted);
    font-size: 13px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    min-height: 100px;
    justify-content: center;
  }

  .hint { color: var(--text-muted); }
  .file-btn {
    color: var(--accent);
    cursor: pointer;
    font-size: 13px;
  }
  .file-btn input { display: none; }

  .thumb-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
  }

  .thumb {
    position: relative;
    width: 72px;
    height: 72px;
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid var(--border);
  }
  .thumb img { width: 100%; height: 100%; object-fit: cover; }
  .remove {
    position: absolute;
    top: 2px;
    right: 2px;
    width: 18px;
    height: 18px;
    padding: 0;
    background: rgba(0,0,0,0.7);
    color: #fff;
    font-size: 14px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    line-height: 1;
  }

  .add-more {
    width: 72px;
    height: 72px;
    border-radius: 6px;
    border: 2px dashed var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    color: var(--text-muted);
    cursor: pointer;
  }
  .add-more input { display: none; }

  .provider-row { display: flex; gap: 8px; flex-wrap: wrap; }
  .provider-btn {
    padding: 6px 12px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text);
    font-size: 13px;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
  }
  .provider-btn:hover:not(:disabled) { border-color: var(--accent); }
  .provider-btn.active { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 15%, transparent); color: var(--accent); }
  .provider-btn:disabled { opacity: 0.5; cursor: not-allowed; }

  .variants-row { display: flex; gap: 8px; }
  .variant-btn {
    width: 44px;
    height: 36px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text);
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
  }
  .variant-btn:hover:not(:disabled) { border-color: var(--accent); }
  .variant-btn.active { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 15%, transparent); color: var(--accent); }
  .variant-btn:disabled { opacity: 0.5; cursor: not-allowed; }

  .actions { display: flex; align-items: center; gap: 10px; }
  .actions .btn-primary { flex: 1; padding: 10px; font-size: 15px; }

  .running-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--accent);
    font-size: 13px;
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1.2s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  .err { color: var(--danger); font-size: 13px; }

  .log-wrap { padding: 12px; }
  .log-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-muted);
    margin-bottom: 10px;
  }
  .session-id { font-family: monospace; font-weight: 400; }

  /* Results panel */
  .right { display: flex; flex-direction: column; gap: 12px; }

  .placeholder {
    aspect-ratio: 4/3;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
    font-size: 14px;
  }

  .result-main {
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--border);
    background: var(--surface2);
  }
  .main-img { width: 100%; display: block; }

  .result-thumbs {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .result-thumb {
    width: 80px;
    height: 80px;
    padding: 0;
    border-radius: 6px;
    overflow: hidden;
    border: 2px solid var(--border);
    background: var(--surface2);
    transition: border-color 0.15s;
  }
  .result-thumb.active { border-color: var(--accent); }
  .result-thumb img { width: 100%; height: 100%; object-fit: cover; }
</style>
