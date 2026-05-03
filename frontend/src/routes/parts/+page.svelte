<script>
  import { onMount } from 'svelte';
  import { fetchParts, uploadPart, deletePart } from '$lib/api.js';
  import PartCard from '$lib/components/PartCard.svelte';

  let parts = [];
  let loading = false;
  let error = '';
  let uploading = false;
  let uploadError = '';

  // filters
  let filterDomain = '';
  let filterCategory = '';

  // upload form
  let imageFile = null;
  let name = '';
  let category = 'other';
  let domain = 'car';
  let description = '';

  const categories = ['spoiler', 'wheels', 'bumper', 'hood', 'mirror', 'headlights', 'wheel', 'other'];
  const domains = ['car', 'moto', 'interior'];

  async function load() {
    loading = true;
    error = '';
    try {
      parts = await fetchParts({ domain: filterDomain, category: filterCategory });
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
  }

  async function handleUpload() {
    if (!imageFile || !name) return;
    uploading = true;
    uploadError = '';
    try {
      const fd = new FormData();
      fd.append('image', imageFile);
      fd.append('name', name);
      fd.append('category', category);
      fd.append('domain', domain);
      if (description) fd.append('description', description);
      await uploadPart(fd);
      name = ''; description = ''; imageFile = null;
      await load();
    } catch (e) {
      uploadError = e.message;
    } finally {
      uploading = false;
    }
  }

  async function handleDelete(id) {
    try {
      await deletePart(id);
      parts = parts.filter(p => p.id !== id);
    } catch (e) {
      error = e.message;
    }
  }

  onMount(load);
</script>

<div class="page">
  <div class="header">
    <h2>Parts Library</h2>
    <div class="filters">
      <select bind:value={filterDomain} on:change={load}>
        <option value="">All domains</option>
        {#each domains as d}<option value={d}>{d}</option>{/each}
      </select>
      <select bind:value={filterCategory} on:change={load}>
        <option value="">All categories</option>
        {#each categories as c}<option value={c}>{c}</option>{/each}
      </select>
    </div>
  </div>

  <div class="layout">
    <!-- Upload form -->
    <div class="upload-form card">
      <h3>Upload Part</h3>

      <div class="dropzone" class:has-file={imageFile}
        on:dragover|preventDefault
        on:drop|preventDefault={(e) => { imageFile = e.dataTransfer.files[0]; }}>
        {#if imageFile}
          <span>📎 {imageFile.name}</span>
          <button class="btn-ghost" style="margin-top:8px;font-size:12px" on:click={() => imageFile = null}>Remove</button>
        {:else}
          <span>Drop image here or</span>
          <label class="file-btn">
            browse
            <input type="file" accept="image/*" on:change={(e) => imageFile = e.target.files[0]} />
          </label>
        {/if}
      </div>

      <div class="field">
        <label for="pname">Name</label>
        <input id="pname" bind:value={name} placeholder="e.g. Carbon Spoiler GT" />
      </div>
      <div class="row">
        <div class="field">
          <label for="pcat">Category</label>
          <select id="pcat" bind:value={category}>
            {#each categories as c}<option value={c}>{c}</option>{/each}
          </select>
        </div>
        <div class="field">
          <label for="pdom">Domain</label>
          <select id="pdom" bind:value={domain}>
            {#each domains as d}<option value={d}>{d}</option>{/each}
          </select>
        </div>
      </div>
      <div class="field">
        <label for="pdesc">Description</label>
        <textarea id="pdesc" bind:value={description} rows="2" placeholder="Optional description…" />
      </div>

      {#if uploadError}<p class="err">{uploadError}</p>{/if}

      <button class="btn-primary" disabled={uploading || !imageFile || !name} on:click={handleUpload}>
        {uploading ? 'Uploading…' : 'Upload Part'}
      </button>
    </div>

    <!-- Grid -->
    <div class="grid-wrap">
      {#if error}<p class="err">{error}</p>{/if}
      {#if loading}
        <p class="muted">Loading…</p>
      {:else if parts.length === 0}
        <p class="muted">No parts found. Upload one to get started.</p>
      {:else}
        <div class="grid">
          {#each parts as part (part.id)}
            <PartCard {part} onDelete={handleDelete} />
          {/each}
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .page { display: flex; flex-direction: column; gap: 20px; }

  .header { display: flex; align-items: center; justify-content: space-between; }
  h2 { font-size: 20px; font-weight: 700; }
  h3 { font-size: 15px; font-weight: 600; margin-bottom: 12px; }

  .filters { display: flex; gap: 8px; }
  .filters select { width: auto; }

  .layout { display: grid; grid-template-columns: 280px 1fr; gap: 20px; align-items: start; }

  .upload-form { display: flex; flex-direction: column; gap: 12px; }

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
    gap: 4px;
    transition: border-color 0.15s;
  }
  .dropzone.has-file { border-color: var(--accent); color: var(--text); }

  .file-btn {
    color: var(--accent);
    cursor: pointer;
    text-transform: none;
    letter-spacing: 0;
    font-size: 13px;
    margin: 0;
  }
  .file-btn input { display: none; }

  .field { display: flex; flex-direction: column; gap: 4px; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }

  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; }
  .muted { color: var(--text-muted); }
  .err { color: var(--danger); font-size: 13px; }
</style>
