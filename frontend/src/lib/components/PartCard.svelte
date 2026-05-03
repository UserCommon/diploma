<script>
  import { partImageUrl } from '$lib/api.js';
  export let part;
  export let onDelete;
</script>

<div class="part-card">
  <div class="img-wrap">
    {#if part.processed_image_url}
      <img src={partImageUrl(part.processed_image_url)} alt={part.name} />
    {:else if part.original_image_url}
      <img src={partImageUrl(part.original_image_url)} alt={part.name} />
    {:else}
      <div class="no-img">No image</div>
    {/if}
  </div>

  <div class="info">
    <div class="name">{part.name}</div>
    <div class="meta">
      <span>{part.category}</span>
      <span>·</span>
      <span>{part.domain}</span>
    </div>
    <div class="footer">
      <span class="badge badge-{part.status}">{part.status}</span>
      <button class="btn-danger" on:click={() => onDelete(part.id)}>Delete</button>
    </div>
  </div>
</div>

<style>
  .part-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: border-color 0.15s;
  }
  .part-card:hover { border-color: #444; }

  .img-wrap {
    aspect-ratio: 1;
    background: var(--surface2);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }
  img { width: 100%; height: 100%; object-fit: contain; }
  .no-img { color: var(--text-muted); font-size: 12px; }

  .info { padding: 10px 12px; display: flex; flex-direction: column; gap: 6px; }
  .name { font-weight: 600; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .meta { font-size: 11px; color: var(--text-muted); display: flex; gap: 4px; }
  .footer { display: flex; align-items: center; justify-content: space-between; }
  button { padding: 4px 10px; font-size: 12px; }
</style>
