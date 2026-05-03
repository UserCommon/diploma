<script>
  export let events = [];

  const icons = {
    thinking: '💭',
    tool_call: '🔧',
    tool_result: '↩',
    result: '✅',
    error: '❌',
    done: '🏁',
  };
</script>

<div class="log">
  {#if events.length === 0}
    <div class="empty">Waiting for agent…</div>
  {:else}
    {#each events as event}
      <div class="entry type-{event.type}">
        <span class="icon">{icons[event.type] ?? '·'}</span>
        <div class="body">
          {#if event.type === 'thinking'}
            <span class="muted">{event.content}</span>
          {:else if event.type === 'tool_call'}
            <span class="tool">{event.tool}</span>
            <span class="muted"> ← {event.input}</span>
          {:else if event.type === 'tool_result'}
            <span class="muted output">{event.output?.slice(0, 200)}{event.output?.length > 200 ? '…' : ''}</span>
          {:else if event.type === 'result'}
            <span>{event.output}</span>
          {:else if event.type === 'error'}
            <span class="error">{event.message}</span>
          {/if}
        </div>
      </div>
    {/each}
  {/if}
</div>

<style>
  .log {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 13px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .empty { color: var(--text-muted); text-align: center; padding: 24px 0; }

  .entry {
    display: flex;
    gap: 8px;
    align-items: flex-start;
    padding: 6px 8px;
    border-radius: 6px;
    line-height: 1.5;
  }

  .entry.type-thinking { color: var(--text-muted); }
  .entry.type-tool_call { background: #0d1a2e; }
  .entry.type-tool_result { background: var(--surface2); }
  .entry.type-result { background: #14230a; }
  .entry.type-error { background: #2a0a0a; }

  .icon { flex-shrink: 0; width: 18px; text-align: center; }
  .body { flex: 1; min-width: 0; word-break: break-word; }
  .tool { color: var(--accent); font-weight: 600; }
  .muted { color: var(--text-muted); }
  .output { display: block; white-space: pre-wrap; }
  .error { color: var(--danger); }
</style>
