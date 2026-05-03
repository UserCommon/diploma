<script>
  import '../app.css';
  import { page } from '$app/stores';

  const links = [
    { href: '/', label: 'Home' },
    { href: '/studio', label: 'Studio' },
    { href: '/parts', label: 'Parts' },
  ];
</script>

<div class="shell">
  <nav>
    <span class="logo">⚡ TuneAI</span>
    <ul>
      {#each links as { href, label }}
        <li><a {href} class:active={$page.url.pathname === href}>{label}</a></li>
      {/each}
    </ul>
  </nav>

  <main>
    <slot />
  </main>
</div>

<style>
  .shell {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
  }

  nav {
    display: flex;
    align-items: center;
    gap: 32px;
    padding: 0 24px;
    height: 52px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }

  .logo {
    font-weight: 700;
    font-size: 16px;
    letter-spacing: -0.02em;
    color: var(--text);
  }

  ul {
    display: flex;
    gap: 4px;
    list-style: none;
  }

  a {
    display: block;
    padding: 6px 12px;
    border-radius: 6px;
    color: var(--text-muted);
    font-weight: 500;
    transition: color 0.15s, background 0.15s;
  }

  a:hover { color: var(--text); background: var(--surface2); }
  a.active { color: var(--text); background: var(--surface2); }

  main {
    flex: 1;
    padding: 24px;
    max-width: 1200px;
    width: 100%;
    margin: 0 auto;
  }
</style>
