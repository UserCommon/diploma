const ACCESSORIES = import.meta.env.VITE_ACCESSORIES_URL ?? 'http://localhost:8000';
const CORE = import.meta.env.VITE_CORE_URL ?? 'http://localhost:8001';

// ── Parts ─────────────────────────────────────────────────────────────────────

export async function fetchParts({ domain = '', category = '' } = {}) {
  const params = new URLSearchParams();
  if (domain) params.set('domain', domain);
  if (category) params.set('category', category);
  const res = await fetch(`${ACCESSORIES}/api/parts/?${params}`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.items;
}

export async function uploadPart(formData) {
  const res = await fetch(`${ACCESSORIES}/api/parts/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deletePart(id) {
  const res = await fetch(`${ACCESSORIES}/api/parts/${id}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(await res.text());
}

export function partImageUrl(url) {
  if (!url) return null;
  return `${ACCESSORIES}${url}`;
}

// ── Sessions ──────────────────────────────────────────────────────────────────

export async function createSession(formData) {
  const res = await fetch(`${CORE}/api/agent/sessions`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getSession(id) {
  const res = await fetch(`${CORE}/api/agent/sessions/${id}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function streamSession(id, onEvent, onDone) {
  const es = new EventSource(`${CORE}/api/agent/sessions/${id}/stream`);
  es.onmessage = (e) => {
    const event = JSON.parse(e.data);
    if (event.type === 'done') {
      es.close();
      onDone(event.status);
    } else {
      onEvent(event);
    }
  };
  es.onerror = () => { es.close(); onDone('failed'); };
  return () => es.close();
}

export function resultImageUrl(url) {
  return `${CORE}${url}`;
}
