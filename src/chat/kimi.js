import config from '../config.js';

// Kimi (Moonshot) exposes an OpenAI-compatible Chat Completions API, so this is
// a thin fetch wrapper — swap KIMI_BASE_URL/KIMI_MODEL for any compatible
// provider. Nothing here imports the quant pipeline.

const BASE = (config.KIMI_BASE_URL || 'https://api.moonshot.ai/v1').replace(/\/+$/, '');

/**
 * Send an OpenAI-style messages array to Kimi and return the reply text.
 * Throws on a non-2xx response; callers handle it best-effort.
 */
export async function kimiChat(messages, { signal } = {}) {
  if (!config.KIMI_API_KEY) throw new Error('KIMI_API_KEY not set');
  const res = await fetch(`${BASE}/chat/completions`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${config.KIMI_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      // No temperature: kimi-for-coding rejects anything but its default, and
      // omitting it is fine for other OpenAI-compatible providers too.
      model: config.KIMI_MODEL || 'kimi-k2.7-code',
      messages,
      max_tokens: config.CHAT_MAX_TOKENS || 1024,
    }),
    signal,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`Kimi API ${res.status}: ${body.slice(0, 200)}`);
  }
  const data = await res.json();
  return (data.choices?.[0]?.message?.content || '').trim();
}
