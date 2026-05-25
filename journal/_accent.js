// life-os :: daily accent picker + PRNG.
// Importable as ES module OR usable as a classic script (exposes window.lifeOs).
//
// Spec: PLAN.md §2.3 — accent comes from a 12-color palette, picked by date
// seed; cannot repeat within 5 days. Also exposes a date-seeded mulberry32
// PRNG so pages (today.html, etc.) can deterministically pick widgets and
// prompts.
//
// Usage in a page:
//   <script src="_accent.js"></script>
//   <script>lifeOs.applyAccent();</script>
// or
//   import { applyAccent, makeRng, accentPalette } from './_accent.js';

(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.lifeOs = api;
}(typeof self !== 'undefined' ? self : this, function () {

  const ACCENTS = [
    { name: 'acid',     hex: '#b5fa3a' },
    { name: 'electric', hex: '#5c9eff' },
    { name: 'hotpink',  hex: '#ff3d9a' },
    { name: 'amber',    hex: '#ffb020' },
    { name: 'lavender', hex: '#b794ff' },
    { name: 'rust',     hex: '#ff6b35' },
    { name: 'cyan',     hex: '#4dd4ff' },
    { name: 'lime',     hex: '#d9ff4d' },
    { name: 'coral',    hex: '#ff7a7a' },
    { name: 'violet',   hex: '#8b5cf6' },
    { name: 'mint',     hex: '#6ff5c4' },
    { name: 'gold',     hex: '#f5c842' },
  ];

  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = seed;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return (((t ^ (t >>> 14)) >>> 0)) / 4294967296;
    };
  }

  function dateSeed(d) {
    d = d || new Date();
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return parseInt(`${y}${m}${day}`, 10);
  }

  function pickAccent(d) {
    d = d || new Date();
    // Anti-monotony: skip accents used in the past 5 days.
    const seed = dateSeed(d);
    const banned = new Set();
    for (let i = 1; i <= 5; i++) {
      const prev = new Date(d); prev.setDate(d.getDate() - i);
      const prevIdx = Math.floor(mulberry32(dateSeed(prev))() * ACCENTS.length);
      banned.add(prevIdx);
    }
    // Re-seed (seed + offset) and re-pick rather than stepping +1 on collision,
    // so consecutive days that hit the same banned slot don't all land on the
    // same fallback. Smoother distribution.
    let guard = 0;
    let idx;
    do {
      const rng = mulberry32(seed + guard);
      idx = Math.floor(rng() * ACCENTS.length);
      guard++;
    } while (banned.has(idx) && guard < 30);
    return ACCENTS[idx];
  }

  function applyAccent(d) {
    const a = pickAccent(d);
    document.documentElement.style.setProperty('--accent', a.hex);
    document.documentElement.dataset.accent = a.name;
    return a;
  }

  function makeRng(d, offset) {
    return mulberry32(dateSeed(d) + (offset || 0));
  }

  // Format helpers used by status line.
  const WEEKDAYS = ['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'];
  const MONTHS = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'];

  function formatDate(d) {
    d = d || new Date();
    return `${MONTHS[d.getMonth()]} ${String(d.getDate()).padStart(2,'0')} ${d.getFullYear()}`;
  }
  function formatWeekday(d) {
    d = d || new Date();
    return WEEKDAYS[d.getDay()];
  }
  function formatTime(d) {
    d = d || new Date();
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    const ss = String(d.getSeconds()).padStart(2, '0');
    return `${hh}:${mm}:${ss}`;
  }

  function mountStatusLine(el, opts) {
    opts = opts || {};
    const page = opts.page || 'JOURNAL';
    const version = opts.version || 'v0.1';
    const now = new Date();
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || '';
    function render() {
      const n = new Date();
      el.innerHTML =
        `<span>${page}</span>` +
        `<span class="sep">·</span>` +
        `<span>${formatDate(n)}</span>` +
        `<span class="sep">·</span>` +
        `<span>${formatWeekday(n)}</span>` +
        `<span class="sep">·</span>` +
        `<span><span class="mono-time">${formatTime(n)}</span></span>` +
        `<span class="sep">·</span>` +
        `<span><span class="live-dot"></span>live</span>` +
        `<span class="sep">·</span>` +
        `<span class="ver">${version}</span>`;
    }
    render();
    setInterval(render, 1000);
  }

  return {
    accentPalette: ACCENTS,
    mulberry32: mulberry32,
    dateSeed: dateSeed,
    pickAccent: pickAccent,
    applyAccent: applyAccent,
    makeRng: makeRng,
    formatDate: formatDate,
    formatWeekday: formatWeekday,
    formatTime: formatTime,
    mountStatusLine: mountStatusLine,
  };
}));
