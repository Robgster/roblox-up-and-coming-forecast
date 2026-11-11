// Roblox Up-and-Coming top-10 forecaster (Node 22, standard library only, no network).
//
// How it works: every pair of past reads (a, b) of the chart is a lesson in "a game at rank r at time a, rising or
// falling this fast, was at place k at time b". For each game on the chart now, the forecast weighs those lessons by
// how alike they are (rank on a log scale, recent rank momentum, the game's age among the chart's games, and how far
// ahead b was compared with the hours of the scored window) and turns them into a probability for each of the 10
// places. Every game on the chart is named at every place, so the share a place leaves unnamed goes only to games that
// join the chart later; that share is learned from the history too (how often a place was held by a game that was not
// on the chart at the earlier read).
//
// Run:      mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
// Backtest: MAKEFUN_INPUT_DIR=input node main.ts --backtest   (scores past cutoffs of the history itself)

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const HOUR = 3600e3;
const FLOOR = 1e-4;

// Model settings (picked by the backtest, see README).
const DEFAULTS = {
  sigRank: 0.7, // kernel width on ln(rank)
  sigMom: 0.4, // kernel width on momentum (ln of rank 12-36h earlier over rank now)
  sigAge: 0.6, // kernel width on age (0 = newest game on the chart, 1 = oldest)
  sigGrowth: 0.6, // kernel width on ln(player-count growth) over the same span
  sigLead: 1.0, // kernel width on ln(hours ahead)
  alpha: 1.5, // pseudo-count of the place prior per game
  maxLead: 110, // longest pair gap used, hours
  momLo: 12, // momentum looks back at the latest read at least this many hours before ...
  momHi: 36, // ... and at most this many
  newMom: 1.5, // momentum given to a game that was not on the earlier chart
  newGrowth: 0.5, // growth given to a game with no earlier player count
  stayMix: 0.4, // share of each place blended toward "the chart stays put" (the game now at #k, and its neighbours)
  stayDecay: 24, // hours over which that stay share fades as the window moves away from the last read
};

function loadHistory(dir) {
  const lines = readFileSync(join(dir, 'history.ndjson'), 'utf8').split('\n').filter((l) => l.trim() !== '');
  const reads = [];
  for (const line of lines) {
    const r = JSON.parse(line);
    if (r.ok === false || !Array.isArray(r.up_and_coming) || r.up_and_coming.length === 0) continue;
    const list = [...r.up_and_coming].sort((x, y) => x.rank - y.rank).map((g) => g.universe_id);
    const playing = new Map();
    for (const g of r.up_and_coming) if (typeof g.playing === 'number') playing.set(g.universe_id, g.playing);
    const rank = new Map();
    list.forEach((u, i) => { if (!rank.has(u)) rank.set(u, i + 1); });
    reads.push({ t: Date.parse(r.at), list, rank, playing });
  }
  reads.sort((x, y) => x.t - y.t);
  return reads;
}

// Momentum and age of every game on read i.
function featuresOf(reads, i, s) {
  const cur = reads[i];
  let prev = null;
  for (let j = i - 1; j >= 0; j--) {
    const dt = (cur.t - reads[j].t) / HOUR;
    if (dt < s.momLo) continue;
    if (dt > s.momHi) break;
    prev = reads[j];
    break;
  }
  const n = cur.list.length;
  const ids = [...cur.list].sort((x, y) => (x < y ? 1 : x > y ? -1 : 0)); // newest (highest id) first
  const agePos = new Map();
  ids.forEach((u, k) => agePos.set(u, n > 1 ? k / (n - 1) : 0));
  const out = new Map();
  for (const u of cur.list) {
    const r = cur.rank.get(u);
    let mom = 0;
    let gr = 0;
    if (prev) {
      const pr = prev.rank.get(u);
      mom = pr === undefined ? s.newMom : Math.log(pr / r);
      const pn = cur.playing.get(u), pp = prev.playing.get(u);
      gr = pn !== undefined && pp !== undefined ? Math.log((pn + 50) / (pp + 50)) : s.newGrowth;
    }
    out.set(u, { lr: Math.log(r), mom, age: agePos.get(u), gr });
  }
  return out;
}

// Lessons from every pair (a, b) with b at or before `until`.
function buildLessons(reads, until, s) {
  const lessons = []; // { lr, mom, age, llead, place }
  const newcomer = []; // { llead, places: [bool x10] }
  const feats = reads.map((_, i) => featuresOf(reads, i, s));
  for (let a = 0; a < reads.length; a++) {
    if (reads[a].t > until) break;
    for (let b = a + 1; b < reads.length; b++) {
      if (reads[b].t > until) break;
      const lead = (reads[b].t - reads[a].t) / HOUR;
      if (lead > s.maxLead) break;
      if (lead < 0.5) continue;
      const llead = Math.log(lead);
      const fa = feats[a];
      for (const [u, f] of fa) {
        const pb = reads[b].rank.get(u);
        lessons.push({ lr: f.lr, mom: f.mom, age: f.age, gr: f.gr, llead, place: pb !== undefined && pb <= 10 ? pb : 0 });
      }
      const flags = [];
      for (let k = 1; k <= 10; k++) {
        const occ = reads[b].list[k - 1];
        flags.push(occ !== undefined && !reads[a].rank.has(occ));
      }
      newcomer.push({ llead, flags });
    }
  }
  return { lessons, newcomer };
}

function forecast(reads, fromIdx, windowFrom, slots, s, lessonsPack) {
  const base = reads[fromIdx];
  const { lessons, newcomer } = lessonsPack;
  // Hours ahead of each scored slot, measured from the last read.
  const leads = [];
  for (let h = 0; h < slots; h++) leads.push(Math.max(0.5, (windowFrom + (h + 0.5) * HOUR - base.t) / HOUR));
  const leadW = (llead) => {
    let w = 0;
    for (const L of leads) { const d = (llead - Math.log(L)) / s.sigLead; w += Math.exp(-0.5 * d * d); }
    return w / leads.length;
  };
  // Lead weights depend only on the pair, cache by value.
  const cache = new Map();
  const lw = (x) => { let v = cache.get(x); if (v === undefined) { v = leadW(x); cache.set(x, v); } return v; };

  // Share of each place held by a game that was not on the earlier chart.
  const newShare = [];
  for (let k = 0; k < 10; k++) {
    let num = 0.05, den = 1; // small prior
    for (const c of newcomer) { const w = lw(c.llead); den += w; if (c.flags[k]) num += w; }
    newShare.push(Math.min(0.6, num / den));
  }
  // Prior over places for a game: flat-ish, small.
  const feats = featuresOf(reads, fromIdx, s);
  const games = base.list;
  const raw = new Map();
  for (const u of games) {
    const f = feats.get(u);
    const acc = new Float64Array(11);
    let tot = 0;
    for (const L of lessons) {
      const dr = (L.lr - f.lr) / s.sigRank;
      if (dr > 4 || dr < -4) continue;
      const dm = (L.mom - f.mom) / s.sigMom;
      const da = (L.age - f.age) / s.sigAge;
      const dg = (L.gr - f.gr) / s.sigGrowth;
      const w = Math.exp(-0.5 * (dr * dr + dm * dm + da * da + dg * dg)) * lw(L.llead);
      acc[L.place] += w;
      tot += w;
    }
    const p = new Float64Array(11);
    const r = base.rank.get(u);
    for (let k = 1; k <= 10; k++) {
      // prior: a little mass near the current rank
      const prior = Math.exp(-Math.abs(Math.log(k / r)) * 2) / (games.length * 0.5);
      p[k] = (acc[k] + s.alpha * prior) / (tot + s.alpha);
    }
    raw.set(u, p);
  }
  // Normalise each place to 1 - newcomer share, keep the 100 most likely games.
  const probabilities = [];
  const placeP = [];
  for (let k = 1; k <= 10; k++) {
    let rows = games.map((u) => ({ u, p: raw.get(u)[k] }));
    rows.sort((x, y) => y.p - x.p || (x.u < y.u ? -1 : 1));
    rows = rows.slice(0, 100);
    const sum = rows.reduce((t, x) => t + x.p, 0);
    const target = 1 - newShare[k - 1];
    // Blend toward persistence: early hours of the window sit close to the last read, so the game now at #k
    // (and, less, its neighbours) keeps an extra share that fades with the hours ahead.
    const fade = leads.reduce((t, L) => t + Math.exp(-L / s.stayDecay), 0) / leads.length;
    const mix = s.stayMix * fade;
    const stay = new Map();
    const atK = games[k - 1], up = games[k - 2], down = games[k];
    if (atK !== undefined) stay.set(atK, 0.6);
    if (up !== undefined) stay.set(up, 0.2);
    if (down !== undefined) stay.set(down, 0.2);
    const m = new Map();
    for (const x of rows) {
      const p = Math.round(((1 - mix) * (x.p / sum) + mix * (stay.get(x.u) ?? 0)) * target * 1e9) / 1e9;
      if (p <= 0) continue;
      m.set(x.u, p);
    }
    // rounding must never push a place over 1
    let tot = 0; for (const v of m.values()) tot += v;
    if (tot > 1) for (const [u, v] of m) m.set(u, Math.floor((v / tot) * 1e9) / 1e9);
    placeP.push(m);
    for (const [u, p] of m) probabilities.push({ universe_id: u, place: k, p });
  }
  // Top 10: greedy, best remaining (place, game) pair first.
  const pairs = [];
  placeP.forEach((m, k) => { for (const [u, p] of m) pairs.push({ k, u, p }); });
  pairs.sort((x, y) => y.p - x.p || x.k - y.k || (x.u < y.u ? -1 : 1));
  const top = new Array(10).fill(null);
  const used = new Set();
  for (const x of pairs) { if (top[x.k] === null && !used.has(x.u)) { top[x.k] = x.u; used.add(x.u); } }
  for (let k = 0; k < 10; k++) if (top[k] === null) { const u = games.find((g) => !used.has(g)); top[k] = u; used.add(u); }
  return { probabilities, top10: top, placeP };
}

// Score a forecast against later reads, as the grader does.
function scoreAgainst(fc, read) {
  let total = 0;
  for (let k = 1; k <= 10; k++) {
    const occ = read.list[k - 1];
    if (occ === undefined) continue;
    const m = fc.placeP[k - 1];
    let p;
    if (m.has(occ)) p = m.get(occ);
    else {
      let s = 0; for (const v of m.values()) s += v;
      const unnamed = read.list.filter((u) => !m.has(u)).length;
      p = unnamed > 0 ? (1 - s) / unnamed : 0;
    }
    total += Math.log(Math.max(FLOOR, p));
  }
  return total;
}

function persistence(read) {
  const placeP = [];
  const c = read.list;
  for (let k = 1; k <= 10; k++) {
    const m = new Map();
    m.set(c[k - 1], 0.5);
    if (k === 1) { if (c[1] !== undefined) m.set(c[1], 0.3); } else { m.set(c[k - 2], 0.15); if (c[k] !== undefined) m.set(c[k], 0.15); }
    placeP.push(m);
  }
  return { placeP };
}

function backtest(reads, s, verbose) {
  // Cutoffs: each read from the 4th day on, window = next 72 hours.
  const t0 = reads[0].t;
  let sum = 0, sumB = 0, n = 0;
  for (let i = 0; i < reads.length; i++) {
    if (reads[i].t - t0 < 96 * HOUR) continue;
    const from = reads[i].t + 0.5 * HOUR;
    const later = reads.filter((r) => r.t > from && r.t <= from + 72 * HOUR);
    if (later.length < 3) continue;
    const pack = buildLessons(reads, reads[i].t, s);
    const fc = forecast(reads, i, from, 72, s, pack);
    const pb = persistence(reads[i]);
    const sc = later.reduce((t, r) => t + scoreAgainst(fc, r), 0) / later.length;
    const sb = later.reduce((t, r) => t + scoreAgainst(pb, r), 0) / later.length;
    if (verbose) console.log(new Date(reads[i].t).toISOString(), 'reads', later.length, 'model', sc.toFixed(3), 'baseline', sb.toFixed(3));
    sum += sc; sumB += sb; n++;
  }
  return { model: sum / n, baseline: sumB / n, n };
}

function main() {
  const reads = loadHistory(INPUT_DIR);
  const args = process.argv.slice(2);
  if (args.includes('--backtest')) {
    const r = backtest(reads, DEFAULTS, true);
    console.log('mean over', r.n, 'cutoffs: model', r.model.toFixed(4), 'baseline', r.baseline.toFixed(4));
    return;
  }
  if (args.includes('--tune')) {
    const grid = {
      sigRank: [0.4, 0.5, 0.7, 1.0],
      sigMom: [0.4, 0.6, 1.0, 2.0],
      sigAge: [0.4, 0.6, 1.0, 2.0],
      sigGrowth: [0.3, 0.6, 1.0, 100],
      sigLead: [0.4, 0.6, 1.0],
      alpha: [1.5, 3, 6],
      stayMix: [0, 0.1, 0.25, 0.4, 0.6],
      stayDecay: [6, 12, 24, 48],
    };
    let best = { ...DEFAULTS };
    let bestScore = backtest(reads, best, false).model;
    console.log('start', bestScore.toFixed(4));
    for (let pass = 0; pass < 2; pass++) {
      for (const key of Object.keys(grid)) {
        for (const v of grid[key]) {
          const s = { ...best, [key]: v };
          const sc = backtest(reads, s, false).model;
          if (sc > bestScore + 1e-9) { bestScore = sc; best = s; console.log(key, v, sc.toFixed(4)); }
        }
      }
    }
    console.log('best', JSON.stringify(best), bestScore.toFixed(4));
    return;
  }
  const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8'));
  const windowFrom = Date.parse(task.window.from);
  const slots = task.window.slots ?? 72;
  const until = Date.parse(task.history_until);
  let last = reads.length - 1;
  while (last > 0 && reads[last].t >= until) last--;
  const pack = buildLessons(reads, reads[last].t, DEFAULTS);
  const fc = forecast(reads, last, windowFrom, slots, DEFAULTS, pack);
  writeFileSync(join(OUTPUT_DIR, task.output ?? 'forecast.json'), `${JSON.stringify({ probabilities: fc.probabilities, top10: fc.top10 })}\n`);
}

main();
