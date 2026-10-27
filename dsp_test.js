"use strict";
// ===== DSP 核心（与最终 HTML 内联代码一致）=====
const SAMPLE_RATE = 48000;
const CHIRP_MS = 30;
const N_FFT = 16384;

// 生成复 chirp：re=sin 分量（用于播放），im=cos 分量（用于匹配滤波参考的正交分量）
function makeChirp(f0, f1, ms, sr){
  const n = Math.round(ms / 1000 * sr);
  const re = new Float32Array(n), im = new Float32Array(n);
  const T = n / sr;
  for (let i = 0; i < n; i++){
    const t = i / sr;
    const phi = 2 * Math.PI * (f0 * t + (f1 - f0) / (2 * T) * t * t);
    const w = 0.5 * (1 - Math.cos(2 * Math.PI * i / (n - 1)));   // Hann 窗
    re[i] = Math.sin(phi) * w;
    im[i] = Math.cos(phi) * w;
  }
  return { re, im, length: n };
}

function fft(re, im){
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++){
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j){ let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1){
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len){
      let cr = 1, ci = 0;
      for (let j = 0; j < (len >> 1); j++){
        const a = i + j, b = i + j + (len >> 1);
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
}

function ifft(re, im){
  const n = re.length;
  for (let i = 0; i < n; i++) im[i] = -im[i];
  fft(re, im);
  for (let i = 0; i < n; i++){ im[i] = -im[i]; re[i] /= n; im[i] /= n; }
}

// 复 chirp 的 FFT（返回未共轭的频谱）
function makeRefFFT(chirp){
  const re = new Float32Array(N_FFT), im = new Float32Array(N_FFT);
  for (let i = 0; i < chirp.length; i++){ re[i] = chirp.re[i]; im[i] = chirp.im[i]; }
  fft(re, im);
  return { re, im };
}

// 匹配滤波，返回「幅度包络」|IFFT(FFT(buf)·conj(FFT(ref)))|
function matchedFilterEnv(buf, refRe, refIm){
  const n = buf.length;
  const re = new Float32Array(n), im = new Float32Array(n);
  for (let i = 0; i < n; i++) re[i] = buf[i];
  fft(re, im);
  for (let i = 0; i < n; i++){
    const a = re[i], b = im[i];
    re[i] = a * refRe[i] + b * refIm[i];
    im[i] = b * refRe[i] - a * refIm[i];
  }
  ifft(re, im);
  const env = new Float32Array(n);
  for (let i = 0; i < n; i++) env[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
  return env;
}

// 找最早超过阈值的局部峰（作用于平滑包络），抛物线插值钳位到 ±0.5 采样
function findPeak(env, chirpLen){
  const n = env.length;
  let ymax = 0;
  for (let i = 0; i < n - chirpLen; i++) if (env[i] > ymax) ymax = env[i];
  if (ymax <= 0) return null;
  const thr = ymax * 0.4;
  for (let i = 1; i < n - chirpLen - 1; i++){
    if (env[i] > thr && env[i] >= env[i - 1] && env[i] >= env[i + 1]){
      let delta = 0;
      const a = env[i - 1], b = env[i], c = env[i + 1];
      const denom = a - 2 * b + c;
      if (denom !== 0){
        delta = 0.5 * (a - c) / denom;
        if (delta > 0.5) delta = 0.5;
        if (delta < -0.5) delta = -0.5;
      }
      return { index: i, delta, pos: i + delta, value: env[i], ymax, thr };
    }
  }
  return null;
}

// ===== 测试 =====
let pass = 0, fail = 0;
function check(name, cond){ console.log((cond ? 'PASS' : 'FAIL') + '  ' + name); if (cond) pass++; else fail++; }

// 1. FFT 往返
{
  const n = 1024, re = new Float32Array(n), im = new Float32Array(n);
  for (let i = 0; i < n; i++){ re[i] = Math.random() * 2 - 1; im[i] = Math.random() * 2 - 1; }
  const oRe = re.slice(), oIm = im.slice();
  fft(re, im); ifft(re, im);
  let err = 0;
  for (let i = 0; i < n; i++) err = Math.max(err, Math.abs(re[i] - oRe[i]), Math.abs(im[i] - oIm[i]));
  check('FFT 往返误差 < 1e-4', err < 1e-4);
}

// 2. 包络峰值位置
{
  const up = makeChirp(4000, 10000, CHIRP_MS, SAMPLE_RATE);
  const ref = makeRefFFT(up);
  const buf = new Float32Array(N_FFT);
  const pos = 5000;
  for (let i = 0; i < up.length; i++) buf[pos + i] = up.re[i];
  const env = matchedFilterEnv(buf, ref.re, ref.im);
  const peak = findPeak(env, up.length);
  check('包络峰值位置 ≈ 5000（实际 ' + (peak ? peak.pos.toFixed(2) : 'null') + '）',
    peak && Math.abs(peak.pos - pos) < 1.5);
}

// 3. 最早峰 + 回声（回声更强但更晚，应取直接路径）
{
  const up = makeChirp(4000, 10000, CHIRP_MS, SAMPLE_RATE);
  const ref = makeRefFFT(up);
  const buf = new Float32Array(N_FFT);
  const direct = 4000, echo = 7000;
  for (let i = 0; i < up.length; i++){
    buf[direct + i] += up.re[i];
    buf[echo + i] += up.re[i] * 1.3;
  }
  const env = matchedFilterEnv(buf, ref.re, ref.im);
  const peak = findPeak(env, up.length);
  check('取最早峰（回声更强也选直接路径 4000，实际 ' + (peak ? peak.pos.toFixed(2) : 'null') + '）',
    peak && Math.abs(peak.pos - direct) < 1.5);
}

// 4. 上行参考对下行 chirp 响应远弱
{
  const up = makeChirp(4000, 10000, CHIRP_MS, SAMPLE_RATE);
  const down = makeChirp(10000, 4000, CHIRP_MS, SAMPLE_RATE);
  const refUp = makeRefFFT(up);
  const b1 = new Float32Array(N_FFT), b2 = new Float32Array(N_FFT);
  const pos = 5000;
  for (let i = 0; i < up.length; i++){ b1[pos + i] = down.re[i]; b2[pos + i] = up.re[i]; }
  const e1 = matchedFilterEnv(b1, refUp.re, refUp.im);
  const e2 = matchedFilterEnv(b2, refUp.re, refUp.im);
  let m1 = 0, m2 = 0;
  for (let i = 0; i < e1.length; i++){ m1 = Math.max(m1, e1[i]); m2 = Math.max(m2, e2[i]); }
  check('上行参考对下行 chirp 响应远弱（' + (m1/m2).toFixed(3) + ' < 0.3）', m1 / m2 < 0.3);
}

console.log('==== 通过 ' + pass + ' / ' + (pass + fail) + ' ====');
process.exit(fail === 0 ? 0 : 1);
