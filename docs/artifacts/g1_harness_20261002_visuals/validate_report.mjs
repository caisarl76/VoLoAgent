// Check the saved report in a real local browser; never connect to robot services.
import {spawn} from 'node:child_process';
import {mkdtemp, readFile, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, dirname} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const output = dirname(fileURLToPath(import.meta.url));
const profile = await mkdtemp(join(tmpdir(), 'g1-saved-report-chrome-'));
const browser = spawn('/usr/bin/google-chrome', [
  '--headless', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
  '--disable-background-networking', '--disable-sync', '--metrics-recording-only',
  '--no-first-run', '--no-default-browser-check', '--remote-debugging-address=127.0.0.1',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`,
  pathToFileURL(join(output, 'report.html')).href,
], {detached: true, stdio: 'ignore'});
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
let socket;
const errors = [];
const remoteRequests = [];
const pending = new Map();
let nextId = 0;
const summary = {browser: 'local headless Chrome', robot_connection: false};

function require(condition, message) {
  if (!condition) throw new Error(message);
}

async function command(method, params = {}) {
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {pending.delete(id); reject(new Error(`Timeout: ${method}`));}, 10000);
    pending.set(id, {
      resolve: result => {clearTimeout(timer); resolve(result);},
      reject: error => {clearTimeout(timer); reject(error);},
    });
    socket.send(JSON.stringify({id, method, params}));
  });
}

async function evaluate(expression) {
  const result = await command('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}

async function readyImage() {
  return evaluate(`new Promise((resolve,reject)=>{
    const image=document.getElementById('frame');
    if(image.complete&&image.naturalWidth>0){resolve(true);return;}
    image.onload=()=>resolve(true);image.onerror=()=>reject(Error('Frame image failed'));
  })`);
}

async function screenshot(name) {
  const image = await command('Page.captureScreenshot', {format: 'png', captureBeyondViewport: false});
  await writeFile(join(output, name), Buffer.from(image.data, 'base64'));
}

try {
  const deadline = Date.now() + 15000;
  let port;
  while (Date.now() < deadline) {
    try {port = Number((await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]); break;}
    catch {await pause(100);}
  }
  require(port, 'Browser did not become ready');
  const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  const target = targets.find(t => t.type === 'page' && t.url.endsWith('/report.html'));
  require(target, 'Report page not found');
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, {once: true});
    socket.addEventListener('error', reject, {once: true});
  });
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.id) {
      const handler = pending.get(message.id);
      if (!handler) return;
      pending.delete(message.id);
      if (message.error) handler.reject(new Error(JSON.stringify(message.error)));
      else handler.resolve(message.result);
    } else if (message.method === 'Runtime.exceptionThrown') errors.push(message.params);
    else if (message.method === 'Network.requestWillBeSent') {
      const url = message.params.request.url;
      if (/^https?:/.test(url)) remoteRequests.push(url);
    }
  });
  await command('Runtime.enable');
  await command('Page.enable');
  await command('Network.enable');
  await command('Emulation.setDeviceMetricsOverride', {width: 1440, height: 2400, deviceScaleFactor: 1, mobile: false});
  await command('Page.reload');
  let loaded = false;
  for (let i = 0; i < 100; i++) {
    loaded = await evaluate(`document.readyState==='complete'&&document.getElementById('frame-counter')?.textContent.includes('1 / 44')`);
    if (loaded) break;
    await pause(50);
  }
  require(loaded, 'Default corrected trial failed to render');
  await readyImage();
  summary.default_corrected_frames = await evaluate(`Number(document.getElementById('scrubber').max)+1`);
  require(summary.default_corrected_frames === 44, 'Expected 44 corrected frames');
  await screenshot('report-desktop.png');
  await evaluate(`document.getElementById('original').click();document.getElementById('next').click()`);
  await readyImage();
  const original = await evaluate(`({raw:document.getElementById('raw-status').textContent,monitor:document.getElementById('monitor-status').textContent,caption:document.getElementById('frame-counter').textContent,verdict:document.getElementById('verdict').textContent})`);
  require(original.raw === 'complete' && original.monitor === 'in progress', 'Two-frame confirmation distinction lost');
  require(original.caption.includes('2 / 3') && original.verdict.includes('False completion'), 'Original evidence controls failed');
  await evaluate(`document.getElementById('next').click()`);
  require(await evaluate(`document.getElementById('monitor-status').textContent==='complete'&&document.getElementById('next').disabled`), 'Final original confirmation failed');
  summary.original_confirmation = 'Second raw-complete frame confirms the false success';
  await evaluate(`document.getElementById('corrected').click();const s=document.getElementById('scrubber');s.value=43;s.dispatchEvent(new Event('input'))`);
  await readyImage();
  require(await evaluate(`document.getElementById('frame-counter').textContent.includes('44 / 44')&&document.getElementById('raw-status').textContent==='in progress'`), 'Last corrected frame failed');
  summary.corrected_last_frame = '44/44 remains in progress';
  await evaluate(`document.getElementById('previous').click()`);
  require(await evaluate(`document.getElementById('frame-counter').textContent.includes('43 / 44')`), 'Previous-frame control failed');
  summary.image_navigation = 'tabs, next, previous and range passed';
  await command('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 1, mobile: true});
  await evaluate(`window.scrollTo(0,0)`);
  await pause(100);
  const dimensions = await evaluate(`({width:innerWidth,scroll:document.documentElement.scrollWidth})`);
  require(dimensions.scroll <= dimensions.width, 'Mobile layout overflows horizontally');
  await screenshot('report-mobile.png');
  summary.mobile_width_px = dimensions.width;
  require(errors.length === 0, 'JavaScript errors detected');
  require(remoteRequests.length === 0, 'Report made an external HTTP request');
  summary.javascript_errors = errors.length;
  summary.external_page_http_requests = remoteRequests.length;
  summary.outcome = 'passed';
} catch (error) {
  summary.outcome = 'failed';
  summary.reason = error.message;
  process.exitCode = 1;
} finally {
  socket?.close();
  try {process.kill(-browser.pid, 'SIGTERM');} catch {}
  for (let i = 0; i < 30 && browser.exitCode === null && browser.signalCode === null; i++) await pause(100);
  if (browser.exitCode === null && browser.signalCode === null) {
    try {process.kill(-browser.pid, 'SIGKILL');} catch {}
  }
  await writeFile(join(output, 'browser-checks.json'), JSON.stringify(summary, null, 2) + '\n');
  console.log(JSON.stringify(summary));
}
