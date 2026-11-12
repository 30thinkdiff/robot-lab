/* Real-data research interactions: no fabricated phase labels or model tensors. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.REPLAY_PLAYWRIGHT||'playwright');
const base=process.env.REPLAY_URL||'http://127.0.0.1:8871';
const out=path.resolve(__dirname,'../.local/review');fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.REPLAY_BROWSER?{executablePath:process.env.REPLAY_BROWSER}:{}),args:['--no-sandbox']});
 const result={checks:[],errors:[]};
 try{
  const context=await browser.newContext({viewport:{width:1920,height:1080}}),page=await context.newPage();
  page.on('pageerror',e=>result.errors.push(e.message));page.on('console',m=>{if(m.type()==='error')result.errors.push(m.text());});
  const catalog=await (await context.request.get(base+'/api/catalog')).json();
  const hybrid=method=>catalog.runs.find(r=>r.experiment==='混合 Corrector'&&r.method===method&&r.task===1&&r.init===0&&!r.forced);
  const a=hybrid('A'),b=hybrid('B'),c=hybrid('C');
  const jump=async n=>{await page.locator('#stepInput').fill(String(n));await page.locator('#stepInput').press('Tab');await page.waitForFunction(n=>Number(document.querySelector('#clockStep').textContent)===n,n);};
  const select=async r=>{await page.locator('#clearFilters').click();await page.locator('#modelFilter').selectOption(r.model);await page.locator('#methodFilter').selectOption(r.method);await page.locator('#taskFilter').selectOption(String(r.task));await page.locator('#initFilter').selectOption(String(r.init));await page.locator('#seedFilter').selectOption(String(r.seed));await page.locator(`.run-item[data-id="${r.id}"]`).click();await page.waitForFunction(id=>document.querySelector('#videoA').getAttribute('src')==='/api/video/'+id,r.id);};
  await page.goto(base);await page.waitForFunction(()=>document.querySelector('#videoA').readyState>=2);await jump(61);
  await page.locator('#evidenceDetails > summary').click();await page.waitForSelector('#updateHeatmap svg');
  assert.equal(await page.locator('#commandHeatmap rect[data-step]').count(),175);
  assert((await page.locator('#commandHeatmap svg').boundingBox()).height<=401);
  assert.equal(await page.locator('#updateHeatmap rect').count(),200);
  const diagnostics=await (await context.request.get(base+`/api/diagnostics?id=${b.id}&step=61`)).json();
  assert.equal(diagnostics.heatmaps.updates.length,4);for(const row of diagnostics.heatmaps.updates)assert.deepEqual(row.values.flatMap((v,i)=>v!==0?[i]:[]),[0,2,4,6,8,10,12,14,16,18]);
  assert((await page.locator('#evidenceCards').innerText()).includes('不能自动认定'));
  await page.screenshot({path:path.join(out,'real_action_update_heatmaps.png'),fullPage:true});result.checks.push('real 25x7 submitted commands and 4x50 OGG update norms, A10 support, fact-only diagnostic cards');
  await select(c);await page.waitForFunction(()=>document.querySelector('#evidenceCards').textContent.includes('存在监测空窗'));
  await page.locator('#evidenceCards article').filter({hasText:'存在监测空窗'}).locator('button').click();assert.equal(Number(await page.locator('#clockStep').innerText()),51);
  result.checks.push('real 155/256 coverage evidence jumps to first gap step51; no causal label fabricated');
  await select(b);await page.locator('#compareTab').click();await page.waitForFunction(()=>document.querySelector('#videoB').readyState>=2);
  await page.locator('#pairSelect').selectOption(a.id);await page.waitForFunction(()=>document.querySelector('#labelB').textContent.startsWith('A /'));
  await page.locator('#anchorTools summary').click();await page.locator('#anchorA').fill('61');await page.locator('#anchorB').fill('85');await page.locator('#anchorLabel').fill('工程验收：不同事件位置，无阶段真值');await page.locator('#applyAnchor').click();
  await page.waitForFunction(()=>Math.abs(document.querySelector('#videoA').currentTime-3.051)<.01&&Math.abs(document.querySelector('#videoB').currentTime-4.251)<.01);
  assert((await page.locator('#alignmentNotice').textContent()).includes('非自动阶段对齐'));
  await page.locator('#inspectLane').selectOption('b');assert.equal(Number(await page.locator('#inspectStep').textContent()),85);
  await page.waitForFunction(()=>document.querySelector('#evidencePosition').textContent.includes('原始step 85'));
  await page.waitForSelector('#commandHeatmap rect[data-step="86"]');await page.locator('#commandHeatmap rect[data-step="86"]').first().click();assert.equal(Number(await page.locator('#clockStep').textContent()),62);assert.equal(Number(await page.locator('#inspectStep').textContent()),86);
  await page.screenshot({path:path.join(out,'manual_anchor_real_pair.png'),fullPage:true});result.checks.push('manual anchors map primary61 to comparison85; comparison inspector and heatmap navigation retain original steps');
  const exportedResponse=page.waitForResponse(r=>r.url().endsWith('/api/export')&&r.request().method()==='POST');await page.locator('#exportBtn').click();const exported=await(await exportedResponse).json();await page.locator('#exportDialog').waitFor({state:'visible'});
  const provenance=await(await context.request.get(`${base}/api/export-file/${exported.export}/provenance.json`)).json();assert.deepEqual(provenance.environment_steps,[62,86]);assert.equal(provenance.manual_alignment.a,61);assert.equal(provenance.manual_alignment.b,85);
  assert(exported.files.includes('B_step_0086.png'));assert(exported.files.includes('A_heatmaps.json'));assert(exported.files.includes('B_heatmaps.json'));assert.equal(provenance.sources[1].frame_export.pts_seconds,4.3);await page.locator('#closeExport').click();result.checks.push('manual-offset export preserves 62/86 original steps, anchor note, correct PTS4.3, real PNGs and numeric heatmap arrays');
  await jump(61);await page.locator('#play').click();await page.waitForTimeout(450);await page.locator('#play').click();const position=Number(await page.locator('#clockStep').textContent());assert(position>61);
  await page.waitForFunction(n=>Math.abs(document.querySelector('#videoB').currentTime-((n+24)/20+.001))<.01,position);result.checks.push('manual-offset playback preserves both recorded frame mappings without time stretching');
  await page.locator('#anchorB').fill('10');await page.locator('#anchorLabel').fill('工程验收：负偏移');await page.locator('#applyAnchor').click();await jump(0);
  assert((await page.locator('#readout').textContent()).includes('尚未到达'));assert((await page.locator('#endedB').textContent()).includes('首帧仅作占位'));assert((await page.locator('#rawStep').textContent())==='[]');
  await page.locator('#clearAnchor').click();await jump(65);await page.waitForFunction(()=>Math.abs(document.querySelector('#videoB').currentTime-3.251)<.01);
  assert((await page.locator('#pairNotice').textContent()).startsWith('按相同环境步'));
  result.checks.push('before-start comparison exposes no stale score; reset restores same-step videos');
  const invalid=await context.request.post(base+'/api/export',{data:{ids:[b.id,a.id],step:61,alignment:{a:99999,b:85,label:'invalid'}}});assert.equal(invalid.status(),400);
  assert.deepEqual(result.errors,[]);result.checks.push('invalid anchors rejected before export; no browser errors');result.status='passed';
 }catch(e){result.status='failed';result.failure=e.stack;throw e;}
 finally{fs.writeFileSync(path.join(out,'research_acceptance.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
