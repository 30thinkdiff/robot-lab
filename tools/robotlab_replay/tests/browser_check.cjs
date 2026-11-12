/* Optional end-to-end acceptance against real local experiments. No simulated curves. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require(process.env.REPLAY_PLAYWRIGHT||'playwright');
const base=process.env.REPLAY_URL||'http://127.0.0.1:8870';
const out=path.resolve(__dirname,'../.local/review');fs.mkdirSync(out,{recursive:true});

(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.REPLAY_BROWSER?{executablePath:process.env.REPLAY_BROWSER}:{}),args:['--no-sandbox']});
 const results={checks:[],errors:[]};
 try{
  const context=await browser.newContext({viewport:{width:1920,height:1080}});const page=await context.newPage();
  page.on('pageerror',e=>results.errors.push(e.message));page.on('console',m=>{if(m.type()==='error')results.errors.push(m.text())});
  await page.goto(base);await page.waitForFunction(()=>document.querySelector('#videoA').readyState>=2);
  const catalog=await (await context.request.get(base+'/api/catalog')).json();assert(catalog.runs.length>=1200);
  assert((await page.locator('#title').innerText()).includes('T-H1 → A10-H1'));results.checks.push(`${catalog.runs.length} real records indexed; default is genuine hybrid B failure`);
  const b=catalog.runs.find(r=>r.method==='B'&&r.task===1&&r.init===0&&!r.forced&&r.experiment==='混合 Corrector');
  const detail=await (await context.request.get(base+'/api/run?id='+b.id)).json();assert.equal(detail.metrics.success,false);
  const jump=async n=>{await page.locator('#stepInput').fill(String(n));await page.locator('#stepInput').press('Tab');await page.waitForFunction(n=>Number(document.querySelector('#clockStep').textContent)===n,n);};
  await jump(61);await page.waitForFunction(()=>Math.abs(document.querySelector('#videoA').currentTime-3.051)<.005);
  assert((await page.locator('#stepEvents').innerText()).includes('4次已记录引导'));assert((await page.locator('#rawStep').textContent()).includes('gradient_l2'));
  // Seeking to an already displayed frame need not emit a fresh presentation callback.
  await jump(60);await page.waitForFunction(()=>!document.querySelector('#videoA').seeking);
  const mediaTime=await page.evaluate(()=>new Promise((resolve,reject)=>{const v=document.querySelector('#videoA');const timer=setTimeout(()=>reject(Error('No target decoded video frame')),5000);function presented(now,metadata){if(Math.abs(metadata.mediaTime-3.05)<.0001){clearTimeout(timer);resolve(metadata.mediaTime);}else v.requestVideoFrameCallback(presented);}v.requestVideoFrameCallback(presented);document.querySelector('#stepInput').value='61';document.querySelector('#stepInput').dispatchEvent(new Event('change'));}));
  assert(Math.abs(mediaTime-3.05)<.0001);results.decoded_step61_media_time=mediaTime;
  const image=await page.locator('#videoA').evaluate(v=>{const c=document.createElement('canvas');c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0);return c.toDataURL('image/png').split(',')[1]});fs.writeFileSync(path.join(out,'browser_step61.png'),Buffer.from(image,'base64'));
  results.checks.push('UI step61 decodes recorded PTS3.05 and exposes 4 real OGG calls, gradient, target');
  await page.locator('#next').click();assert.equal(Number(await page.locator('#clockStep').innerText()),62);await page.locator('#prev').click();assert.equal(Number(await page.locator('#clockStep').innerText()),61);
  await page.keyboard.press('ArrowRight');assert.equal(Number(await page.locator('#clockStep').innerText()),62);
  await page.locator('#play').click();await page.waitForTimeout(650);await page.locator('#play').click();const paused=Number(await page.locator('#clockStep').innerText());assert(paused>62);await page.waitForTimeout(180);assert.equal(Number(await page.locator('#clockStep').innerText()),paused);
  await page.locator('#speed').selectOption('2');await page.locator('#seek').evaluate(el=>{el.value='61';el.dispatchEvent(new Event('input'));});assert.equal(Number(await page.locator('#clockStep').innerText()),61);
  results.checks.push('play/pause, speed, slider, forward/backward, keyboard shortcuts');
  await page.locator('#compareTab').click();await page.waitForFunction(()=>document.querySelector('#videoB').readyState>=2&&!document.querySelector('#paneB').classList.contains('hidden'));
  await jump(61);await page.waitForFunction(()=>Math.abs(document.querySelector('#videoB').currentTime-3.051)<.005);
  assert((await page.locator('#pairNotice').innerText()).includes('SHA一致'));assert((await page.locator('#pairTable').innerText()).includes('168'));assert((await page.locator('#pairTable').innerText()).includes('280'));
  await page.locator('#inspectLane').selectOption('b');assert((await page.locator('#rawStep').textContent()).includes('gradient_l2'));
  await jump(200);assert(await page.locator('#endedB').isVisible());assert((await page.locator('#readout').innerText()).includes('已结束'));await page.waitForFunction(()=>Math.abs(document.querySelector('#videoB').currentTime-8.351)<.01);
  await jump(61);await page.locator('#inspectLane').selectOption('a');await page.screenshot({path:path.join(out,'paired_1920.png'),fullPage:true});results.checks.push('real B failure vs A success, same-step dual video, complete cost table, ended run freezes with explicit label');
  await page.locator('#note').fill('人工复盘：第61步同检测目标，比较两种引导。');const response=page.waitForResponse(r=>r.url().endsWith('/api/export')&&r.request().method()==='POST');await page.locator('#exportBtn').click();const exported=await (await response).json();await page.locator('#exportDialog').waitFor({state:'visible'});
  assert(exported.files.includes('A_step_0061.png'));assert(exported.files.includes('B_step_0061.png'));const report=await context.request.get(`${base}/api/export-file/${exported.export}/provenance.json`);const provenance=await report.json();assert.equal(provenance.environment_step,61);assert.equal(provenance.sources[0].frame_export.pts_seconds,3.05);results.checks.push('real keyframes, SVG, event CSV, summary and provenance exported in independent output');await page.locator('#closeExport').click();
  await page.locator('#singleTab').click();const c=catalog.runs.find(r=>r.method==='C'&&r.task===1&&r.init===0&&!r.forced&&r.experiment==='混合 Corrector');await page.locator('#search').fill('A10-H1 → A10-H1');await page.locator(`.run-item[data-id="${c.id}"]`).click();await page.waitForFunction(()=>document.querySelector('#labelA').textContent.startsWith('C /'));
  const cdetail=await (await context.request.get(base+'/api/run?id='+c.id)).json();assert.equal(cdetail.coverage.max_gap_steps,19);await page.locator('#eventFilter').selectOption('gap');await page.locator('#events .event-row').first().click();assert((await page.locator('#readout').innerText()).includes('无到期评分'));assert(await page.locator('#plot rect[fill="#fbf0df"]').count()>0);await page.screenshot({path:path.join(out,'a10_gap_1920.png'),fullPage:true});results.checks.push('real A10 19-step gap, missing score and gap jump; no interpolated score');
  await page.locator('#clearFilters').click();await page.locator('#modelFilter').selectOption('MLP');await page.locator('#methodFilter').selectOption('ogg');await page.locator('#taskFilter').selectOption('1');
  const missing=catalog.runs.find(r=>r.model==='MLP'&&r.method==='ogg'&&r.task===1&&!r.forced);await page.locator(`.run-item[data-id="${missing.id}"]`).click();await page.waitForFunction(()=>document.querySelector('#missingA').classList.contains('hidden')===false);
  await jump(61);assert((await page.locator('#actions').innerText()).includes('未记录'));results.checks.push('real missing-video/partial-fields experiment remains usable without fabricated gradient or action');
  await page.locator('#clearFilters').click();await page.locator('#methodFilter').selectOption('H50');await page.locator('#taskFilter').selectOption('1');
  const baseline=catalog.runs.find(r=>r.model==='MLP'&&r.method==='H50'&&r.task===1&&!r.forced);await page.locator(`.run-item[data-id="${baseline.id}"]`).click();await page.waitForFunction(()=>document.querySelector('#title').textContent.includes('SmolVLA'));
  assert((await page.locator('#metrics').innerText()).includes('不适用'));assert((await page.locator('#readout').innerText()).includes('未启用Corrector'));results.checks.push('pureVLA baseline coverage is not applicable, never falsely reported as zero-percent blind monitoring');
  const range=await context.request.get(base+'/api/video/'+b.id,{headers:{Range:'bytes=0-31'}});assert.equal(range.status(),206);assert.equal((await range.body()).length,32);
  const invalid=await context.request.get(base+'/api/run?id=../../etc/passwd');assert.equal(invalid.status(),400);
  const csrf=await context.request.post(base+'/api/export',{headers:{Origin:'https://untrusted.example'},data:{ids:[b.id],step:61}});assert.equal(csrf.status(),400);results.checks.push('video byte Range, traversal rejection, cross-origin export rejection');
  // 用户截图回归：原正式A10回合必须有自己的录制绑定，不能仅凭与Hybrid C的内容SHA相同就联动。
  const originalA10=catalog.runs.find(r=>r.model==='A10-H1'&&r.method==='ogg'&&r.task===1&&r.init===0&&r.seed===1000&&!r.forced);
  const originalDetail=await (await context.request.get(base+'/api/run?id='+originalA10.id)).json();
  if(originalDetail.video.alignment==='recording_verified'){
   await page.locator('#baselinePreset').click();await page.waitForFunction(()=>document.querySelectorAll('#runList .run-item').length===6);
   assert((await page.locator('#configurationText').textContent()).includes('纯SmolVLA'));
   const selected=catalog.runs.find(r=>r.model==='MLP'&&r.method==='H50'&&r.task===2&&r.init===2&&r.seed===1000&&!r.forced);
   await page.locator(`.run-item[data-id="${selected.id}"]`).click();await page.waitForFunction(()=>document.querySelector('#labelA').textContent==='H50 / MLP'&&document.querySelector('#videoA').readyState>=2);
   await page.locator('#compareTab').click();
   const h20=catalog.runs.find(r=>r.model==='MLP'&&r.method==='H20'&&r.task===2&&r.init===2&&r.seed===1000&&!r.forced);
   await page.locator('#pairSelect').selectOption(h20.id);await page.waitForFunction(()=>document.querySelector('#labelB').textContent==='H20 / MLP'&&document.querySelector('#videoB').readyState>=2);
   await jump(65);await page.waitForFunction(()=>Math.abs(document.querySelector('#videoA').currentTime-3.251)<.01&&Math.abs(document.querySelector('#videoB').currentTime-3.251)<.01);
   assert.equal(await page.locator('#videoA').evaluate(v=>v.controls),false);assert((await page.locator('#syncBadge').textContent()).includes('主选已同步 · 对照已同步'));
   await page.screenshot({path:path.join(out,'review_baseline_pair.png'),fullPage:true});results.checks.push('six genuine H50/H20/MLP+OGG baseline preset; H50/H20 dual video seeks to recorded frame65 PTS3.25');
   await page.locator('#clearFilters').click();await page.locator('#modelFilter').selectOption('A10-H1');await page.locator('#methodFilter').selectOption('ogg');await page.locator('#taskFilter').selectOption('1');await page.locator('#initFilter').selectOption('0');await page.locator('#seedFilter').selectOption('1000');
   await page.locator(`.run-item[data-id="${originalA10.id}"]`).click();await page.waitForFunction(()=>document.querySelector('#labelA').textContent==='ogg / A10-H1'&&document.querySelector('#videoA').readyState>=2);
   await jump(65);await page.waitForFunction(()=>Math.abs(document.querySelector('#videoA').currentTime-3.251)<.01);
   assert.equal(await page.locator('#videoA').evaluate(v=>v.controls),false);assert((await page.locator('#stepA').textContent()).includes('视频 step 65'));
   await page.screenshot({path:path.join(out,'a10_verified_binding.png'),fullPage:true});results.checks.push('original formal A10/OGG recording independently rebound and synchronized; original Hybrid C proof retained');
   await page.locator('#clearFilters').click();await page.locator('#modelFilter').selectOption('T-H1');await page.locator('#methodFilter').selectOption('H20');await page.locator('#taskFilter').selectOption('9');await page.locator('#initFilter').selectOption('2');await page.locator('#seedFilter').selectOption('1000');
   const unbound=catalog.runs.find(r=>r.model==='T-H1'&&r.method==='H20'&&r.task===9&&r.init===2&&r.seed===1000&&!r.forced);
   await page.locator(`.run-item[data-id="${unbound.id}"]`).click();await page.waitForFunction(()=>document.querySelector('#labelA').textContent==='H20 / T-H1'&&document.querySelector('#videoA').readyState>=2);
   await jump(65);assert.equal(await page.locator('#videoA').evaluate(v=>v.controls),true);
   await page.waitForFunction(()=>document.querySelector('#stepA').textContent.includes('数据 step 65 · 视频独立'));
   assert(await page.locator('#videoA').evaluate(v=>v.currentTime<.01));
   const verifiedPair=catalog.runs.find(r=>r.model==='MLP'&&r.method==='H20'&&r.task===9&&r.init===2&&r.seed===1000&&!r.forced);
   await page.locator('#pairSelect').selectOption(verifiedPair.id);await page.waitForFunction(()=>document.querySelector('#labelB').textContent==='H20 / MLP'&&document.querySelector('#videoB').readyState>=2);
   await jump(66);await page.waitForFunction(()=>Math.abs(document.querySelector('#videoB').currentTime-3.301)<.01);
   assert((await page.locator('#syncBadge').textContent()).includes('主选视频独立 · 对照已同步'));
   assert(await page.locator('#videoA').evaluate(v=>v.currentTime<.01));await page.screenshot({path:path.join(out,'unbound_alias_independent.png'),fullPage:true});
   results.checks.push('unbound equal-SHA alias remains independent and explicitly labels data step; verified comparison alone follows common slider');
  }
  for(const viewport of [{width:1920,height:1080},{width:1366,height:768},{width:900,height:900}]){
   await page.setViewportSize(viewport);await page.goto(base);await page.waitForFunction(()=>document.querySelector('#videoA').readyState>=2);assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:path.join(out,`viewport_${viewport.width}.png`)});
   if(viewport.width===1366){await page.locator('#focusBtn').click();const plot=await page.locator('#plot').boundingBox();assert(plot.y+plot.height<=viewport.height);await page.screenshot({path:path.join(out,'focused_laptop.png')});results.checks.push('laptop focus mode keeps real video and full score plot in same viewport');}
  }
  assert.deepEqual(results.errors,[]);results.checks.push('1920x1080, 1366x768, 900x900 without horizontal overflow or browser errors');results.status='passed';
 }catch(error){results.status='failed';results.failure=error.stack;throw error;}
 finally{fs.writeFileSync(path.join(out,'browser_acceptance.json'),JSON.stringify(results,null,2));console.log(JSON.stringify(results,null,2));await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
