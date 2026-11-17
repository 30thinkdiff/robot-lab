const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.REPLAY_PLAYWRIGHT||'playwright');
const base=process.env.REPLAY_URL||'http://127.0.0.1:8874';
const out=path.resolve(__dirname,'../.local/review/motion');fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.REPLAY_BROWSER,args:['--no-sandbox']});
 const result={checks:[],errors:[]};
 try{
  const page=await browser.newPage({viewport:{width:1920,height:1080}});page.on('pageerror',e=>result.errors.push(e.message));
  await page.goto(base);await page.waitForSelector('#runList button');
  const catalog=await (await page.request.get(base+'/api/catalog')).json();
  const runs=catalog.runs.filter(r=>r.experiment==='动作跟踪诊断 · 新采集');assert.equal(runs.length,12);
  const a=runs.find(r=>r.task===1&&r.method==='j_ogg'),b=runs.find(r=>r.task===1&&r.method==='H20');
  await page.locator('#experimentFilter').selectOption(a.experiment);await page.locator(`[data-id="${a.id}"]`).click();
  await page.waitForFunction(()=>!document.querySelector('#motionContent').hidden);
  const jump=async n=>{await page.locator('#stepInput').fill(String(n));await page.locator('#stepInput').press('Tab');await page.waitForFunction(n=>Number(document.querySelector('#clockStep').textContent)===n,n);};
  await jump(65);await page.waitForFunction(()=>Math.abs(document.querySelector('#videoA').currentTime-3.251)<.02);
  const detail=await (await page.request.get(base+'/api/run?id='+a.id)).json();
  assert.equal(JSON.parse(await page.locator('#motionRaw').textContent()).step,65);
  assert((await page.locator('#motionSummary').textContent()).includes((detail.motion.rows[65].tracking_error.position_m*1000).toFixed(2)));
  assert((await page.locator('#motionPhase').textContent()).includes('before'));
  result.checks.push('real failed J-OGG step65/videoPTS3.25 and recorded OSC error synchronized');
  await page.locator('#next').click();assert.equal(JSON.parse(await page.locator('#motionRaw').textContent()).step,66);
  await page.locator('#play').click();await page.waitForTimeout(400);await page.locator('#play').click();assert(JSON.parse(await page.locator('#motionRaw').textContent()).step>66);
  await page.locator('#motionProjection').selectOption('xz');await page.locator('#motionView').selectOption('rotation');
  assert((await page.locator('#motionSeries').innerHTML()).includes('朝向误差'));
  await page.locator('#motionSeries').scrollIntoViewIfNeeded();const svg=await page.locator('#motionSeries svg').boundingBox();await page.mouse.click(svg.x+svg.width*.45,svg.y+svg.height*.5);
  assert(JSON.parse(await page.locator('#motionRaw').textContent()).step>90);
  result.checks.push('single step, play/pause, projections, angular error and click-to-seek share one cursor');
  await page.locator('#compareTab').click();await page.locator('#pairSelect').selectOption(b.id);await page.waitForFunction(()=>document.querySelector('#labelB').textContent.includes('H20'));
  await page.locator('#inspectLane').selectOption('b');await jump(65);assert((await page.locator('#motionPosition').textContent()).includes('H20'));
  await jump(180);await page.waitForFunction(()=>document.querySelector('#motionMissing').textContent.includes('超出'));assert(await page.locator('#motionMissing').isVisible());assert((await page.locator('#motionMissing').textContent()).includes('超出'));
  await page.locator('#motionLane').selectOption('a');assert(await page.locator('#motionContent').isVisible());
  result.checks.push('paired lane uses own recorded state and stops at ended H20; no stale pose');
  await page.locator('#workspaceToggle').click();await page.locator('#workspacePreset').selectOption('motion');
  assert(await page.locator('.workspace-panel[data-panel="motion"]').isVisible());
  for(const viewport of [{width:1920,height:1080},{width:1366,height:768}]){
   await page.setViewportSize(viewport);await page.waitForTimeout(150);assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
   await page.screenshot({path:path.join(out,`motion_${viewport.width}.png`)});
  }
  result.checks.push('motion workspace keeps video/monitor/motion simultaneously visible at desktop and laptop widths');
  const exported=await (await page.request.post(base+'/api/export',{data:{ids:[a.id],step:65,note:'真实运动联动验收'}})).json();
  assert(exported.files.includes('A_motion_step.json'));assert(exported.files.includes('A_motion_errors.csv'));
  const recorded=await (await page.request.get(base+`/api/export-file/${exported.export}/A_motion_step.json`)).json();
  assert.equal(recorded.motion_step.step,65);assert.equal(recorded.source.motion.verified,true);
  result.checks.push('derived export includes selected real state, whole-episode errors and source hashes');
  await page.locator('#workspaceToggle').click();
  const losses=catalog.runs.filter(r=>r.experiment.startsWith('OGG Loss · '));assert.equal(losses.length,24);
  const loss=losses.find(r=>r.experiment==='OGG Loss · ordinary · mse'&&r.method==='mlp_ogg'&&r.task===1);
  await page.locator('#experimentFilter').selectOption(loss.experiment);await page.locator(`[data-id="${loss.id}"]`).click();
  await page.waitForFunction(()=>document.querySelector('#configurationText').textContent.includes('原始特征MSE'));
  assert((await page.locator('#motionMissing').textContent()).includes('未记录'));
  await page.locator('#eventFilter').selectOption('ogg');await page.locator('#events .event-row').first().click();
  assert((await page.locator('#stepEvents').textContent()).includes('梯度范数'));
  result.checks.push('24 imported loss trials show explicit MSE objective and real guidance events; absent motion remains absent');
  assert.deepEqual(result.errors,[]);result.status='passed';
 }catch(e){result.status='failed';result.failure=e.stack;throw e;}
 finally{fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
