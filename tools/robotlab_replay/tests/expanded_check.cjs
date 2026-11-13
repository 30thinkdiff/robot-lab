/* Real new J/Plus acceptance; starts no server and performs no GPU work. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.REPLAY_PLAYWRIGHT||'playwright');
const base=process.env.REPLAY_URL||'http://127.0.0.1:8872';
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.REPLAY_BROWSER,args:['--no-sandbox']});
 const errors=[],checks=[];
 try {
  const context=await browser.newContext({viewport:{width:1600,height:1000}}),page=await context.newPage();
  page.on('pageerror',e=>errors.push(e.message));
  const catalog=(await(await context.request.get(base+'/api/catalog')).json()).runs;
  const j=catalog.find(r=>r.method==='j_ogg'&&r.model==='J-36M'&&r.task===7&&r.init===2&&r.seed===1000);
  const mlp=catalog.find(r=>r.method==='ogg'&&r.model==='MLP'&&r.task===7&&r.init===2&&r.seed===1000&&r.experiment==='Object10 五模型');
  assert(j&&mlp&&j.success&&!mlp.success&&j.video_available&&mlp.video_available);
  await page.goto(base);await page.locator('#modelFilter option[value="J-36M"]').waitFor({state:'attached'});
  await page.locator('#modelFilter').selectOption('J-36M');await page.locator('#methodFilter').selectOption('j_ogg');
  await page.locator('#taskFilter').selectOption('7');await page.locator('#initFilter').selectOption('2');
  await page.locator('#seedFilter').selectOption('1000');await page.locator(`.run-item[data-id="${j.id}"]`).click();
  await page.waitForFunction(()=>document.querySelector('#videoA').readyState>=2);
  await page.locator('#compareTab').click();await page.locator('#pairSelect').selectOption(mlp.id);
  await page.waitForFunction(()=>document.querySelector('#videoB').readyState>=2);
  await page.locator('#seek').evaluate(e=>{e.value='75';e.dispatchEvent(new Event('input',{bubbles:true}))});
  await page.waitForFunction(()=>['A','B'].every(k=>Math.abs(document.querySelector('#video'+k).currentTime-3.75)<.02));
  assert((await page.locator('#labelA').innerText()).includes('J-36M'));
  checks.push('real J success vs original MLP failure; both videos seek to step75/PTS3.75');
  const out=path.resolve(__dirname,'../.local/review');fs.mkdirSync(out,{recursive:true});
  await page.screenshot({path:path.join(out,'j_mlp_contrast_20261113.png')});
  const plus=catalog.find(r=>r.method==='transformer-a10-h1_ogg'&&r.condition==='plus/default'&&r.seed===9102);
  assert(plus);const detail=await(await context.request.get(base+'/api/run?id='+plus.id)).json();
  assert.equal(detail.coverage.max_gap_steps,19);assert.equal(detail.video.alignment,'recording_verified');
  assert(detail.events.some(e=>e.kind==='ogg'&&typeof e.data.gradient_l2==='number'));
  await page.locator('#clearFilters').click();await page.locator('#search').fill('Plus');
  await page.locator(`.run-item[data-id="${plus.id}"]`).click();
  await page.waitForFunction(()=>document.querySelector('#labelA').textContent.includes('A10-H1'));
  checks.push('actual Plus A10 score gaps, OGG loss/gradient records and video indexed');
  assert.deepEqual(errors,[]);fs.writeFileSync(path.join(out,'expanded_acceptance_20261113.json'),JSON.stringify({checks,errors},null,2));
  console.log(JSON.stringify({checks,errors}));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
