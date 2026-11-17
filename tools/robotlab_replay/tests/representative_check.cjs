/* Real selected cases. Runs against a separately started local Replay server. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.REPLAY_PLAYWRIGHT||'playwright');
const base=process.env.REPLAY_URL||'http://127.0.0.1:8874';
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.REPLAY_BROWSER,args:['--no-sandbox']});
 const errors=[],checks=[];
 try{
  const context=await browser.newContext({viewport:{width:1600,height:1000}}),page=await context.newPage();
  page.on('pageerror',e=>errors.push(e.message));await page.goto(base);
  await page.locator('#casebankOpen').click();await page.locator('#casebankDialog').waitFor({state:'visible'});
  await page.locator('#casebankPreset').selectOption('joint_mlp');
  await page.locator('#casebankStudy').selectOption('reviewed_ordinary_j_mlp_20261114');
  assert.equal(await page.locator('#casebankRows tr').count(),3);
  assert((await page.locator('#casebankHead').innerText()).includes('J-36M'));
  await page.locator('#casebankComplete').uncheck();assert.equal(await page.locator('#casebankRows tr').count(),4);
  assert((await page.locator('#casebankRows').innerText()).includes('视频缺'));
  checks.push('ordinary four observed patterns; three complete groups; missing MLP video explicitly shown');
  await page.locator('#casebankComplete').check();
  await page.locator('#casebankStudy').selectOption('replay_casebank_representative_plus_20261114');
  assert.equal(await page.locator('#casebankRows tr').count(),8);
  await page.locator('#casebankPattern').selectOption('0011');
  assert.equal(await page.locator('#casebankRows tr').count(),1);
  await page.locator('#casebankFrom').selectOption('j_ogg');await page.locator('#casebankTo').selectOption('mlp_ogg');
  await page.locator('#casebankRows button').first().click();
  await page.waitForFunction(()=>['A','B'].every(k=>document.querySelector('#video'+k).readyState>=2));
  assert((await page.locator('#configurationText').textContent()).includes('Block 8'));
  await page.locator('#seek').evaluate(e=>{e.value='60';e.dispatchEvent(new Event('input',{bubbles:true}))});
  await page.waitForFunction(()=>['A','B'].every(k=>Math.abs(document.querySelector('#video'+k).currentTime-3)<.02));
  const catalog=(await(await context.request.get(base+'/api/catalog')).json()).runs;
  const j=catalog.find(r=>r.method==='j_ogg'&&r.condition==='plus/cheese/sensor_noise/1'&&r.seed===1000);
  const mlp=catalog.find(r=>r.method==='mlp_ogg'&&r.condition===j.condition&&r.seed===j.seed);
  assert(j&&!j.success&&mlp.success);
  const detail=await(await context.request.get(base+'/api/run?id='+j.id)).json();
  assert(detail.events.some(e=>e.kind==='ogg'&&typeof e.data.gradient_l2==='number'));
  assert.equal(detail.video.alignment,'recording_verified');
  checks.push('real Plus J failure versus MLP success; both videos seek step60/PTS3.0; OGG gradient retained');
  const out=path.resolve(__dirname,'../.local/review');fs.mkdirSync(out,{recursive:true});
  await page.screenshot({path:path.join(out,'representative_plus_contrast.png')});
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(out,'representative_acceptance.json'),JSON.stringify({checks,errors},null,2));
  console.log(JSON.stringify({checks,errors}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
