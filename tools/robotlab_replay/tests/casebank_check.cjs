const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.REPLAY_PLAYWRIGHT||'playwright');
const base=process.env.REPLAY_URL||'http://127.0.0.1:8871',out=path.resolve(__dirname,'../.local/review');fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.REPLAY_BROWSER?{executablePath:process.env.REPLAY_BROWSER}:{}),args:['--no-sandbox']});
 const result={checks:[],errors:[]};
 try{
  const context=await browser.newContext({viewport:{width:1920,height:1080}}),page=await context.newPage();page.on('pageerror',e=>result.errors.push(e.message));
  await page.goto(base);await page.waitForFunction(()=>document.querySelector('#videoA').readyState>=2);
  const bank=await(await context.request.get(base+'/api/casebank')).json();assert.equal(bank.groups.filter(g=>g.study==='legacy_five_models').length,60);
  await page.locator('#casebankOpen').click();await page.locator('#casebankDialog').waitFor({state:'visible'});await page.locator('#casebankStudy').selectOption('legacy_five_models');
  await page.locator('#casebankPattern').selectOption('001');assert.equal(await page.locator('#casebankRows tr').count(),3);
  const row=await page.locator('#casebankRows tr').first().innerText();assert(row.includes('失败'));assert(row.includes('成功'));assert(row.includes('视频缺'));
  await page.screenshot({path:path.join(out,'casebank_original_001.png'),fullPage:true});result.checks.push('real 60-condition inventory and three H50-fail/H20-fail/MLP-success cases, missing old success video explicitly labeled');
  await page.locator('#casebankRows tr').first().locator('button[data-base="H20"]').click();await page.waitForFunction(()=>document.querySelector('#labelA').textContent==='H20 / MLP'&&document.querySelector('#labelB').textContent==='ogg / MLP');
  assert((await page.locator('#pairTable').innerText()).includes('任务失败'));assert((await page.locator('#pairTable').innerText()).includes('任务成功'));assert((await page.locator('#pairNotice').textContent()).includes('配对')||(await page.locator('#pairNotice').textContent()).includes('SHA'));
  result.checks.push('chosen case opens exact condition/seed H20 versus original MLP Corrector without mixing phases');
  await page.locator('#casebankOpen').click();await page.locator('#casebankComplete').check();assert((await page.locator('#casebankRows').textContent()).includes('没有满足条件'));await page.locator('#casebankComplete').uncheck();
  await page.locator('#casebankReference').selectOption('transformer-a10-h1_ogg');await page.locator('#casebankPattern').selectOption('');
  const original=bank.groups.filter(g=>g.study==='legacy_five_models');const expected=original.filter(g=>!g.methods.H50.success&&!g.methods.H20.success&&g.methods['transformer-a10-h1_ogg'].success).length;
  assert.equal(Number(await page.locator('#casebankCounts [data-pattern="001"] strong').textContent()),expected);
  result.checks.push('completeness filter excludes incomplete historical data; changing reference recomputes actual A10 outcomes');
  const newStudy=bank.groups.find(g=>g.study!=='legacy_five_models')?.study;
  if(newStudy){await page.locator('#casebankStudy').selectOption(newStudy);await page.locator('#casebankReference').selectOption('mlp_ogg');await page.locator('#casebankComplete').check();
   const groups=bank.groups.filter(g=>g.study===newStudy&&['H50','H20','mlp_ogg'].every(m=>g.methods[m]?.replay_complete));assert(groups.length>=4);
   const patterns=new Set(groups.map(g=>['H50','H20','mlp_ogg'].map(m=>g.methods[m].success?'1':'0').join('')));for(const p of ['001','000','011','111'])assert(patterns.has(p));
   await page.locator('#casebankPattern').selectOption('001');await page.locator('#casebankRows tr').first().locator('button[data-base="H50"]').click();
   await page.waitForFunction(()=>document.querySelector('#labelA').textContent==='H50 / SmolVLA'&&document.querySelector('#labelB').textContent==='mlp_ogg / MLP'&&document.querySelector('#videoA').readyState>=2&&document.querySelector('#videoB').readyState>=2);
   await page.locator('#stepInput').fill('61');await page.locator('#stepInput').press('Tab');await page.waitForFunction(()=>Math.abs(document.querySelector('#videoA').currentTime-3.051)<.01&&Math.abs(document.querySelector('#videoB').currentTime-3.051)<.01);
   await page.locator('#inspectLane').selectOption('b');assert((await page.locator('#readout').textContent()).includes('实测'));assert((await page.locator('#actions').textContent()).includes('提交给环境'));
   await page.screenshot({path:path.join(out,'casebank_new_complete_pair.png'),fullPage:true});result.checks.push('new full-capture bank covers all four requested combinations; real paired videos seek to step61/PTS3.05 and show measured queue/actions');
  }
  if(await page.locator('#casebankDialog').isVisible())await page.locator('#casebankClose').click();
  for(const viewport of [{width:1366,height:768},{width:900,height:900}]){await page.setViewportSize(viewport);await page.locator('#casebankOpen').click();assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.locator('#casebankClose').click();}
  assert.deepEqual(result.errors,[]);result.checks.push('sample library remains usable at laptop widths without browser errors');result.status='passed';
 }catch(e){result.status='failed';result.failure=e.stack;throw e;}
 finally{fs.writeFileSync(path.join(out,'casebank_acceptance.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
