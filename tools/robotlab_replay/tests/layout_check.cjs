const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.REPLAY_PLAYWRIGHT||'playwright');
const base=process.env.REPLAY_URL||'http://127.0.0.1:8871',key='robotlab-replay.workspace.v1';
const out=path.resolve(__dirname,'../.local/review');fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.REPLAY_BROWSER?{executablePath:process.env.REPLAY_BROWSER}:{}),args:['--no-sandbox']});
 const result={checks:[],errors:[]};
 try{
  const context=await browser.newContext({viewport:{width:1920,height:1080}}),page=await context.newPage();
  page.on('pageerror',e=>result.errors.push(e.message));page.on('console',m=>{if(m.type()==='error')result.errors.push(m.text());});
  const panel=id=>page.locator(`.workspace-panel[data-panel="${id}"]`);
  const jump=async n=>{await page.locator('#stepInput').fill(String(n));await page.locator('#stepInput').press('Tab');await page.waitForFunction(n=>Number(document.querySelector('#clockStep').textContent)===n,n);};
  const move=async(locator,dx,dy)=>{const r=await locator.boundingBox();await page.mouse.move(r.x+r.width*.4,r.y+r.height*.5);await page.mouse.down();await page.mouse.move(r.x+r.width*.4+dx,r.y+r.height*.5+dy,{steps:12});await page.mouse.up();};
  await page.goto(base);await page.waitForFunction(()=>document.querySelector('#videoA').readyState>=2);await jump(61);
  await page.evaluate(()=>window.originalReplayVideo=document.querySelector('#videoA'));
  await page.locator('#workspaceToggle').click();await page.waitForSelector('body.free-workspace');
  assert.equal(await page.locator('.workspace-panel:visible').count(),4);assert(await page.evaluate(()=>document.querySelector('#videoA')===window.originalReplayVideo));assert.equal(await page.locator('#videoA').count(),1);
  await page.waitForFunction(()=>Math.abs(document.querySelector('#videoA').currentTime-3.051)<.01);assert.equal(Number(await page.locator('#clockStep').textContent()),61);
  const replay=await panel('replay').boundingBox(),timeline=await panel('timeline').boundingBox(),inspector=await panel('inspector').boundingBox();
  assert(timeline.y+timeline.height<=1081);assert(replay.x+replay.width<=inspector.x+2);
  result.checks.push('real video DOM and step61/PTS3.05 preserved when entering four-panel workspace');
  const before=await panel('replay').boundingBox();await page.locator('#play').click();await page.waitForTimeout(120);await move(panel('replay').locator('.workspace-handle'),34,25);assert.equal(await page.locator('#play').textContent(),'Ⅱ');assert(Number(await page.locator('#clockStep').textContent())>61);await page.locator('#play').click();await jump(61);const moved=await panel('replay').boundingBox();assert(moved.x>before.x+25&&moved.y>before.y+18);
  await move(panel('replay').locator('.workspace-resize'),-110,-65);const resized=await panel('replay').boundingBox();assert(resized.width<moved.width-90&&resized.height<moved.height-45);
  await panel('replay').locator('.workspace-handle').focus();await page.keyboard.press('ArrowRight');const keyed=await panel('replay').boundingBox();assert(keyed.x>=resized.x+10);assert.equal(Number(await page.locator('#clockStep').textContent()),61);
  await page.keyboard.press('Shift+ArrowDown');assert((await panel('replay').boundingBox()).height>resized.height+9);
  result.checks.push('pointer drag/resize and keyboard movement work without seeking the shared clock');
  await page.locator('#workspaceLock').check();const locked=await panel('replay').boundingBox();await move(panel('replay').locator('.workspace-handle'),55,30);assert.deepEqual(await panel('replay').boundingBox(),locked);await page.locator('#workspaceLock').uncheck();
  await panel('replay').locator('[aria-label="最大化操作回放与时间轴"]').click();assert.equal(await page.locator('.workspace-panel:visible').count(),1);await page.keyboard.press('Escape');assert.equal(await page.locator('.workspace-panel:visible').count(),4);
  await panel('replay').locator('[aria-label="最大化操作回放与时间轴"]').click();await page.locator('#workspaceManage').click();await page.locator('#workspacePanelList input[data-panel="events"]').check();await page.locator('#workspaceDialogClose').click();assert.equal(await page.locator('.workspace-panel:visible').count(),5);await panel('events').locator('[aria-label="隐藏关键事件"]').click();
  const position=await panel('replay').boundingBox();await panel('inspector').locator('.workspace-body').hover();await page.mouse.wheel(0,550);await page.waitForTimeout(150);assert.equal(await page.evaluate(()=>scrollY),0);assert.deepEqual(await panel('replay').boundingBox(),position);
  result.checks.push('locked layout, maximize/Escape restore and independent inspector scrolling retain visible video');
  await page.locator('#workspaceManage').click();await page.locator('#workspaceName').fill('抓取复盘');await page.locator('#workspaceSave').click();const savedId=await page.locator('#workspacePreset').inputValue();assert(savedId.startsWith('user_'));
  await page.locator('#workspacePanelList input[data-panel="events"]').check();await page.locator('#workspaceDialogClose').click();assert(await panel('events').isVisible());
  await panel('events').locator('[aria-label="隐藏关键事件"]').click();assert(!(await panel('events').isVisible()));
  const geometry=await panel('replay').boundingBox();await page.reload();await page.waitForFunction(()=>document.querySelector('#videoA').readyState>=2);assert(await page.locator('body').evaluate(b=>b.classList.contains('free-workspace')));assert.deepEqual(await panel('replay').boundingBox(),geometry);assert.equal(await page.locator('#workspacePreset').inputValue(),savedId);
  result.checks.push('module hide/show and named layout survive reload in browser-local storage');
  await page.locator('#workspacePreset').selectOption('heatmap');await jump(61);await page.waitForSelector('#updateHeatmap svg');assert.equal(await page.locator('.workspace-panel:visible').count(),3);assert(await panel('replay').isVisible());assert(await panel('evidence').isVisible());assert.equal(await page.locator('#updateHeatmap rect').count(),200);
  const clip=await panel('evidence').boundingBox();assert(clip.y+clip.height<=1081);await page.screenshot({path:path.join(out,'workspace_heatmap_1920.png')});
  result.checks.push('video/curve/real A10 update heatmap preset displays live linked modules on one viewport');
  await page.locator('#workspacePreset').selectOption('compare');assert((await panel('comparison').locator('.workspace-empty').textContent()).includes('配对'));await page.locator('#compareTab').click();await page.waitForFunction(()=>document.querySelector('#videoB').readyState>=2);assert(!(await panel('comparison').locator('.workspace-empty').isVisible()));await jump(61);
  await page.waitForFunction(()=>Math.abs(document.querySelector('#videoA').currentTime-3.051)<.01&&Math.abs(document.querySelector('#videoB').currentTime-3.051)<.01);assert((await page.locator('#pairTable').innerText()).includes('168'));await page.screenshot({path:path.join(out,'workspace_compare_1920.png')});
  result.checks.push('comparison preset responds to real pair selection; two videos and metrics remain linked');
  for(const viewport of [{width:1366,height:768},{width:900,height:900}]){await page.setViewportSize(viewport);await page.waitForTimeout(180);const c=await page.locator('#workspaceCanvas').boundingBox();for(const p of await page.locator('.workspace-panel:visible').all()){const b=await p.boundingBox();assert(b.x>=-.5&&b.y>=c.y-.5&&b.x+b.width<=viewport.width+.5&&b.y+b.height<=viewport.height+.5);}assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:path.join(out,`workspace_${viewport.width}.png`)});}
  result.checks.push('all visible panels remain reachable within 1366x768 and 900x900 viewports');
  await page.setViewportSize({width:1920,height:1080});await page.locator('#workspaceManage').click();for(const checkbox of await page.locator('#workspacePanelList input').all())await checkbox.uncheck();await page.locator('#workspaceDialogClose').click();assert(await page.locator('#workspaceEmpty').isVisible());await page.locator('#workspaceEmptyOpen').click();await page.locator('#workspacePanelList input[data-panel="replay"]').check();await page.locator('#workspaceDialogClose').click();assert(await panel('replay').isVisible());
  await page.locator('#workspaceManage').click();await page.locator('#workspaceShowAll').click();assert.equal(await page.locator('.workspace-panel:visible').count(),9);assert(await page.locator('#evidenceDetails').evaluate(d=>d.open));await page.locator('#workspaceDialogClose').click();
  await page.locator('#workspaceToggle').click();assert(!(await page.locator('body').evaluate(b=>b.classList.contains('free-workspace'))));assert(await page.locator('main .playback-card').isVisible());assert.equal(await page.locator('#videoA').count(),1);assert.equal(Number(await page.locator('#clockStep').textContent()),61);
  await page.waitForFunction(()=>Math.abs(document.querySelector('#videoA').currentTime-3.051)<.01);await page.reload();await page.waitForFunction(()=>document.querySelector('#videoA').readyState>=2);assert(!(await page.locator('body').evaluate(b=>b.classList.contains('free-workspace'))));
  result.checks.push('hide-all recovery and return to original page preserve DOM/timebase; page-mode choice persists');
  const badContext=await browser.newContext();await badContext.addInitScript(k=>localStorage.setItem(k,'not JSON'),key);const badPage=await badContext.newPage();await badPage.goto(base);await badPage.waitForFunction(()=>document.querySelector('#videoA').readyState>=2);assert(!(await badPage.locator('body').evaluate(b=>b.classList.contains('free-workspace'))));await badContext.close();
  const blockedContext=await browser.newContext();await blockedContext.addInitScript(()=>{Storage.prototype.setItem=function(){throw new DOMException('blocked','SecurityError');};});const blocked=await blockedContext.newPage();await blocked.goto(base);await blocked.waitForFunction(()=>document.querySelector('#videoA').readyState>=2);await blocked.locator('#workspaceToggle').click();assert((await blocked.locator('#workspaceStatus').textContent()).includes('存储不可用'));assert(await blocked.locator('#workspaceCanvas').isVisible());await blockedContext.close();
  result.checks.push('corrupt/blocked browser storage gracefully falls back without breaking replay');assert.deepEqual(result.errors,[]);result.status='passed';
 }catch(e){result.status='failed';result.failure=e.stack;throw e;}
 finally{fs.writeFileSync(path.join(out,'layout_acceptance.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
